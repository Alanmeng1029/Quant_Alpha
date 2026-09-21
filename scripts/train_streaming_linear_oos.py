"""Rolling Ridge/Elastic-Net using Rust-built temporary factor windows."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

TRAIN_DAYS = 756
LABEL_LAG = 11
VALID_DAYS = 63
VALID_PURGE_DAYS = 11
HORIZONS = ("h1", "h5", "h10")
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ELASTIC_ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
ELASTIC_L1_RATIO = 0.5


def calendar(catalog: Path) -> list[str]:
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        return [str(x[0]) for x in conn.execute(
            "SELECT trade_date FROM observed_calendar WHERE is_observed_market_day ORDER BY trade_date"
        ).fetchall()]
    finally:
        conn.close()


def windows(dates: list[str], start: str, end: str) -> list[tuple[int, int]]:
    start_month = int(start[:4]) * 12 + int(start[5:7]) - 1
    result = []
    for i, date in enumerate(dates):
        if date < start or date > end or i < TRAIN_DAYS + LABEL_LAG:
            continue
        first = i == 0 or dates[i - 1][:7] != date[:7]
        month = int(date[:4]) * 12 + int(date[5:7]) - 1
        if not first or (month - start_month) % 3:
            continue
        finish = i
        while finish < len(dates) and dates[finish] <= end:
            value = int(dates[finish][:4]) * 12 + int(dates[finish][5:7]) - 1
            if value >= month + 3:
                break
            finish += 1
        if finish > i:
            result.append((i, finish))
    return result


def gram(x: np.ndarray, y: np.ndarray, mask: np.ndarray, chunk: int = 32768):
    indices = np.flatnonzero(mask)
    p = x.shape[1]
    g = np.zeros((p, p), dtype=np.float64)
    xy = np.zeros(p, dtype=np.float64)
    y_mean = float(np.mean(y[indices]))
    yy = 0.0
    for offset in range(0, len(indices), chunk):
        take = indices[offset:offset + chunk]
        xb = np.asarray(x[take], dtype=np.float64)
        yb = y[take] - y_mean
        g += xb.T @ xb
        xy += xb.T @ yb
        yy += float(yb @ yb)
    return g / len(indices), xy / len(indices), yy / len(indices), y_mean, len(indices)


def ridge(g: np.ndarray, xy: np.ndarray, penalty: float) -> np.ndarray:
    return np.linalg.solve(g + penalty * np.eye(len(xy)), xy)


def soft(value: float, threshold: float) -> float:
    return max(value - threshold, 0.0) - max(-value - threshold, 0.0)


def elastic(g: np.ndarray, xy: np.ndarray, alpha: float, l1_ratio: float,
            max_iter: int = 10000, tol: float = 1e-7) -> tuple[np.ndarray, int]:
    w = np.zeros(len(xy), dtype=np.float64)
    residual = xy.copy()
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    for iteration in range(1, max_iter + 1):
        largest = 0.0
        for j in range(len(w)):
            rho = residual[j] + g[j, j] * w[j]
            updated = soft(rho, l1) / (g[j, j] + l2) if g[j, j] + l2 > 1e-15 else 0.0
            delta = updated - w[j]
            if delta:
                residual -= g[:, j] * delta
                w[j] = updated
                largest = max(largest, abs(delta))
        if largest < tol:
            return w, iteration
    return w, max_iter


def mse(x: np.ndarray, y: np.ndarray, mask: np.ndarray, weights: np.ndarray,
        intercept: float, chunk: int = 32768) -> float:
    indices = np.flatnonzero(mask)
    total = 0.0
    for offset in range(0, len(indices), chunk):
        take = indices[offset:offset + chunk]
        error = np.asarray(x[take], dtype=np.float64) @ weights + intercept - y[take]
        total += float(error @ error)
    return total / len(indices)


def predict(x: np.ndarray, weights: np.ndarray, intercept: float, chunk: int = 32768) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    for offset in range(0, len(x), chunk):
        stop = min(offset + chunk, len(x))
        out[offset:stop] = np.asarray(x[offset:stop], dtype=np.float64) @ weights + intercept
    return out


def zscore_by_date(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    expressions = []
    for column in columns:
        sd = pl.col(column).std().over("trade_date")
        expressions.append(pl.when(sd > 1e-12).then(
            (pl.col(column) - pl.col(column).mean().over("trade_date")) / sd
        ).otherwise(0.0).alias(column.replace("raw_", "pred_")))
    return frame.with_columns(expressions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer", type=Path, default=Path("target/release/build_linear_window"))
    parser.add_argument("--index-code", default="000300.SH,000905.SH")
    parser.add_argument("--oos-start", default="2021-04-01")
    parser.add_argument("--oos-end", default="2026-08-28")
    parser.add_argument("--models", nargs="+", choices=("ridge", "elasticnet"), default=["ridge", "elasticnet"])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--memory-limit-mb", type=int, default=4096)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_root = args.output / "models"
    model_root.mkdir(exist_ok=True)
    dates = calendar(args.catalog)
    rolling = windows(dates, args.oos_start, args.oos_end)
    if args.max_windows:
        rolling = rolling[:args.max_windows]
    predictions = []
    coefficient_rows = []
    logs = []
    for window_index, (test_begin, test_end) in enumerate(rolling):
        train_end = test_begin - LABEL_LAG
        train_begin = train_end - TRAIN_DAYS
        signal = dates[test_begin]
        month_root = model_root / f"month={signal[:7]}"
        complete = month_root / "complete.json"
        if complete.exists():
            predictions.append(pl.read_parquet(month_root / "predictions.parquet"))
            coefficient_rows.extend(pl.read_parquet(month_root / "coefficients.parquet").to_dicts())
            saved = json.loads(complete.read_text())
            logs.append(saved)
            print(json.dumps({"window": window_index, "signal": signal, "status": "reused"}), flush=True)
            continue
        started = time.perf_counter()
        coefficient_start = len(coefficient_rows)
        with tempfile.TemporaryDirectory(prefix="quant-linear-window-") as temp_name:
            temp = Path(temp_name)
            command = [str(args.producer), "--catalog", str(args.catalog), "--daily-root", str(args.daily_root),
                       "--output", str(temp), "--train-start", dates[train_begin], "--train-end", dates[train_end - 1],
                       "--test-start", signal, "--test-end", dates[test_end - 1], "--index-code", args.index_code,
                       "--memory-limit-mb", str(args.memory_limit_mb)]
            subprocess.run(command, check=True)
            manifest = json.loads((temp / "manifest.json").read_text())
            factors = manifest["factor_ids"]
            p = manifest["feature_count"]
            metadata = pl.read_csv(temp / "rows.tsv", separator="\t", try_parse_dates=True)
            train_meta = metadata.head(manifest["train_rows"])
            test_meta = metadata.tail(manifest["test_rows"])
            all_x = np.memmap(temp / "x_col_major.f32", dtype="<f4", mode="r",
                              shape=(manifest["rows"], p), order="F")
            train_x = all_x[:manifest["train_rows"]]
            test_x = all_x[manifest["train_rows"]:]
            unique_dates = train_meta.get_column("trade_date").unique().sort().to_list()
            validation_dates = set(unique_dates[-VALID_DAYS:])
            fit_dates = set(unique_dates[:-(VALID_DAYS + VALID_PURGE_DAYS)])
            train_dates_np = train_meta.get_column("trade_date").to_numpy()
            fit_date_mask = np.isin(train_dates_np, list(fit_dates))
            valid_date_mask = np.isin(train_dates_np, list(validation_dates))
            out = test_meta.select("trade_date", "ts_code", "execution_date", "h1", "h5", "h10")
            window_log = {"window": window_index, "signal": signal, "test_end": dates[test_end - 1],
                          "train_start": dates[train_begin], "train_end": dates[train_end - 1],
                          "feature_count": p, "models": {}}
            for horizon in HORIZONS:
                y = train_meta.get_column(horizon).cast(pl.Float64).to_numpy()
                finite = np.isfinite(y)
                g_fit, xy_fit, _, mean_fit, fit_rows = gram(train_x, y, fit_date_mask & finite)
                model_outputs = {}
                if "ridge" in args.models:
                    trials = []
                    for penalty in RIDGE_LAMBDAS:
                        w = ridge(g_fit, xy_fit, penalty)
                        trials.append((mse(train_x, y, valid_date_mask & finite, w, mean_fit), penalty))
                    _, chosen = min(trials)
                    g, xy, _, intercept, rows = gram(train_x, y, finite)
                    weights = ridge(g, xy, chosen)
                    model_outputs["ridge"] = predict(test_x, weights, intercept)
                    window_log["models"][f"ridge_{horizon}"] = {"penalty": chosen, "fit_rows": rows, "validation_mse": min(trials)[0]}
                    coefficient_rows.extend({"model_month": signal[:7], "model": "ridge", "horizon": horizon,
                                             "feature": factor, "coefficient": float(value), "nonzero": True}
                                            for factor, value in zip(factors, weights))
                if "elasticnet" in args.models:
                    trials = []
                    for alpha in ELASTIC_ALPHAS:
                        w, iterations = elastic(g_fit, xy_fit, alpha, ELASTIC_L1_RATIO)
                        trials.append((mse(train_x, y, valid_date_mask & finite, w, mean_fit), alpha, iterations))
                    _, chosen, _ = min(trials)
                    g, xy, _, intercept, rows = gram(train_x, y, finite)
                    weights, iterations = elastic(g, xy, chosen, ELASTIC_L1_RATIO)
                    model_outputs["elasticnet"] = predict(test_x, weights, intercept)
                    window_log["models"][f"elasticnet_{horizon}"] = {"alpha": chosen, "l1_ratio": ELASTIC_L1_RATIO,
                        "fit_rows": rows, "iterations": iterations, "converged": iterations < 10000,
                        "nonzero": int(np.count_nonzero(weights)), "validation_mse": min(trials)[0]}
                    coefficient_rows.extend({"model_month": signal[:7], "model": "elasticnet", "horizon": horizon,
                                             "feature": factor, "coefficient": float(value), "nonzero": bool(value != 0)}
                                            for factor, value in zip(factors, weights))
                for model, values in model_outputs.items():
                    out = out.with_columns(pl.Series(f"raw_{model}_{horizon}", values))
            raw_columns = [f"raw_{model}_{horizon}" for model in args.models for horizon in HORIZONS]
            out = zscore_by_date(out, raw_columns).with_columns(pl.lit(signal[:7]).alias("model_month"))
            predictions.append(out)
            window_log["elapsed_seconds"] = time.perf_counter() - started
            logs.append(window_log)
            month_root.mkdir(exist_ok=True)
            (month_root / "manifest.json").write_text(json.dumps(window_log, indent=2) + "\n")
            out.write_parquet(month_root / "predictions.parquet", compression="zstd")
            pl.DataFrame(coefficient_rows[coefficient_start:]).write_parquet(month_root / "coefficients.parquet", compression="zstd")
            (month_root / "complete.json").write_text(json.dumps(window_log, indent=2) + "\n")
            print(json.dumps({"window": window_index, "signal": signal, "seconds": window_log["elapsed_seconds"], "models": window_log["models"]}), flush=True)
    result = pl.concat(predictions).sort("trade_date", "ts_code")
    result.write_parquet(args.output / "predictions.parquet", compression="zstd")
    pl.DataFrame(coefficient_rows).write_parquet(args.output / "coefficients.parquet", compression="zstd")
    (args.output / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({"engine": "rust-window-python-gram-v1", "models": args.models,
        "factor_count": len(factors), "training_days": TRAIN_DAYS, "label_lag": LABEL_LAG, "windows": len(logs),
        "oos_start": args.oos_start, "oos_end": args.oos_end, "daily_root": str(args.daily_root)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
