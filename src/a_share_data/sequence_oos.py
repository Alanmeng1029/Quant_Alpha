"""Orchestrate Rust sequence caching, isolated MPS fitting, and OOS comparison."""
from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import duckdb
import numpy as np
import polars as pl

from a_share_data.predict import (INFEASIBLE_EXECUTION_CODES, build_labels,
                                  standardize_features)
from a_share_data.research_oos import (_diagnostics, _execution_calendar, _metrics,
                                       _period_metrics, _registered_factor_ids,
                                       _run_backtests, block_bootstrap_difference,
                                       daily_normalize, quarter_windows)
from a_share_data.sequence_data import SequenceCache


PREPROCESSING_VERSION = "daily_cross_section_p01_p99_zscore_v1"


def _label_options(config: dict[str, Any]) -> dict[str, Any]:
    return {"universe_index_codes": tuple(config.get("universe_index_codes", ("000300.SH", "000905.SH"))),
            "raw_eligible_universe": bool(config.get("raw_eligible_universe", False))}


def _sequence_windows(dates: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    lag = int(config.get("training_label_lag", 6))
    if lag < 6:
        raise ValueError("training_label_lag must cover the H5 label's six-day maturity")
    positions = {day: index for index, day in enumerate(dates)}
    result = []
    for signal, _, test in quarter_windows(dates, config["oos_start"], config["oos_end"]):
        end = positions[signal] - lag
        if end < 756:
            continue
        result.append({"signal": signal, "train_dates": dates[end-756:end], "test_dates": test})
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _fingerprint(config: dict[str, Any], feature_manifest: dict[str, Any], files: list[Path],
                 factor_ids: tuple[str, ...], calendar: list[date]) -> str:
    payload = {
        "feature_manifest": feature_manifest,
        "feature_files": [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files],
        "factor_ids": factor_ids,
        "calendar": [str(day) for day in calendar],
        "sequence_length": int(config["sequence_length"]),
        "preprocessing": PREPROCESSING_VERSION,
    }
    # Keep the old 105-factor cache identity unchanged; new label policies must
    # never silently reuse a cache prepared under another eligibility rule.
    if "raw_eligible_universe" in config or "universe_index_codes" in config:
        payload["label_options"] = _label_options(config)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _calendar(catalog: Path) -> list[date]:
    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT trade_date FROM observed_calendar WHERE is_observed_market_day ORDER BY trade_date"
        ).fetchall()]
    finally:
        connection.close()


def _build_cache(config: dict[str, Any], root: Path) -> tuple[Path, dict[str, Any]]:
    feature_root = Path(config["feature_root"]).expanduser()
    catalog = Path(config["catalog"]).expanduser()
    formal_path = Path(config["formal_factor_set"]).expanduser()
    feature_manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    factor_ids = tuple(feature_manifest["factor_ids"])
    registered = _registered_factor_ids(formal_path, json.loads(formal_path.read_text(encoding="utf-8")))
    if factor_ids != registered:
        raise ValueError("feature cache does not exactly match the active formal factor registry")
    files = sorted(feature_root.glob("year=*/features.parquet"))
    if not files: raise FileNotFoundError("feature cache contains no yearly parquet files")
    calendar = _calendar(catalog)
    fingerprint = _fingerprint(config, feature_manifest, files, factor_ids, calendar)
    destination = root / "sequence_cache" / fingerprint[:16]
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        cache = SequenceCache(destination)
        if cache.manifest["fingerprint"] != fingerprint:
            raise ValueError("sequence cache fingerprint collision")
        return destination, cache.manifest

    root.mkdir(parents=True, exist_ok=True)
    staging_root = root / "sequence_cache" / f".{fingerprint[:16]}.staging-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"stale sequence-cache staging directory: {staging_root}")
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    try:
        features = pl.concat([pl.read_parquet(path) for path in files]).with_columns(
            pl.col("trade_date").cast(pl.Date)).filter(
            ~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
        labels = build_labels(catalog, **_label_options(config))
        day_map = pl.DataFrame({"trade_date": calendar}).with_row_index("day_index")
        panel = standardize_features(features, factor_ids).join(
            labels.select("trade_date", "ts_code", "excess_h1", "excess_h5"),
            on=["trade_date", "ts_code"], how="left").join(day_map, on="trade_date", how="left")
        panel = panel.sort(["ts_code", "day_index"]).select(
            "trade_date", "ts_code", pl.col("day_index").cast(pl.Int32),
            *[pl.col(name).cast(pl.Float32) for name in factor_ids],
            pl.col("excess_h1").cast(pl.Float32), pl.col("excess_h5").cast(pl.Float32))
        input_path = staging_root / "panel.parquet"
        factors_path = staging_root / "factor_ids.json"
        panel.write_parquet(input_path, compression="zstd")
        factors_path.write_text(json.dumps(factor_ids), encoding="utf-8")
        binary = Path(__file__).resolve().parents[2] / "target/release/quant-sequence-cache"
        if not binary.is_file():
            build_environment = os.environ.copy()
            command_line_tools = Path("/Library/Developer/CommandLineTools")
            if sys.platform == "darwin" and command_line_tools.is_dir():
                build_environment.update({
                    "SDKROOT": str(command_line_tools / "SDKs/MacOSX.sdk"),
                    "CC": str(command_line_tools / "usr/bin/clang"),
                    "AR": str(command_line_tools / "usr/bin/ar"),
                    "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER": str(command_line_tools / "usr/bin/clang"),
                })
            subprocess.run([str(Path.home() / ".cargo/bin/cargo"), "build", "--release", "-p",
                            "quant-sequence-cache"], cwd=Path(__file__).resolve().parents[2], check=True,
                           env=build_environment)
        built = staging_root / "built"
        subprocess.run([str(binary), "--input", str(input_path), "--output", str(built),
                        "--factor-ids", str(factors_path), "--sequence-length",
                        str(config["sequence_length"]), "--fingerprint", fingerprint], check=True)
        input_path.unlink(); factors_path.unlink()
        os.replace(built, destination)
    finally:
        if staging_root.exists(): shutil.rmtree(staging_root)
    cache = SequenceCache(destination)
    return destination, cache.manifest


def _coverage(cache: SequenceCache, start: str, end: str, destination: Path) -> dict[str, Any]:
    metadata = cache.metadata
    ends = np.asarray(cache.sample_rows, dtype=np.int64)
    eligible = metadata.filter((pl.col("trade_date") >= date.fromisoformat(start)) &
                               (pl.col("trade_date") <= date.fromisoformat(end)))
    samples = metadata.filter(pl.col("row_index").is_in(ends)).filter(
        (pl.col("trade_date") >= date.fromisoformat(start)) &
        (pl.col("trade_date") <= date.fromisoformat(end)))
    daily = eligible.group_by("trade_date").len().rename({"len": "eligible_rows"}).join(
        samples.group_by("trade_date").len().rename({"len": "sequence_rows"}),
        on="trade_date", how="left").with_columns(
        pl.col("sequence_rows").fill_null(0),
        (pl.col("sequence_rows").fill_null(0) / pl.col("eligible_rows")).alias("coverage"),
    ).sort("trade_date")
    destination.parent.mkdir(parents=True, exist_ok=True); daily.write_csv(destination)
    return {"eligible_rows": eligible.height, "sequence_rows": samples.height,
            "coverage": samples.height / eligible.height if eligible.height else None,
            "daily_csv": str(destination)}


def _top_overlap(left: pl.DataFrame, right: pl.DataFrame, size: int = 100) -> dict[str, Any]:
    joined = left.join(right.select("trade_date", "ts_code", "pred_h1", "pred_h5").rename(
        {"pred_h1": "r_h1", "pred_h5": "r_h5"}), on=["trade_date", "ts_code"])
    values = []
    for day, group in joined.group_by("trade_date"):
        left_top = set(group.with_columns((pl.col("pred_h1") + pl.col("pred_h5")).alias("s"))
                       .top_k(size, by="s")["ts_code"].to_list())
        right_top = set(group.with_columns((pl.col("r_h1") + pl.col("r_h5")).alias("s"))
                        .top_k(size, by="s")["ts_code"].to_list())
        values.append({"trade_date": str(day[0] if isinstance(day, tuple) else day),
                       "overlap": len(left_top & right_top) / size})
    return {"size": size, "mean_overlap": float(np.mean([row["overlap"] for row in values])),
            "daily": values}


def _csi500(catalog: Path, frame: pl.DataFrame) -> pl.DataFrame:
    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        universe = pl.from_arrow(connection.execute(
            "SELECT DISTINCT trade_date,ts_code FROM index_trading_universe WHERE index_code='000905.SH'"
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        connection.close()
    return frame.join(universe, on=["trade_date", "ts_code"], how="semi")


def _comparison_chart(backtest_root: Path, names: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for name in names:
        frame = pl.read_parquet(backtest_root / name / "portfolio_daily.parquet").sort("execution_date")
        nav = np.cumprod(1 + frame["net_return"].to_numpy())
        axis.plot(frame["execution_date"].to_list(), nav, label=name)
    axis.set_title("Common-sample CSI500 Top100 / swap3 net value")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def run_sequence_oos(config_path: Path, *, pilot: bool = False, max_windows: int | None = None,
                     prepare_only: bool = False, skip_backtests: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["output"]).expanduser(); root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(config.get("sequence_cache_root", root)).expanduser()
    cache_path, cache_manifest = _build_cache(config, cache_root)
    coverage = _coverage(SequenceCache(cache_path), config["oos_start"], config["oos_end"],
                         root / "reports/sequence_coverage.csv")
    if prepare_only:
        return {"output": str(root), "cache": str(cache_path), "coverage": coverage,
                "fingerprint": cache_manifest["fingerprint"]}
    run_root = root / ("pilot" if pilot else "full")
    run_root.mkdir(parents=True, exist_ok=True)
    dates = [str(day) for day in _calendar(Path(config["catalog"]))]
    windows = _sequence_windows(dates, config)
    model_tag = str(config.get("model_tag", cache_manifest["factor_count"]))
    lstm_name, baseline_name = f"lstm{model_tag}", f"lgbm_default{model_tag}_common"
    windows_path = run_root / "windows.json"; _write_json(windows_path, windows)
    effective_windows = 1 if pilot else max_windows
    effective_epochs = 2 if pilot else int(config.get("max_epochs", 100))
    trainer_python = Path(config.get("trainer_python", sys.executable)).expanduser()
    if not trainer_python.is_file(): raise FileNotFoundError(f"trainer Python does not exist: {trainer_python}")
    command = [str(trainer_python), "-m", "a_share_data.sequence_train", "--cache", str(cache_path),
               "--windows", str(windows_path), "--output", str(run_root / "lstm"),
               "--batch-size", str(config.get("batch_size", 512)), "--max-epochs", str(effective_epochs),
               "--patience", str(config.get("patience", 10)), "--learning-rate",
               str(config.get("learning_rate", .001)), "--weight-decay",
               str(config.get("weight_decay", 0.0)), "--dropout", str(config.get("dropout", 0.0)),
               "--hidden-sizes", *[str(value) for value in config.get("hidden_sizes", [128, 64])],
               "--seed", str(config.get("seed", 20260908))]
    for key, flag in (("recurrent_residual", "--recurrent-residual"),
                      ("output_residual", "--output-residual"),
                      ("zero_init_output_residual", "--zero-init-output-residual")):
        if config.get(key, False): command.append(flag)
    if effective_windows: command += ["--max-windows", str(effective_windows)]
    source_root = str(Path(__file__).resolve().parents[1])
    training_environment = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "PYTHONPATH": source_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run(command, check=True, env=training_environment)

    raw = pl.read_parquet(run_root / "lstm/raw_predictions.parquet")
    # The final observed market day has no known next trading day and therefore
    # cannot be turned into an executable signal yet. Keep its raw prediction in
    # the training artifact, but exclude it from OOS evaluation and backtests.
    last_observed_day = date.fromisoformat(dates[-1])
    non_executable_rows = raw.filter(pl.col("trade_date") >= last_observed_day).height
    raw = raw.filter(pl.col("trade_date") < last_observed_day)
    execution = _execution_calendar(Path(config["catalog"]), raw["trade_date"].unique().sort().to_list())
    lstm = daily_normalize(raw, "raw_h1", "raw_h5").join(execution, on="trade_date", how="left").drop_nulls("execution_date")
    prediction_root = run_root / "predictions"; prediction_root.mkdir(exist_ok=True)
    lstm_path = prediction_root / f"{lstm_name}.parquet"; lstm.write_parquet(lstm_path, compression="zstd")
    baseline_full = pl.read_parquet(Path(config["baseline_predictions"])).with_columns(
        pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date))
    common_keys = lstm.select("trade_date", "ts_code")
    baseline = baseline_full.join(common_keys, on=["trade_date", "ts_code"], how="semi")
    if baseline.height != lstm.height:
        raise ValueError("baseline does not cover every sequence prediction key; compare on identical keys")
    baseline_path = prediction_root / f"{baseline_name}.parquet"
    baseline.write_parquet(baseline_path, compression="zstd")
    labels = build_labels(Path(config["catalog"]), **_label_options(config)).select("trade_date", "ts_code", "excess_h1", "excess_h5")
    lstm_measured = lstm.join(labels, on=["trade_date", "ts_code"], how="left")
    baseline_measured = baseline.join(labels, on=["trade_date", "ts_code"], how="left")
    report = {"fingerprint": cache_manifest["fingerprint"], "pilot": pilot,
              "factor_count": cache_manifest["factor_count"],
              "training_label_lag": int(config.get("training_label_lag", 6)),
              "label_options": _label_options(config),
              "execution_filter": {"last_observed_day": str(last_observed_day),
                                   "excluded_rows_without_next_trading_day": non_executable_rows},
              "coverage": coverage, "models": {}, "top100": _top_overlap(lstm, baseline)}
    for name, measured in ((lstm_name, lstm_measured), (baseline_name, baseline_measured)):
        report["models"][name] = {"h1": _metrics(measured, "raw_h1", "excess_h1"),
                                  "h5": _metrics(measured, "raw_h5", "excess_h5"),
                                  "period_metrics": _period_metrics(measured, "raw_h1", "raw_h5"),
                                  "diagnostics": _diagnostics(measured, "raw_h1", "raw_h5")}
    report["paired_ic_difference_lstm_minus_lgbm"] = {
        horizon: block_bootstrap_difference(baseline_measured, lstm_measured, horizon)
        for horizon in ("h1", "h5")}
    joined = lstm.join(baseline.select("trade_date", "ts_code", "raw_h1", "raw_h5").rename(
        {"raw_h1": "b_h1", "raw_h5": "b_h5"}), on=["trade_date", "ts_code"])
    report["prediction_correlation"] = {h: float(joined.select(pl.corr(f"raw_{h}", f"b_{h}")).item())
                                        for h in ("h1", "h5")}

    if not (skip_backtests or pilot):
        strategy = json.loads(Path(config["strategy_config"]).read_text(encoding="utf-8"))
        policy = {key: value for key, value in {**strategy["portfolio"], **strategy["costs"]}.items()
                  if key != "strategy"}
        csi_paths = {}
        for name, frame in ((lstm_name, lstm), (baseline_name, baseline)):
            path = prediction_root / f"{name}_csi500.parquet"
            _csi500(Path(config["catalog"]), frame).write_parquet(path, compression="zstd")
            csi_paths[name] = path
        backtest_root = run_root / "backtests"
        report["backtests"] = _run_backtests(Path(config["catalog"]), csi_paths, backtest_root, policy)
        chart = run_root / "reports/common_sample_nav.png"; chart.parent.mkdir(parents=True, exist_ok=True)
        _comparison_chart(backtest_root, list(csi_paths), chart); report["comparison_chart"] = str(chart)

    rows = []
    for model, item in report["models"].items():
        for year, metrics in item["period_metrics"]["year"].items():
            for horizon in ("h1", "h5"):
                rows.append({"model": model, "year": year, "horizon": horizon, **metrics[horizon]})
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(run_root / "reports/yearly_metrics.csv")
    _write_json(run_root / "report.json", report)
    markdown = ["# LSTM vs LightGBM rolling OOS", "", f"Sequence coverage: {coverage['coverage']:.4%}.", "",
                "| Model | H1 Rank IC | H5 Rank IC |", "| --- | ---: | ---: |"]
    for model, item in report["models"].items():
        markdown.append(f"| {model} | {item['h1'].get('mean_rank_ic', float('nan')):.6f} | {item['h5'].get('mean_rank_ic', float('nan')):.6f} |")
    (run_root / "reports/README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"output": str(run_root), "cache": str(cache_path), "windows": len(windows) if not effective_windows else effective_windows,
            "predictions": str(lstm_path), "report": str(run_root / "report.json"), "coverage": coverage}
