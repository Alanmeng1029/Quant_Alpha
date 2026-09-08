"""Leakage-safe rolling OOS model comparison for the formal 98-factor set.

This module intentionally sits beside the legacy ``run-oos`` command.  It does
not change the production prediction or the V2 policy: it produces a fully
versioned research run whose outputs can be handed to that policy unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
from itertools import product
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import lightgbm as lgb
import duckdb
import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import (INFEASIBLE_EXECUTION_CODES, LABEL_LAG, TRAIN_DAYS,
                                  build_labels, render_backtest_report, standardize_features)


SEED = 20260908
THREADS = min(8, max(1, os.cpu_count() or 1))
HORIZONS = ("h1", "h5")
TARGETS = {"h1": "excess_h1", "h5": "excess_h5"}
BASE_MODELS = ("lgbm_default98", "lgbm_tuned98", "xgboost_tuned98", "mlp_tuned98")
ALL_MODELS = BASE_MODELS + ("lgbm_xgb_equal", "tree_mlp_equal")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _source_fingerprints() -> dict[str, str]:
    """Hash the code which determines a recoverable research run's semantics."""
    root = Path(__file__).resolve().parents[2]
    files = (root / "src/a_share_data/research_oos.py", root / "src/a_share_data/predict.py", root / "src/a_share_data/policy.py")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def next_trading_date_map(calendar: list[date], signals: list[date]) -> dict[date, date]:
    positions = {day: index for index, day in enumerate(calendar)}
    result: dict[date, date] = {}
    for signal in signals:
        index = positions.get(signal)
        if index is None or index + 1 >= len(calendar):
            raise ValueError(f"no next observed trading date for signal {signal}")
        result[signal] = calendar[index + 1]
    return result


def _execution_calendar(catalog: Path, signal_dates: list[date]) -> pl.DataFrame:
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        calendar = [row[0] for row in conn.execute("SELECT trade_date FROM observed_calendar WHERE is_observed_market_day ORDER BY trade_date").fetchall()]
    finally:
        conn.close()
    mapping = next_trading_date_map(calendar, signal_dates)
    return pl.DataFrame({"trade_date": list(mapping), "execution_date": list(mapping.values())}).with_columns(pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _date_strings(frame: pl.DataFrame) -> list[str]:
    return sorted(str(value) for value in frame.select("trade_date").unique().to_series().to_list())


def _date_values(values: list[str]) -> list[date]:
    return [date.fromisoformat(value) for value in values]


def validation_folds(train_dates: list[str], purge_days: int = LABEL_LAG) -> list[tuple[list[str], list[str]]]:
    """Three contiguous 63-day folds in the last 189 training dates.

    The six-date gap is deliberate: an H5 label for the final training signal
    settles six trading days later, before the validation block starts.
    """
    if len(train_dates) != TRAIN_DAYS:
        raise ValueError(f"expected exactly {TRAIN_DAYS} training dates, got {len(train_dates)}")
    folds: list[tuple[list[str], list[str]]] = []
    for start in (TRAIN_DAYS - 189, TRAIN_DAYS - 126, TRAIN_DAYS - 63):
        valid = train_dates[start:start + 63]
        fit = train_dates[:start - purge_days]
        if len(fit) < 1 or len(valid) != 63:
            raise ValueError("invalid inner validation fold")
        # Calendar positions make the maturity assertion explicit and testable.
        assert train_dates.index(fit[-1]) + LABEL_LAG < train_dates.index(valid[0])
        folds.append((fit, valid))
    return folds


def quarter_windows(dates: list[str], start: str, end: str) -> list[tuple[str, list[str], list[str]]]:
    """Quarterly OOS blocks with a 756-day historical window and six-day gap."""
    result: list[tuple[str, list[str], list[str]]] = []
    for index, signal in enumerate(dates):
        if signal < start or signal > end or index < TRAIN_DAYS + LABEL_LAG:
            continue
        previous = dates[index - 1] if index else ""
        if signal[:7] == previous[:7]:
            continue
        # Refit on the first signal of each three-month block, not every month.
        month_number = int(signal[5:7])
        if (month_number - int(start[5:7])) % 3:
            continue
        train = dates[index - LABEL_LAG - TRAIN_DAYS:index - LABEL_LAG]
        # Calendar-quarter boundary rather than an assumed 23 trading days/month.
        signal_year, signal_month = int(signal[:4]), int(signal[5:7])
        end_month_index = signal_year * 12 + signal_month - 1 + 3
        test = [day for day in dates[index:] if day <= end and (int(day[:4]) * 12 + int(day[5:7]) - 1) < end_month_index]
        if len(train) == TRAIN_DAYS and test:
            result.append((signal, train, test))
    return result


def _daily_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=False)
    finite = np.isfinite(values)
    out = np.full(len(values), np.nan)
    if finite.sum() < 2:
        return out
    mean, std = values[finite].mean(), values[finite].std()
    if std <= 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - mean) / std
    return out


def daily_normalize(frame: pl.DataFrame, raw_h1: str, raw_h5: str) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for day, group in frame.group_by("trade_date", maintain_order=True):
        parts.append(group.with_columns(
            pl.Series("pred_h1", _daily_zscore(group.get_column(raw_h1).to_numpy())),
            pl.Series("pred_h5", _daily_zscore(group.get_column(raw_h5).to_numpy())),
        ))
    return pl.concat(parts) if parts else frame.with_columns(pl.lit(None).cast(pl.Float64).alias("pred_h1"), pl.lit(None).cast(pl.Float64).alias("pred_h5"))


def _rank_ic(frame: pl.DataFrame, prediction: str, target: str) -> float | None:
    valid = frame.select("trade_date", prediction, target).drop_nulls()
    if valid.is_empty():
        return None
    daily = valid.group_by("trade_date").agg(pl.corr(pl.col(prediction).rank(), pl.col(target).rank()).alias("ic"))
    value = daily.select(pl.col("ic").mean()).item()
    return float(value) if value is not None and np.isfinite(value) else None


def _feature_list(frame: pl.DataFrame, dates: list[str], factor_ids: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    train = frame.filter(pl.col("trade_date").is_in(_date_values(dates)))
    selected: list[str] = []
    excluded: dict[str, str] = {}
    for feature in factor_ids:
        values = train.get_column(feature).to_numpy().astype(float, copy=False)
        finite = values[np.isfinite(values)]
        if not len(finite):
            excluded[feature] = "all_missing_training_window"
        elif np.nanmax(finite) - np.nanmin(finite) <= 1e-12:
            excluded[feature] = "constant_training_window"
        else:
            selected.append(feature)
    return selected, excluded


def _matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    return frame.select(features).to_numpy().astype(np.float32, copy=False)


def _target(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame.get_column(name).to_numpy().astype(np.float64, copy=False)


def _winsorized_training(frame: pl.DataFrame, target: str) -> np.ndarray:
    # Bounds are calculated independently within each *training* signal date.
    return frame.with_columns(
        pl.col(target).quantile(.01).over("trade_date").alias("__lo"),
        pl.col(target).quantile(.99).over("trade_date").alias("__hi"),
    ).with_columns(pl.col(target).clip(pl.col("__lo"), pl.col("__hi")).alias("__y")).get_column("__y").to_numpy().astype(np.float64)


def _lgb_params(choice: dict[str, Any]) -> dict[str, Any]:
    return {"objective": "regression", "metric": "l2", "learning_rate": .03, "bagging_fraction": .8,
            "bagging_freq": 1, "feature_pre_filter": False, "verbosity": -1, "seed": SEED,
            "feature_fraction_seed": SEED, "bagging_seed": SEED, "data_random_seed": SEED,
            "deterministic": True, "force_row_wise": True, "num_threads": THREADS, **choice}


def _lgb_choices() -> list[dict[str, Any]]:
    all_choices = [dict(zip(("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2"), item))
                   for item in product((7, 15, 31), (500, 1000, 2000), (.6, .8, 1.0), (1, 10, 50))]
    return [all_choices[index] for index in np.random.default_rng(SEED).choice(len(all_choices), size=12, replace=False)]


def _fit_lgbm(train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, features: list[str], target: str, choice: dict[str, Any], final_rounds: int | None = None) -> tuple[np.ndarray, int, Any]:
    usable = train.filter(pl.col(target).is_not_null())
    y = _winsorized_training(usable, target)
    params = _lgb_params(choice)
    if final_rounds is not None:
        model = lgb.train(params, lgb.Dataset(_matrix(usable, features), label=y, feature_name=features), num_boost_round=max(1, final_rounds))
        return model.predict(_matrix(test, features)), final_rounds, model
    validation = valid.filter(pl.col(target).is_not_null())
    model = lgb.train(params, lgb.Dataset(_matrix(usable, features), label=y, feature_name=features),
                      valid_sets=[lgb.Dataset(_matrix(validation, features), label=_target(validation, target))],
                      num_boost_round=2000, callbacks=[lgb.early_stopping(50, verbose=False)])
    rounds = max(1, int(model.best_iteration or 1))
    return model.predict(_matrix(test, features), num_iteration=rounds), rounds, model


def _xgb_choices() -> list[dict[str, Any]]:
    all_choices = [dict(zip(("max_depth", "min_child_weight", "colsample_bytree", "reg_lambda"), item))
                   for item in product((3, 4, 5), (100, 500, 1000), (.6, .8, 1.0), (1, 10, 50))]
    return [all_choices[index] for index in np.random.default_rng(SEED).choice(len(all_choices), size=12, replace=False)]


def _fit_xgb(train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, features: list[str], target: str, choice: dict[str, Any], final_rounds: int | None = None) -> tuple[np.ndarray, int, Any]:
    import xgboost as xgb
    usable = train.filter(pl.col(target).is_not_null())
    y = _winsorized_training(usable, target)
    common = dict(n_estimators=2000 if final_rounds is None else max(1, final_rounds), learning_rate=.03,
                  tree_method="hist", subsample=.8, random_state=SEED, n_jobs=THREADS, objective="reg:squarederror", **choice)
    if final_rounds is not None:
        model = xgb.XGBRegressor(**common).fit(_matrix(usable, features), y, verbose=False)
        return model.predict(_matrix(test, features)), final_rounds, model
    validation = valid.filter(pl.col(target).is_not_null())
    model = xgb.XGBRegressor(**common, early_stopping_rounds=50).fit(
        _matrix(usable, features), y, eval_set=[(_matrix(validation, features), _target(validation, target))], verbose=False)
    rounds = int(getattr(model, "best_iteration", 0) or 0) + 1
    return model.predict(_matrix(test, features)), rounds, model


def _mlp_choices() -> list[dict[str, Any]]:
    return [{"width": width, "weight_decay": decay} for width, decay in product((64, 128), (.0001, .001))]


def _fit_mlp(train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, features: list[str], choice: dict[str, Any], final_epochs: int | None = None) -> tuple[np.ndarray, int, dict[str, Any]]:
    import torch
    from torch import nn
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    class Net(nn.Module):
        def __init__(self, width: int):
            super().__init__(); self.net = nn.Sequential(nn.Linear(len(features), width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, 2))
        def forward(self, x): return self.net(x)
    x_train = np.nan_to_num(_matrix(train, features), nan=0.0)
    raw_train = np.column_stack([_target(train, TARGETS[h]) for h in HORIZONS]).astype(np.float32)
    means = np.array([np.nanmean(raw_train[:, i]) for i in range(2)], dtype=np.float32)
    scales = np.array([max(np.nanstd(raw_train[:, i]), 1e-8) for i in range(2)], dtype=np.float32)
    y_train = (raw_train - means) / scales
    y_train[~np.isfinite(y_train)] = 0.0
    mask_train = np.isfinite(raw_train)
    model = Net(int(choice["width"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=float(choice["weight_decay"]))
    batch = 4096; best_state: dict[str, Any] | None = None; best_loss = float("inf"); best_epoch = 1; stale = 0
    x_valid = np.nan_to_num(_matrix(valid, features), nan=0.0)
    raw_valid = np.column_stack([_target(valid, TARGETS[h]) for h in HORIZONS]).astype(np.float32)
    def batches():
        # Date ordered, deterministic batches: no randomized future leakage.
        for begin in range(0, len(x_train), batch): yield slice(begin, min(begin + batch, len(x_train)))
    epochs = final_epochs or 50
    for epoch in range(1, epochs + 1):
        model.train()
        for sl in batches():
            x = torch.as_tensor(x_train[sl], device=device); y = torch.as_tensor(y_train[sl], device=device); mask = torch.as_tensor(mask_train[sl], device=device)
            pred = model(x); loss = ((pred - y).square() * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if final_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            prediction = model(torch.as_tensor(x_valid, device=device)).cpu().numpy() * scales + means
        observed = np.isfinite(raw_valid)
        mse = float(((prediction[observed] - raw_valid[observed]) ** 2).mean()) if observed.any() else float("inf")
        if mse < best_loss - 1e-14:
            best_loss = mse; best_epoch = epoch; stale = 0; best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5: break
    if final_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(np.nan_to_num(_matrix(test, features), nan=0.0), device=device)).cpu().numpy() * scales + means
    state = {"state_dict": {key: value.cpu() for key, value in model.state_dict().items()}, "means": means, "scales": scales, "choice": choice, "epochs": final_epochs or best_epoch}
    return prediction, int(final_epochs or best_epoch), state


def _tune(panel: pl.DataFrame, train_dates: list[str], features: list[str], family: str, checkpoint: Path | None = None) -> dict[str, Any]:
    choices = _lgb_choices() if family == "lgbm" else _xgb_choices() if family == "xgb" else _mlp_choices()
    trials: list[dict[str, Any]] = []
    if checkpoint and checkpoint.exists():
        trials = json.loads(checkpoint.read_text(encoding="utf-8")).get("trials", [])
    completed = {_json_hash(item["params"]) for item in trials}
    for choice in choices:
        if _json_hash(choice) in completed:
            continue
        result: dict[str, Any] = {"params": choice, "folds": {}}
        for horizon in HORIZONS if family != "mlp" else ("joint",):
            scores: list[float] = []
            rounds: list[int] = []
            for fit_dates, valid_dates in validation_folds(train_dates):
                train = panel.filter(pl.col("trade_date").is_in(_date_values(fit_dates))); valid = panel.filter(pl.col("trade_date").is_in(_date_values(valid_dates)))
                if family == "lgbm":
                    prediction, used, _ = _fit_lgbm(train, valid, valid, features, TARGETS[horizon], choice)
                    scored = valid.with_columns(pl.Series("prediction", prediction))
                    score = _rank_ic(scored, "prediction", TARGETS[horizon])
                elif family == "xgb":
                    prediction, used, _ = _fit_xgb(train, valid, valid, features, TARGETS[horizon], choice)
                    scored = valid.with_columns(pl.Series("prediction", prediction))
                    score = _rank_ic(scored, "prediction", TARGETS[horizon])
                else:
                    prediction, used, _ = _fit_mlp(train, valid, valid, features, choice)
                    scored = valid.with_columns(pl.Series("prediction_h1", prediction[:, 0]), pl.Series("prediction_h5", prediction[:, 1]))
                    score = np.nanmean([_rank_ic(scored, "prediction_h1", TARGETS["h1"]), _rank_ic(scored, "prediction_h5", TARGETS["h5"])])
                scores.append(float(score) if score is not None else float("nan")); rounds.append(used)
            result["folds"][horizon] = {"rank_ic": scores, "mean_rank_ic": float(np.nanmean(scores)), "mean_rounds": float(np.mean(rounds))}
        result["selection_score"] = float(np.nanmean([entry["mean_rank_ic"] for entry in result["folds"].values()]))
        trials.append(result)
        if checkpoint:
            _write_json(checkpoint, {"family": family, "seed": SEED, "threads": THREADS, "trials": trials})
    best: dict[str, Any] = {"trials": trials}
    if family == "mlp":
        chosen = max(trials, key=lambda item: item["selection_score"])
        best["params"] = chosen["params"]
    else:
        for horizon in HORIZONS:
            chosen = max(trials, key=lambda item: item["folds"][horizon]["mean_rank_ic"])
            best[horizon] = {"params": chosen["params"], "selection_score": chosen["folds"][horizon]["mean_rank_ic"]}
    return best


def _predict_final(panel: pl.DataFrame, train_dates: list[str], test_dates: list[str], features: list[str], tuned: dict[str, Any], families: tuple[str, ...]) -> tuple[dict[str, pl.DataFrame], dict[str, Any], dict[str, Any]]:
    train = panel.filter(pl.col("trade_date").is_in(_date_values(train_dates))); test = panel.filter(pl.col("trade_date").is_in(_date_values(test_dates)))
    final_fit_dates, final_valid_dates = validation_folds(train_dates)[-1]
    early_train = panel.filter(pl.col("trade_date").is_in(_date_values(final_fit_dates))); early_valid = panel.filter(pl.col("trade_date").is_in(_date_values(final_valid_dates)))
    bases = test.select("trade_date", "ts_code").with_columns(pl.col("trade_date").cast(pl.Date))
    raw: dict[str, pl.DataFrame] = {}; metadata: dict[str, Any] = {}; saved: dict[str, Any] = {}
    tree_outputs = (("lgbm", "lgbm_tuned98"), ("xgb", "xgboost_tuned98"))
    for family, output_name in tree_outputs:
        if family not in families:
            continue
        cols: list[pl.Series] = []; rounds_record: dict[str, int] = {}; saved[output_name] = {}
        for horizon in HORIZONS:
            choice = tuned[family][horizon]["params"]
            fitter = _fit_lgbm if family == "lgbm" else _fit_xgb
            _, rounds, _ = fitter(early_train, early_valid, early_valid, features, TARGETS[horizon], choice)
            prediction, _, model = fitter(train, early_valid, test, features, TARGETS[horizon], choice, rounds)
            cols.append(pl.Series(f"raw_{horizon}", prediction)); rounds_record[horizon] = rounds; saved[output_name][horizon] = model
        raw[output_name] = bases.with_columns(cols); metadata[output_name] = {"params": tuned[family], "rounds": rounds_record}
    # Default LGBM is a true all-98 reference, with no search and the same final validation protocol.
    default_choice = {"num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 1.0, "lambda_l2": 0.0}
    cols = []; rounds_record = {}; saved["lgbm_default98"] = {}
    for horizon in HORIZONS:
        _, rounds, _ = _fit_lgbm(early_train, early_valid, early_valid, features, TARGETS[horizon], default_choice)
        prediction, _, model = _fit_lgbm(train, early_valid, test, features, TARGETS[horizon], default_choice, rounds)
        cols.append(pl.Series(f"raw_{horizon}", prediction)); rounds_record[horizon] = rounds; saved["lgbm_default98"][horizon] = model
    raw["lgbm_default98"] = bases.with_columns(cols); metadata["lgbm_default98"] = {"params": default_choice, "rounds": rounds_record}
    if "mlp" in families:
        _, epochs, _ = _fit_mlp(early_train, early_valid, early_valid, features, tuned["mlp"]["params"])
        prediction, _, state = _fit_mlp(train, early_valid, test, features, tuned["mlp"]["params"], final_epochs=epochs)
        raw["mlp_tuned98"] = bases.with_columns(pl.Series("raw_h1", prediction[:, 0]), pl.Series("raw_h5", prediction[:, 1]))
        metadata["mlp_tuned98"] = {"params": tuned["mlp"]["params"], "epochs": epochs}; saved["mlp_tuned98"] = state
    return raw, metadata, saved


def _save_models(models: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for model_name, item in models.items():
        if model_name == "mlp_tuned98":
            import torch
            torch.save(item, destination / "mlp_tuned98.pt")
        elif model_name.startswith("lgbm"):
            for horizon, model in item.items(): model.save_model(str(destination / f"{model_name}_{horizon}.txt"))
        else:
            for horizon, model in item.items(): model.save_model(str(destination / f"{model_name}_{horizon}.json"))


def _metrics(frame: pl.DataFrame, raw_horizon: str, target: str) -> dict[str, Any]:
    valid = frame.select("trade_date", raw_horizon, target).drop_nulls()
    if valid.is_empty(): return {"observations": 0}
    daily = valid.group_by("trade_date").agg(pl.corr(raw_horizon, target).alias("pearson"), pl.corr(pl.col(raw_horizon).rank(), pl.col(target).rank()).alias("rank"))
    rank = daily.get_column("rank").to_numpy(); mean = float(np.nanmean(rank)); std = float(np.nanstd(rank, ddof=1))
    err = valid.with_columns((pl.col(raw_horizon) - pl.col(target)).alias("error"))
    return {"observations": valid.height, "days": daily.height, "mean_rank_ic": mean, "mean_pearson_ic": float(np.nanmean(daily.get_column("pearson").to_numpy())), "positive_ic_ratio": float(np.nanmean(rank > 0)), "rank_icir_unannualized": mean / std if std > 0 else None, "rank_icir_annualized": mean / std * np.sqrt(252) if std > 0 else None, "mae_raw_return": float(err.select(pl.col("error").abs().mean()).item()), "mse_raw_return": float(err.select((pl.col("error") ** 2).mean()).item())}


def block_bootstrap_difference(base: pl.DataFrame, candidate: pl.DataFrame, horizon: str, repeats: int = 2000, block_days: int = 20) -> dict[str, Any]:
    target = TARGETS[horizon]; raw = f"raw_{horizon}"
    def daily_ic(frame: pl.DataFrame) -> dict[str, float]:
        return {str(day): float(value) for day, value in frame.select("trade_date", raw, target).drop_nulls().group_by("trade_date").agg(pl.corr(pl.col(raw).rank(), pl.col(target).rank()).alias("ic")).iter_rows()}
    left, right = daily_ic(base), daily_ic(candidate); dates = sorted(set(left) & set(right)); delta = np.array([right[d] - left[d] for d in dates])
    if len(delta) < block_days: return {"days": len(delta), "mean_difference": float(np.nanmean(delta)) if len(delta) else None, "ci95": None}
    starts = np.arange(0, len(delta) - block_days + 1); rng = np.random.default_rng(SEED); sampled = np.empty(repeats)
    need = int(np.ceil(len(delta) / block_days))
    for i in range(repeats): sampled[i] = np.concatenate([delta[start:start + block_days] for start in rng.choice(starts, need)] )[:len(delta)].mean()
    return {"days": len(delta), "mean_difference": float(delta.mean()), "block_days": block_days, "repeats": repeats, "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]}


def _diagnostics(frame: pl.DataFrame, raw_h1: str, raw_h5: str) -> dict[str, Any]:
    joined = frame.sort(["trade_date", "ts_code"])
    daily: list[dict[str, Any]] = []
    prior_ranks: dict[str, float] = {}; prior_top: set[str] = set()
    for day, group in joined.group_by("trade_date", maintain_order=True):
        score = .5 * _daily_zscore(group.get_column(raw_h1).to_numpy()) + .5 * _daily_zscore(group.get_column(raw_h5).to_numpy())
        codes = group.get_column("ts_code").to_list(); order = np.lexsort((np.array(codes), -np.nan_to_num(score, nan=-np.inf))); ranks = {codes[i]: float(j + 1) for j, i in enumerate(order)}; top = set(codes[i] for i in order[:80])
        common = sorted(set(ranks) & set(prior_ranks)); corr = float(np.corrcoef([ranks[x] for x in common], [prior_ranks[x] for x in common])[0, 1]) if len(common) > 2 else None
        actual = group.with_columns(pl.Series("__score", score)).sort("__score", descending=True).head(80)
        daily.append({"trade_date": str(day[0] if isinstance(day, tuple) else day), "top80_excess_h1": actual.select(pl.col("excess_h1").mean()).item(), "top80_excess_h5": actual.select(pl.col("excess_h5").mean()).item(), "adjacent_rank_corr": corr, "top80_overlap": len(top & prior_top) / 80 if prior_top else None})
        prior_ranks, prior_top = ranks, top
    rank_corr = [value for item in daily if (value := item["adjacent_rank_corr"]) is not None]
    overlap = [value for item in daily if (value := item["top80_overlap"]) is not None]
    return {"daily": daily, "mean_adjacent_rank_corr": float(np.nanmean(rank_corr)) if rank_corr else None, "mean_top80_overlap": float(np.nanmean(overlap)) if overlap else None}


def _period_metrics(frame: pl.DataFrame, raw_h1: str, raw_h5: str) -> dict[str, Any]:
    """Metrics split by calendar year and natural quarter, never used for selection."""
    dated = frame.with_columns(
        pl.col("trade_date").dt.year().cast(pl.Utf8).alias("__year"),
        (pl.col("trade_date").dt.year().cast(pl.Utf8) + "Q" + pl.col("trade_date").dt.quarter().cast(pl.Utf8)).alias("__quarter"),
    )
    result: dict[str, Any] = {"year": {}, "quarter": {}}
    for column, name in (("__year", "year"), ("__quarter", "quarter")):
        for value, group in dated.group_by(column, maintain_order=True):
            key = value[0] if isinstance(value, tuple) else value
            result[name][str(key)] = {"h1": _metrics(group, raw_h1, "excess_h1"), "h5": _metrics(group, raw_h5, "excess_h5")}
    return result


def _fuse(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Preserve raw-return averages while blending the stipulated daily z-scores."""
    left_z = daily_normalize(left, "raw_h1", "raw_h5")
    right_z = daily_normalize(right, "raw_h1", "raw_h5")
    joined = left_z.join(right_z.select("trade_date", "ts_code", "raw_h1", "raw_h5", "pred_h1", "pred_h5").rename({"raw_h1": "__h1", "raw_h5": "__h5", "pred_h1": "__z1", "pred_h5": "__z5"}), on=["trade_date", "ts_code"])
    return joined.with_columns(
        ((pl.col("raw_h1") + pl.col("__h1")) / 2).alias("raw_h1"),
        ((pl.col("raw_h5") + pl.col("__h5")) / 2).alias("raw_h5"),
        ((pl.col("pred_h1") + pl.col("__z1")) / 2).alias("blend_h1"),
        ((pl.col("pred_h5") + pl.col("__z5")) / 2).alias("blend_h5"),
    ).select("trade_date", "ts_code", "raw_h1", "raw_h5", "blend_h1", "blend_h5")


def _run_backtests(catalog: Path, predictions: dict[str, Path], output: Path, policy: dict[str, Any]) -> dict[str, Any]:
    config = LimitedReplacementConfig(**policy); config.validate(); results = {}
    for name, path in predictions.items():
        target = output / name
        result = run_limited_replacement_policy(catalog, path, target, config)
        result["charged_report"] = render_backtest_report(target / "portfolio_daily.parquet", target / "report", f"{name} — Optimizer V2 charged")
        result["zero_cost_report"] = render_backtest_report(target / "zero_cost" / "portfolio_daily.parquet", target / "zero_cost" / "report", f"{name} — Optimizer V2 zero cost")
        results[name] = result
    return results


def run_research_oos(config_path: Path, start_override: str | None = None, end_override: str | None = None, skip_backtests: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8")); root = Path(config["output"]).expanduser(); root.mkdir(parents=True, exist_ok=True)
    feature_root = Path(config["feature_root"]).expanduser(); catalog = Path(config["catalog"]).expanduser(); feature_manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    factor_ids = tuple(feature_manifest["factor_ids"])
    if len(factor_ids) != 98 or feature_manifest.get("factor_count", len(factor_ids)) != 98:
        raise ValueError("research-oos requires the fixed 98-factor cache")
    if any("ohlcv_candidates_v2" in value for value in factor_ids): raise ValueError("minute v2 factors are explicitly excluded")
    formal = json.loads(Path(config["formal_factor_set"]).read_text(encoding="utf-8"))
    if formal.get("factor_count") != 98: raise ValueError("formal factor registry is not the 98-factor v1 set")
    start = start_override or config["oos_start"]; end = end_override or config["oos_end"]
    families = tuple(config.get("model_families", ("lgbm", "xgb", "mlp")))
    if "lgbm" not in families or set(families) - {"lgbm", "xgb", "mlp"}:
        raise ValueError("model_families must include lgbm and may additionally include xgb and/or mlp")
    base_models: tuple[str, ...] = (("lgbm_default98", "lgbm_tuned98") if "lgbm" in families else ()) + (("xgboost_tuned98",) if "xgb" in families else ()) + (("mlp_tuned98",) if "mlp" in families else ())
    source_fingerprints = _source_fingerprints()
    fingerprint = _json_hash({"config": config, "features": feature_manifest, "source": source_fingerprints, "python": sys.version, "platform": platform.platform(), "lightgbm": lgb.__version__})
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("existing research output has a different input/configuration fingerprint")
    _write_json(manifest_path, {"fingerprint": fingerprint, "config": config, "feature_manifest": feature_manifest, "source_fingerprints": source_fingerprints, "dependencies": {"lightgbm": lgb.__version__, "python": sys.version, "platform": platform.platform()}})
    files = sorted(feature_root.glob("year=*/features.parquet")); features = pl.concat([pl.read_parquet(path) for path in files]).with_columns(pl.col("trade_date").cast(pl.Date)).filter(~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    labels = build_labels(catalog)
    panel = standardize_features(features, factor_ids).join(labels, on=["trade_date", "ts_code"], how="left")
    dates = _date_strings(panel); windows = quarter_windows(dates, start, end)
    if not windows: raise ValueError("no valid quarterly windows in requested OOS range")
    all_predictions: dict[str, list[pl.DataFrame]] = {name: [] for name in base_models}; training_log: list[dict[str, Any]] = []
    tuned_by_year: dict[int, dict[str, Any]] = {}
    for signal, train_dates, test_dates in windows:
        started = time.time(); year = int(signal[:4]); selected, excluded = _feature_list(panel, train_dates, factor_ids)
        if not selected: raise ValueError(f"no viable features for {signal}")
        tuning_file = root / "tuning" / f"year={year}.json"
        if year not in tuned_by_year:
            if tuning_file.exists(): tuned_by_year[year] = json.loads(tuning_file.read_text())
            else:
                tuned_by_year[year] = {family: _tune(panel, train_dates, selected, family, root / "tuning" / f"year={year}-{family}.partial.json") for family in families}
                _write_json(tuning_file, tuned_by_year[year])
        quarter_dir = root / "quarters" / f"month={signal[:7]}"; quarter_dir.mkdir(parents=True, exist_ok=True)
        existing = {name: quarter_dir / f"{name}.parquet" for name in base_models}
        if all(path.exists() for path in existing.values()):
            generated = {name: pl.read_parquet(path) for name, path in existing.items()}; metadata = {"resumed": True}; saved = {}
        else:
            generated, metadata, saved = _predict_final(panel, train_dates, test_dates, selected, tuned_by_year[year], families); _save_models(saved, quarter_dir / "models")
            for name, frame in generated.items(): frame.write_parquet(quarter_dir / f"{name}.parquet", compression="zstd")
        for name, frame in generated.items(): all_predictions[name].append(frame)
        training_log.append({"signal": signal, "test_start": test_dates[0], "test_end": test_dates[-1], "year": year, "feature_count": len(selected), "excluded_features": excluded, "metadata": metadata, "seconds": time.time() - started})
        _write_json(root / "training_log.json", training_log)
    outputs: dict[str, pl.DataFrame] = {name: pl.concat(parts).unique(["trade_date", "ts_code"], keep="first").sort(["trade_date", "ts_code"]) for name, parts in all_predictions.items()}
    if {"lgbm", "xgb"}.issubset(families):
        outputs["lgbm_xgb_equal"] = _fuse(outputs["lgbm_tuned98"], outputs["xgboost_tuned98"])
    if {"lgbm", "xgb", "mlp"}.issubset(families):
        # Combine the two-tree daily blend (weight 2/3) with MLP's daily z-score.
        tree = outputs["lgbm_xgb_equal"].join(daily_normalize(outputs["mlp_tuned98"], "raw_h1", "raw_h5").select("trade_date", "ts_code", "raw_h1", "raw_h5", "pred_h1", "pred_h5").rename({"raw_h1": "m_h1", "raw_h5": "m_h5", "pred_h1": "m_z1", "pred_h5": "m_z5"}), on=["trade_date", "ts_code"])
        outputs["tree_mlp_equal"] = tree.with_columns(
            ((pl.col("raw_h1") * 2 + pl.col("m_h1")) / 3).alias("raw_h1"),
            ((pl.col("raw_h5") * 2 + pl.col("m_h5")) / 3).alias("raw_h5"),
            ((pl.col("blend_h1") * 2 + pl.col("m_z1")) / 3).alias("blend_h1"),
            ((pl.col("blend_h5") * 2 + pl.col("m_z5")) / 3).alias("blend_h5"),
        ).select("trade_date", "ts_code", "raw_h1", "raw_h5", "blend_h1", "blend_h5")
    prediction_paths: dict[str, Path] = {}; report: dict[str, Any] = {"fingerprint": fingerprint, "oos_start": start, "oos_end": end, "models": {}, "paired_ic_difference_vs_default": {}}
    label_oos = panel.select("trade_date", "ts_code", "excess_h1", "excess_h5").filter((pl.col("trade_date") >= date.fromisoformat(start)) & (pl.col("trade_date") <= date.fromisoformat(end)))
    execution_dates = _execution_calendar(catalog, [date.fromisoformat(value) for value in _date_strings(label_oos)])
    for name, raw in outputs.items():
        normalized = (raw.rename({"blend_h1": "pred_h1", "blend_h5": "pred_h5"}) if "blend_h1" in raw.columns else daily_normalize(raw, "raw_h1", "raw_h5")).join(execution_dates, on="trade_date", how="left")
        path = root / "predictions" / f"{name}.parquet"; path.parent.mkdir(parents=True, exist_ok=True); normalized.write_parquet(path, compression="zstd"); prediction_paths[name] = path
        measured = normalized.join(label_oos, on=["trade_date", "ts_code"], how="left")
        report["models"][name] = {"h1": _metrics(measured, "raw_h1", "excess_h1"), "h5": _metrics(measured, "raw_h5", "excess_h5"), "period_metrics": _period_metrics(measured, "raw_h1", "raw_h5"), "diagnostics": _diagnostics(measured, "raw_h1", "raw_h5")}
        if name != "lgbm_default98": report["paired_ic_difference_vs_default"][name] = {h: block_bootstrap_difference(outputs["lgbm_default98"].join(label_oos, on=["trade_date", "ts_code"], how="left"), raw.join(label_oos, on=["trade_date", "ts_code"], how="left"), h) for h in HORIZONS}
    correlations = {}
    for left, right in product(tuple(outputs), repeat=2):
        if left < right:
            joined = outputs[left].join(outputs[right].rename({"raw_h1": "r_h1", "raw_h5": "r_h5"}), on=["trade_date", "ts_code"])
            correlations[f"{left}__{right}"] = {h: float(joined.select(pl.corr(f"raw_{h}", f"r_{h}")).item()) for h in HORIZONS}
    report["model_prediction_correlations"] = correlations
    if not skip_backtests and config.get("run_backtests", True): report["backtests"] = _run_backtests(catalog, prediction_paths, root / "backtests", config["optimizer_v2"])
    _write_json(root / "report.json", report)
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{item['h1'].get('mean_rank_ic')}</td><td>{item['h5'].get('mean_rank_ic')}</td><td>{item['h1'].get('rank_icir_annualized')}</td><td>{item['h5'].get('rank_icir_annualized')}</td></tr>" for name, item in report["models"].items())
    (root / "report.html").write_text(f"<html><body><h1>98-factor rolling OOS research</h1><p>{start} to {end}; fixed Optimizer V2.</p><table border='1'><tr><th>model</th><th>H1 Rank IC</th><th>H5 Rank IC</th><th>H1 annual ICIR</th><th>H5 annual ICIR</th></tr>{rows}</table><p>Full machine-readable report: report.json</p></body></html>", encoding="utf-8")
    return {"output": str(root), "windows": len(windows), "models": list(outputs), "report": str(root / "report.json")}
