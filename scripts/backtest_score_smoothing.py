#!/usr/bin/env python3
"""Backtest causal EWMA smoothing of the composite h1/h5 score before the rank policy.

The policy scores by 0.5*z(pred_h1)+0.5*z(pred_h5); writing the smoothed
composite into both prediction columns makes it rank by z(smoothed composite)
exactly, so smoothing is the only thing that changes versus the baseline files.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _smoothed_frame(frame: pl.DataFrame, half_life: float) -> pl.DataFrame:
    """Causal per-stock EWMA of the composite z-score, keyed by trade_date only."""
    alpha = 1.0 - np.exp(np.log(0.5) / half_life)
    base = frame.with_columns(((pl.col("pred_h1") + pl.col("pred_h5")) / 2.0).alias("composite"))
    base = base.sort("ts_code", "trade_date")
    codes = base["ts_code"].to_numpy()
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    segments = np.split(np.arange(base.height), boundaries)
    values = base["composite"].to_numpy()
    smoothed = np.full(base.height, np.nan)
    for segment in segments:
        state = np.nan
        for position in segment:
            current = values[position]
            if np.isnan(current):
                smoothed[position] = state
                continue
            state = current if np.isnan(state) else alpha * current + (1.0 - alpha) * state
            smoothed[position] = state
    return base.with_columns(pl.Series("smoothed", smoothed)).select(
        "trade_date", "ts_code", "execution_date", "raw_h1", "raw_h5",
        pl.col("smoothed").alias("pred_h1"), pl.col("smoothed").alias("pred_h5"))


def _metrics(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    daily = pl.read_parquet(path / "portfolio_daily.parquet").sort("execution_date")
    executions = pl.read_parquet(path / "executions.parquet")
    nav = daily["nav"].to_numpy()
    return {
        "days": daily.height,
        "final_nav": float(nav[-1]),
        "net_total_return": float(nav[-1] - 1),
        "net_annualized_return": float(nav[-1] ** (252 / daily.height) - 1),
        "information_ratio": result["charged"]["information_ratio"],
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_buy_turnover": float(daily["buy_turnover"].sum()),
        "total_sell_turnover": float(daily["sell_turnover"].sum()),
        "transaction_cost_rate_sum": result["charged"]["sum_daily_transaction_cost_rate"],
        "average_cash_weight": result["charged"]["average_cash_weight"],
        "average_holding_count": result["charged"]["average_holding_count"],
        "minimum_holding_count": int(daily["holding_count"].min()),
        "filled_orders": executions.height,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    strategy = _read_json(args.strategy_config)
    policy = {**strategy["portfolio"], **strategy["costs"]}
    policy.pop("strategy", None)
    base = LimitedReplacementConfig(**policy)
    variants = {
        "daily_swap3": base,
        "daily_swap15": replace(base, max_daily_replacements=15,
                                daily_buy_budget=.50, daily_sell_budget=.50),
        "weekly_swap15": replace(base, rebalance_frequency="weekly", max_daily_replacements=15,
                                 daily_buy_budget=.50, daily_sell_budget=.50),
    }
    source = pl.read_parquet(args.predictions.expanduser())
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "predictions"
    prepared.mkdir(exist_ok=True)
    prediction_paths = {"hl0": args.predictions.expanduser()}
    for half_life in [float(value) for value in args.half_lives.split(",")]:
        name = f"hl{half_life:g}"
        target = prepared / f"blend_smooth_{name}_csi500.parquet"
        _smoothed_frame(source, half_life).write_parquet(target, compression="zstd")
        prediction_paths[name] = target
        print(f"prepared {name}", flush=True)

    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for variant, config in variants.items():
        for name, predictions in prediction_paths.items():
            print(f"running {variant} {name}", flush=True)
            target = output / "backtests" / variant / name
            result = run_limited_replacement_policy(args.catalog.expanduser(), predictions, target, config)
            rows.append({"variant": variant, "half_life": name, **_metrics(target, result)})
            annual = pl.read_parquet(target / "annual_metrics.parquet")
            annual_rows.extend({"variant": variant, "half_life": name, **row}
                               for row in annual.to_dicts())
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    pl.DataFrame(rows).write_csv(output / "summary.csv")
    pl.DataFrame(annual_rows).write_csv(output / "annual_summary.csv")
    (output / "manifest.json").write_text(json.dumps({
        "predictions": str(args.predictions.expanduser().resolve()),
        "half_lives": list(prediction_paths),
        "interpretation": "hl0 is the untouched blend file; hlN ranks by z(EWMA(composite, half-life N))",
        "variants": {name: {"rebalance_frequency": config.rebalance_frequency,
                            "max_daily_replacements": config.max_daily_replacements}
                     for name, config in variants.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-config", type=Path, default=Path("configs/production_strategy_csi500_top100_v1.json"))
    parser.add_argument("--catalog", type=Path, default=Path("A_stock_database/lake/catalog/a_share.duckdb"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/weekly_comparison/predictions/blend_50_50_csi500.parquet"))
    parser.add_argument("--half-lives", default="2,3,5")
    parser.add_argument("--output", type=Path,
                        default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/score_smoothing_experiment"))
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
