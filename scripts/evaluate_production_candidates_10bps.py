#!/usr/bin/env python3
"""Serially revalidate production candidates with 10bp buy and sell costs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/predict/production-candidates-10bps-v1"
CATALOG = ROOT / "A_stock_database/lake/catalog/a_share.duckdb"
CANDIDATES = {
    "105_h1h5_50_50": (
        ROOT / "results/predict/dos-minute20-model-comparison-v1/daily60_minute45_baseline/csi500_predictions.parquet", 0.5),
    "105_h1": (
        ROOT / "results/predict/dos-minute20-model-comparison-v1/daily60_minute45_baseline/csi500_predictions.parquet", 1.0),
    "125_h1": (
        ROOT / "results/predict/dos-minute20-model-comparison-v1/daily60_minute45_dos20/csi500_predictions.parquet", 1.0),
}


def metrics(frame: pl.DataFrame) -> dict[str, float | int]:
    net = frame["net_return"].to_numpy(); benchmark = frame["csi500_return"].to_numpy()
    net_nav = np.cumprod(1.0 + net); benchmark_nav = np.cumprod(1.0 + benchmark)
    active = net - benchmark
    return {
        "days": len(net), "net_return": float(net_nav[-1] - 1.0),
        "csi500_return": float(benchmark_nav[-1] - 1.0),
        "relative_return": float(net_nav[-1] / benchmark_nav[-1] - 1.0),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252.0)),
        "max_drawdown": float((net_nav / np.maximum.accumulate(net_nav) - 1.0).min()),
        "average_buy_turnover": float(frame["buy_turnover"].mean()),
        "average_sell_turnover": float(frame["sell_turnover"].mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (prediction, h1_weight) in CANDIDATES.items():
        config = LimitedReplacementConfig(
            target_holdings=100, entry_rank=100, exit_rank=120,
            max_daily_replacements=3, h1_weight=h1_weight,
            buy_bps=10.0, sell_bps=10.0,
        )
        destination = OUTPUT / name
        run_limited_replacement_policy(CATALOG, prediction, destination, config)
        daily = pl.read_parquet(destination / "portfolio_daily.parquet")
        rows.append({"candidate": name, "period": "full", **metrics(daily), "config": asdict(config)})
        rows.append({"candidate": name, "period": "2024-2026",
                     **metrics(daily.filter(pl.col("execution_date") >= pl.date(2024, 1, 1))),
                     "config": asdict(config)})
        rows.append({"candidate": name, "period": "2025-2026",
                     **metrics(daily.filter(pl.col("execution_date") >= pl.date(2025, 1, 1))),
                     "config": asdict(config)})
        print(name, json.dumps(rows[-3], ensure_ascii=False), flush=True)
    serializable = [{**row, "config": json.dumps(row["config"], ensure_ascii=False)} for row in rows]
    pl.DataFrame(serializable).write_csv(OUTPUT / "comparison.csv")
    (OUTPUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
