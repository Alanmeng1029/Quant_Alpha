#!/usr/bin/env python3
"""Filter LambdaRank predictions to point-in-time CSI500 and backtest H1 serially."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import render_backtest_report


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "A_stock_database/lake/catalog/a_share.duckdb"
OUTPUT = ROOT / "results/predict/lambdarank-excess-oos-v1/evaluation-2bps"
MODELS = {
    "new125_lambdarank_h1": ROOT / "results/predict/lambdarank-excess-oos-v1/new125/predictions.parquet",
    "old105_lambdarank_h1": ROOT / "results/predict/lambdarank-excess-oos-v1/old105/predictions.parquet",
}


def summarize(frame: pl.DataFrame) -> dict[str, float | int]:
    net = frame["net_return"].to_numpy(); benchmark = frame["csi500_return"].to_numpy()
    nav = np.cumprod(1.0 + net); benchmark_nav = np.cumprod(1.0 + benchmark)
    active = net - benchmark
    return {
        "days": len(net), "net_return": float(nav[-1] - 1.0),
        "csi500_return": float(benchmark_nav[-1] - 1.0),
        "relative_return": float(nav[-1] / benchmark_nav[-1] - 1.0),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252.0)),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "average_buy_turnover": float(frame["buy_turnover"].mean()),
        "average_sell_turnover": float(frame["sell_turnover"].mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(CATALOG), read_only=True)
    try:
        universe = pl.from_arrow(connection.execute(
            "SELECT DISTINCT trade_date, ts_code FROM index_trading_universe WHERE index_code='000905.SH'"
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        connection.close()
    config = LimitedReplacementConfig(
        target_holdings=100, entry_rank=100, exit_rank=120,
        max_daily_replacements=3, h1_weight=1.0,
        buy_bps=2.0, sell_bps=2.0,
    )
    rows = []
    for name, source in MODELS.items():
        destination = OUTPUT / name
        destination.mkdir(parents=True, exist_ok=True)
        predictions = (
            pl.read_parquet(source)
            .join(universe, on=["trade_date", "ts_code"], how="semi")
            # The first LambdaRank OOS run produced scores whose realized
            # excess-return rank direction is negative.  Normalize the
            # tradable convention here so a larger score always means buy.
            .with_columns(
                [(-pl.col(column)).alias(column) for column in (
                    "raw_h1", "raw_h5", "raw_h10", "pred_h1", "pred_h5", "pred_h10"
                )]
            )
            .sort("trade_date", "ts_code")
        )
        prediction_path = destination / "csi500_predictions.parquet"
        predictions.write_parquet(prediction_path, compression="zstd")
        run_limited_replacement_policy(CATALOG, prediction_path, destination / "backtest", config)
        render_backtest_report(
            destination / "backtest/portfolio_daily.parquet",
            destination / "backtest/report",
            title=f"{name}: CSI500 H1 Top100 / Exit120 / Swap3 / 2bps each side",
        )
        daily = pl.read_parquet(destination / "backtest/portfolio_daily.parquet")
        for period, start in (("full", None), ("2024-2026", "2024-01-01"), ("2025-2026", "2025-01-01")):
            sample = daily if start is None else daily.filter(pl.col("execution_date") >= pl.lit(start).str.to_date())
            rows.append({"candidate": name, "period": period, **summarize(sample)})
    pl.DataFrame(rows).write_csv(OUTPUT / "comparison.csv")
    (OUTPUT / "summary.json").write_text(
        json.dumps({"config": asdict(config), "results": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
