"""Serial CSI500 policy backtests for streaming linear models and LGBM60."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import render_backtest_report

ROOT = Path(__file__).resolve().parents[1]
LINEAR = ROOT / "results/predict/research-raw-daily272-stream-linear-v1/predictions.parquet"
LGBM = ROOT / "results/predict/research-raw-daily60-original-standard-csi300-csi500-lgbm-v1/predictions.parquet"
OUTPUT = ROOT / "results/predict/research-raw-daily272-stream-linear-v1/backtests"
CATALOG = ROOT / "A_stock_database/lake/catalog/a_share.duckdb"


def metrics(frame: pl.DataFrame) -> dict[str, float | int]:
    nav = frame.get_column("nav").to_numpy()
    returns = frame.get_column("net_return").to_numpy()
    benchmark = frame.get_column("csi500_return").to_numpy()
    active = returns - benchmark
    return {
        "days": frame.height,
        "net_total_return": float(nav[-1] - 1),
        "annualized_return": float(nav[-1] ** (252 / frame.height) - 1),
        "csi500_total_return": float(np.prod(1 + benchmark) - 1),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "average_buy_turnover": float(frame.get_column("buy_turnover").mean()),
        "average_holding_count": float(frame.get_column("holding_count").mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    production = json.loads((ROOT / "configs/production_strategy_csi500_top100_v1.json").read_text())
    config = LimitedReplacementConfig(
        **{key: value for key, value in production["portfolio"].items() if key != "strategy"},
        **production["costs"],
    )
    with duckdb.connect(str(CATALOG), read_only=True) as conn:
        universe = pl.from_arrow(conn.execute(
            "SELECT DISTINCT trade_date,ts_code FROM index_trading_universe WHERE index_code='000905.SH'"
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    linear = pl.read_parquet(LINEAR)
    sources = {
        "ridge272": linear.select(
            "trade_date", "ts_code", "execution_date",
            pl.col("pred_ridge_h1").alias("pred_h1"),
            pl.col("pred_ridge_h5").alias("pred_h5"),
            pl.col("pred_ridge_h10").alias("pred_h10"),
        ),
        "elasticnet272": linear.select(
            "trade_date", "ts_code", "execution_date",
            pl.col("pred_elasticnet_h1").alias("pred_h1"),
            pl.col("pred_elasticnet_h5").alias("pred_h5"),
            pl.col("pred_elasticnet_h10").alias("pred_h10"),
        ),
        "lgbm60": pl.read_parquet(LGBM).select(
            "trade_date", "ts_code", "execution_date", "pred_h1", "pred_h5", "pred_h10"
        ),
    }
    summaries = []
    quarterly = []
    for name, source in sources.items():
        destination = OUTPUT / name
        destination.mkdir(parents=True, exist_ok=True)
        predictions = source.join(universe, on=["trade_date", "ts_code"], how="semi").sort("trade_date", "ts_code")
        predictions.write_parquet(destination / "csi500_predictions.parquet", compression="zstd")
        result = run_limited_replacement_policy(CATALOG, destination / "csi500_predictions.parquet", destination / "account", config)
        portfolio = pl.read_parquet(destination / "account/portfolio_daily.parquet")
        summary = {"model": name, **metrics(portfolio), "zero_cost_total_return": result["zero_cost"]["net_total_return"]}
        summaries.append(summary)
        for key, frame in portfolio.with_columns(
            (pl.col("execution_date").dt.year().cast(pl.String) + "Q" + pl.col("execution_date").dt.quarter().cast(pl.String)).alias("quarter")
        ).group_by("quarter", maintain_order=True):
            quarterly.append({"model": name, "quarter": key[0], "net_return": float(np.prod(1 + frame["net_return"].to_numpy()) - 1),
                              "csi500_return": float(np.prod(1 + frame["csi500_return"].to_numpy()) - 1)})
        render_backtest_report(destination / "account/portfolio_daily.parquet", destination / "report", f"{name} — CSI500 Top100 / max 3 replacements")
        print(json.dumps(summary), flush=True)
    pl.DataFrame(summaries).write_csv(OUTPUT / "summary.csv")
    pl.DataFrame(quarterly).write_csv(OUTPUT / "quarterly.csv")
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
