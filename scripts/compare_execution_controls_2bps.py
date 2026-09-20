#!/usr/bin/env python3
"""Compare capped, unconstrained, and cost-aware execution at 2 bps each side."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import backtest_targets, render_backtest_report
from scripts.optimize_mu_turnover import optimize_day


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "A_stock_database/lake/catalog/a_share.duckdb"
OUTPUT = ROOT / "results/predict/regression-105-125-execution-2bps-v1"
PREDICTIONS = {
    "old105": ROOT / "results/predict/dos-minute20-model-comparison-v1/daily60_minute45_baseline/csi500_predictions.parquet",
    "new125": ROOT / "results/predict/dos-minute20-model-comparison-v1/daily60_minute45_dos20/csi500_predictions.parquet",
}
PERIODS = {"full": None, "2024-2026": "2024-01-01", "2025-2026": "2025-01-01"}


def metrics(frame: pl.DataFrame) -> dict[str, float | int]:
    net = frame["net_return"].to_numpy()
    benchmark = frame["csi500_return"].to_numpy()
    nav = np.cumprod(1.0 + net)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    active = net - benchmark
    return {
        "days": len(frame),
        "net_return": float(nav[-1] - 1.0),
        "csi500_return": float(benchmark_nav[-1] - 1.0),
        "relative_return": float(nav[-1] / benchmark_nav[-1] - 1.0),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252.0)),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "average_buy_turnover": float(frame["buy_turnover"].mean()),
        "average_sell_turnover": float(frame["sell_turnover"].mean()),
        "average_holding_count": float(frame["holding_count"].mean()),
    }


def optimizer_weights(source: Path, output: Path) -> Path:
    frame = (
        pl.read_parquet(source)
        .with_columns(pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date))
        .filter(pl.col("execution_date").is_not_null() & pl.col("raw_h1").is_finite())
    )
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for key, day in frame.partition_by("trade_date", as_dict=True, maintain_order=True).items():
        trade_date = key[0] if isinstance(key, tuple) else key
        day = day.sort("ts_code")
        codes = day["ts_code"].to_list()
        mu = day["raw_h1"].to_numpy()
        weights, buys, sells, objective = optimize_day(
            codes, mu, previous, invested_weight=0.98, max_weight=0.01,
            buy_cost=2.0 / 10_000.0, sell_cost=2.0 / 10_000.0,
        )
        previous = {code: float(weight) for code, weight in zip(codes, weights) if weight > 1e-12}
        execution_date = day["execution_date"][0]
        rows.extend(
            {"trade_date": trade_date, "execution_date": execution_date,
             "ts_code": code, "target_weight": float(weight)}
            for code, weight in zip(codes, weights) if weight > 1e-12
        )
        diagnostics.append({"trade_date": trade_date, "execution_date": execution_date,
                            "target_buy_turnover": buys, "target_sell_turnover": sells,
                            "objective": objective, "holding_count": len(previous)})
    output.mkdir(parents=True, exist_ok=True)
    target = output / "target_weights.parquet"
    pl.DataFrame(rows).write_parquet(target, compression="zstd")
    pl.DataFrame(diagnostics).write_parquet(output / "optimizer_daily.parquet", compression="zstd")
    return target


def daily_top100_weights(source: Path, output: Path) -> Path:
    frame = (
        pl.read_parquet(source)
        .with_columns(pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date))
        .filter(pl.col("execution_date").is_not_null() & pl.col("raw_h1").is_finite())
        .sort(["trade_date", "raw_h1", "ts_code"], descending=[False, True, False])
        .group_by("trade_date", maintain_order=True)
        .head(100)
        .select("trade_date", "execution_date", "ts_code")
        .with_columns(pl.lit(0.0098).alias("target_weight"))
    )
    output.mkdir(parents=True, exist_ok=True)
    target = output / "target_weights.parquet"
    frame.write_parquet(target, compression="zstd")
    return target


def add_rows(rows: list[dict[str, object]], model: str, method: str, daily_path: Path) -> None:
    daily = pl.read_parquet(daily_path).with_columns(pl.col("execution_date").cast(pl.Date))
    for period, start in PERIODS.items():
        sample = daily if start is None else daily.filter(pl.col("execution_date") >= pl.lit(start).str.to_date())
        rows.append({"model": model, "method": method, "period": period, **metrics(sample)})


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model, prediction in PREDICTIONS.items():
        for method, replacements, exit_rank in (("swap3", 3, 120),):
            target = OUTPUT / model / method
            config = LimitedReplacementConfig(
                target_holdings=100, entry_rank=100, exit_rank=exit_rank,
                max_daily_replacements=replacements, h1_weight=1.0,
                buy_bps=2.0, sell_bps=2.0,
            )
            run_limited_replacement_policy(CATALOG, prediction, target / "backtest", config)
            render_backtest_report(
                target / "backtest/portfolio_daily.parquet", target / "report",
                title=f"{model} regression H1 — {method} — buy/sell 2bps",
            )
            add_rows(rows, model, method, target / "backtest/portfolio_daily.parquet")

        target = OUTPUT / model / "uncontrolled_top100"
        weights = daily_top100_weights(prediction, target)
        backtest_targets(CATALOG, weights, target / "backtest", buy_bps=2.0, sell_bps=2.0)
        render_backtest_report(
            target / "backtest/portfolio_daily.parquet", target / "report",
            title=f"{model} regression H1 — daily Top100 unconstrained — buy/sell 2bps",
            target_weights=weights,
        )
        add_rows(rows, model, "uncontrolled_top100", target / "backtest/portfolio_daily.parquet")

        target = OUTPUT / model / "optimizer"
        weights = optimizer_weights(prediction, target)
        backtest_targets(CATALOG, weights, target / "backtest", buy_bps=2.0, sell_bps=2.0)
        render_backtest_report(
            target / "backtest/portfolio_daily.parquet", target / "report",
            title=f"{model} regression H1 — mu-turnover optimizer — buy/sell 2bps",
            target_weights=weights,
        )
        add_rows(rows, model, "optimizer", target / "backtest/portfolio_daily.parquet")

    result = pl.DataFrame(rows).sort("period", "model", "method")
    result.write_csv(OUTPUT / "comparison.csv")
    (OUTPUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
