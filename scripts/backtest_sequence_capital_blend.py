#!/usr/bin/env python3
"""Backtest a 50/50 capital split between independent LSTM and LGBM sleeves."""
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


def _combine(left: pl.DataFrame, right: pl.DataFrame, sleeve_capital: float) -> pl.DataFrame:
    keys = ["signal_date", "execution_date", "next_execution_date"]
    right = right.select(*keys, "equity", "net_return", "transaction_cost", "buy_turnover",
                         "sell_turnover").rename({name: f"r_{name}" for name in
                                                  ("equity", "net_return", "transaction_cost",
                                                   "buy_turnover", "sell_turnover")})
    joined = left.join(right, on=keys, how="inner").sort("execution_date")
    if joined.height != left.height or joined.height != right.height:
        raise ValueError("the two sleeves do not have identical trading calendars")
    l_equity = joined["equity"].to_numpy()
    r_equity = joined["r_equity"].to_numpy()
    l_previous = np.r_[sleeve_capital, l_equity[:-1]]
    r_previous = np.r_[sleeve_capital, r_equity[:-1]]
    previous = l_previous + r_previous
    equity = l_equity + r_equity
    net_return = equity / previous - 1.0
    transaction_cost = (joined["transaction_cost"].to_numpy() * l_previous +
                        joined["r_transaction_cost"].to_numpy() * r_previous) / previous
    buy_turnover = (joined["buy_turnover"].to_numpy() * l_previous +
                    joined["r_buy_turnover"].to_numpy() * r_previous) / previous
    sell_turnover = (joined["sell_turnover"].to_numpy() * l_previous +
                     joined["r_sell_turnover"].to_numpy() * r_previous) / previous
    benchmark = joined["csi500_return"].to_numpy()
    return joined.select(*keys, "csi500_return", "csi500_nav").with_columns(
        pl.Series("net_return", net_return),
        pl.Series("active_return", net_return - benchmark),
        pl.Series("transaction_cost", transaction_cost),
        pl.Series("buy_turnover", buy_turnover),
        pl.Series("sell_turnover", sell_turnover),
        pl.Series("equity", equity),
        pl.Series("nav", equity / (2 * sleeve_capital)),
        pl.Series("lstm_sleeve_weight", l_equity / equity),
        pl.Series("lgbm_sleeve_weight", r_equity / equity),
    )


def _fixed_weight_diagnostic(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, Any]:
    """Daily 50/50 return mix; excludes any extra cost for reallocating sleeves."""
    left = left.sort("execution_date")
    right = right.sort("execution_date")
    if left["execution_date"].to_list() != right["execution_date"].to_list():
        raise ValueError("the two sleeves do not have identical trading calendars")
    returns = .5 * left["net_return"].to_numpy() + .5 * right["net_return"].to_numpy()
    benchmark = left["csi500_return"].to_numpy()
    nav = np.cumprod(1 + returns)
    active = returns - benchmark
    return {
        "days": len(nav), "final_nav": float(nav[-1]), "net_total_return": float(nav[-1] - 1),
        "net_annualized_return": float(nav[-1] ** (252 / len(nav)) - 1),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
    }


def _metrics(frame: pl.DataFrame) -> dict[str, Any]:
    nav = frame["nav"].to_numpy()
    active = frame["active_return"].to_numpy()
    years = frame.height / 252
    return {
        "days": frame.height,
        "final_nav": float(nav[-1]),
        "net_total_return": float(nav[-1] - 1),
        "net_annualized_return": float(nav[-1] ** (1 / years) - 1),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_buy_turnover": float(frame["buy_turnover"].sum()),
        "total_sell_turnover": float(frame["sell_turnover"].sum()),
        "transaction_cost_rate_sum": float(frame["transaction_cost"].sum()),
        "ending_lstm_weight": float(frame["lstm_sleeve_weight"][-1]),
        "ending_lgbm_weight": float(frame["lgbm_sleeve_weight"][-1]),
    }


def _annual(frame: pl.DataFrame, variant: str, mode: str) -> list[dict[str, Any]]:
    return frame.with_columns(pl.col("execution_date").dt.year().alias("year")).group_by("year").agg(
        ((1 + pl.col("net_return")).product() - 1).alias("net_return")
    ).sort("year").with_columns(pl.lit(variant).alias("variant"), pl.lit(mode).alias("mode")).select(
        "variant", "mode", "year", "net_return").to_dicts()


def run(args: argparse.Namespace) -> dict[str, Any]:
    strategy = _read_json(args.strategy_config)
    policy = {**strategy["portfolio"], **strategy["costs"]}
    policy.pop("strategy", None)
    base = LimitedReplacementConfig(**policy)
    sleeve_capital = base.initial_capital / 2
    base = replace(base, initial_capital=sleeve_capital)
    variants = {
        "daily_swap3": base,
        "weekly_swap3": replace(base, rebalance_frequency="weekly"),
        "weekly_swap15": replace(base, rebalance_frequency="weekly", max_daily_replacements=15,
                                 daily_buy_budget=.50, daily_sell_budget=.50),
    }
    catalog = args.catalog.expanduser()
    source = args.comparison_root.expanduser()
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for variant, config in variants.items():
        target = output / variant
        lstm = run_limited_replacement_policy(catalog, source / "predictions/lstm_csi500.parquet",
                                              target / "sleeves/lstm", config)
        lgbm = run_limited_replacement_policy(catalog, source / "predictions/lgbm_csi500.parquet",
                                              target / "sleeves/lgbm", config)
        for fee_mode, subdir in (("charged", Path()), ("zero_cost", Path("zero_cost"))):
            left = pl.read_parquet(target / "sleeves/lstm" / subdir / "portfolio_daily.parquet")
            right = pl.read_parquet(target / "sleeves/lgbm" / subdir / "portfolio_daily.parquet")
            combined = _combine(left, right, sleeve_capital)
            destination = target / "capital_50_50" / subdir
            destination.mkdir(parents=True, exist_ok=True)
            combined.write_parquet(destination / "portfolio_daily.parquet", compression="zstd")
            metrics = _metrics(combined)
            (destination / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({"variant": variant, "mode": fee_mode, **metrics})
            annual_rows.extend(_annual(combined, variant, fee_mode))
            if fee_mode == "charged":
                fixed = _fixed_weight_diagnostic(left, right)
                rows.append({"variant": variant, "mode": "fixed_daily_50_50_diagnostic",
                             **fixed, "total_buy_turnover": None, "total_sell_turnover": None,
                             "transaction_cost_rate_sum": None, "ending_lstm_weight": .5,
                             "ending_lgbm_weight": .5})
        (target / "manifest.json").write_text(json.dumps({
            "method": "independent_sleeves", "initial_weights": {"lstm": .5, "lgbm": .5},
            "sleeve_capital": sleeve_capital, "lstm": lstm, "lgbm": lgbm,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    pl.DataFrame(rows).write_csv(output / "summary.csv")
    pl.DataFrame(annual_rows).write_csv(output / "annual_summary.csv")
    return {"output": str(output), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-config", type=Path, default=Path("configs/production_strategy_csi500_top100_v1.json"))
    parser.add_argument("--catalog", type=Path, default=Path("A_stock_database/lake/catalog/a_share.duckdb"))
    parser.add_argument("--comparison-root", type=Path, default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/weekly_comparison"))
    parser.add_argument("--output", type=Path, default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/capital_blend_50_50"))
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
