"""Backtest CSI500 and CSI1000 as independent strategies, then blend 80/20."""
from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import polars as pl

from a_share_data.dual_sleeve import CSI500, CSI1000
from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import render_backtest_report
from backtest_raw_dual_sleeve import targets


def sleeve_config(
    holdings: int,
    replacement_rate: float,
    initial_capital: float,
    max_weight: float,
    daily_trade_budget: float,
) -> LimitedReplacementConfig:
    return LimitedReplacementConfig(
        target_holdings=holdings,
        entry_rank=holdings,
        exit_rank=holdings,
        max_daily_replacements=max(1, ceil(holdings * replacement_rate)),
        max_weight=max_weight,
        rebalance_to_weight=max_weight * 0.93,
        min_new_weight=0.5 / holdings,
        cash_reserve=0.02,
        daily_buy_budget=daily_trade_budget,
        daily_sell_budget=daily_trade_budget,
        h1_weight=0.5,
        entry_sizing="cash_balanced",
        rank_tilt=0.0,
        lot_size=100,
        initial_capital=initial_capital,
        buy_bps=2.1,
        sell_bps=7.1,
    )


def blend_accounts(csi500_path: Path, csi1000_path: Path, output: Path) -> pl.DataFrame:
    left = pl.read_parquet(csi500_path).sort("signal_date")
    right = pl.read_parquet(csi1000_path).sort("signal_date")
    joined = left.join(right, on=["signal_date", "execution_date", "next_execution_date"], suffix="_1000")
    result = joined.select("signal_date", "execution_date", "next_execution_date").with_columns(
        (0.8 * joined["gross_return"] + 0.2 * joined["gross_return_1000"]).alias("gross_return"),
        (0.8 * joined["net_return"] + 0.2 * joined["net_return_1000"]).alias("net_return"),
        (0.8 * joined["transaction_cost"] + 0.2 * joined["transaction_cost_1000"]).alias("transaction_cost"),
        (0.8 * joined["buy_turnover"] + 0.2 * joined["buy_turnover_1000"]).alias("buy_turnover"),
        (0.8 * joined["sell_turnover"] + 0.2 * joined["sell_turnover_1000"]).alias("sell_turnover"),
        (joined["holding_count"] + joined["holding_count_1000"]).alias("holding_count"),
        (0.8 * joined["cash_weight"] + 0.2 * joined["cash_weight_1000"]).alias("cash_weight"),
        joined["csi500_return"],
    )
    net = result["net_return"].to_numpy()
    benchmark = result["csi500_return"].to_numpy()
    nav = np.cumprod(1.0 + net)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    result = result.with_columns(
        pl.Series("cash", result["cash_weight"].to_numpy() * nav * 10_000_000),
        pl.Series("equity", nav * 10_000_000),
        pl.Series("nav", nav),
        pl.Series("csi500_nav", benchmark_nav),
        pl.Series("active_return", net - benchmark),
    )
    output.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output / "portfolio_daily.parquet", compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csi500-holdings", type=int, default=80)
    parser.add_argument("--csi1000-holdings", type=int, required=True)
    parser.add_argument("--csi500-turnover-rate", type=float, default=0.03)
    parser.add_argument("--csi1000-turnover-rate", type=float, default=0.02)
    parser.add_argument("--csi500-daily-trade-budget", type=float, default=0.10)
    parser.add_argument("--csi1000-daily-trade-budget", type=float, default=0.10)
    parser.add_argument(
        "--targets",
        type=Path,
        help="Reuse an existing dual_sleeve_targets.parquet instead of generating selections.",
    )
    args = parser.parse_args()

    if args.targets is None:
        target_path = targets(
            args.predictions, args.catalog, args.output,
            0, 0, None, 1, 1, args.csi500_holdings, args.csi1000_holdings,
            args.csi500_turnover_rate, args.csi1000_turnover_rate,
        )
    else:
        target_path = args.targets
    target_frame = pl.read_parquet(target_path)
    sleeve_root = args.output / "sleeves"
    paths: dict[str, Path] = {}
    for sleeve in (CSI500, CSI1000):
        path = sleeve_root / sleeve.replace(".", "_") / "targets.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        target_frame.filter(pl.col("sleeve") == sleeve).write_parquet(path, compression="zstd")
        paths[sleeve] = path

    configs = {
        CSI500: sleeve_config(
            args.csi500_holdings, args.csi500_turnover_rate, 8_000_000, 0.0375,
            args.csi500_daily_trade_budget,
        ),
        CSI1000: sleeve_config(
            args.csi1000_holdings, args.csi1000_turnover_rate, 2_000_000, 0.15,
            args.csi1000_daily_trade_budget,
        ),
    }
    summaries = {}
    for sleeve in (CSI500, CSI1000):
        destination = sleeve_root / sleeve.replace(".", "_") / "account"
        summaries[sleeve] = run_limited_replacement_policy(args.catalog, paths[sleeve], destination, configs[sleeve])

    combined = blend_accounts(
        sleeve_root / "000905_SH" / "account" / "portfolio_daily.parquet",
        sleeve_root / "000852_SH" / "account" / "portfolio_daily.parquet",
        args.output / "account",
    )
    report = render_backtest_report(
        args.output / "account" / "portfolio_daily.parquet",
        args.output / "report",
        f"Independent CSI500 80% + CSI1000 20% (Top{args.csi1000_holdings})",
    )
    payload = {
        "method": "constant-weight daily return blend of two independently traded sleeves",
        "weights": {CSI500: 0.8, CSI1000: 0.2},
        "turnover_rate_targets": {CSI500: args.csi500_turnover_rate, CSI1000: args.csi1000_turnover_rate},
        "daily_trade_budgets": {
            CSI500: args.csi500_daily_trade_budget,
            CSI1000: args.csi1000_daily_trade_budget,
        },
        "reused_targets": str(args.targets.resolve()) if args.targets else None,
        "sleeve_summaries": summaries,
        "combined_report": report,
        "days": combined.height,
    }
    (args.output / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
