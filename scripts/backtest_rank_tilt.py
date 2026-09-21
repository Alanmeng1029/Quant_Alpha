#!/usr/bin/env python3
"""Test rank-based entry sizing without changing selection or replacement rules."""
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


def _metrics(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    daily = pl.read_parquet(path / "portfolio_daily.parquet").sort("execution_date")
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
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    strategy = _read_json(args.strategy_config)
    policy = {**strategy["portfolio"], **strategy["costs"]}
    policy.pop("strategy", None)
    base = LimitedReplacementConfig(**policy)
    variants = {
        "daily_swap3": base,
        "weekly_swap15": replace(base, rebalance_frequency="weekly", max_daily_replacements=15,
                                  daily_buy_budget=.50, daily_sell_budget=.50),
    }
    # Larger tilts interact with the 0.5% minimum admission weight and are
    # intentionally allowed to leave vacancies, producing a more concentrated
    # portfolio rather than forcing the holding count back to 100.
    tilts = (0.0, .05, .10, .15, .30, .50)
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for variant, variant_config in variants.items():
        for tilt in tilts:
            sizing = "equal" if tilt == 0 else "rank_tilt"
            config = replace(variant_config, entry_sizing=sizing, rank_tilt=tilt)
            name = "equal" if tilt == 0 else f"rank_tilt_{int(tilt * 100)}"
            target = output / variant / name
            result = run_limited_replacement_policy(args.catalog.expanduser(), args.predictions.expanduser(),
                                                    target, config)
            rows.append({"variant": variant, "sizing": name, "rank_tilt": tilt,
                         **_metrics(target, result)})
            annual = pl.read_parquet(target / "annual_metrics.parquet")
            annual_rows.extend({"variant": variant, "sizing": name, "rank_tilt": tilt, **row}
                               for row in annual.to_dicts())
    pl.DataFrame(rows).write_csv(output / "summary.csv")
    pl.DataFrame(annual_rows).write_csv(output / "annual_summary.csv")
    (output / "manifest.json").write_text(json.dumps({
        "predictions": str(args.predictions.expanduser().resolve()),
        "interpretation": "rank 1 entry size is 1+tilt times equal size; rank 100 is 1-tilt",
        "tilts": tilts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-config", type=Path, default=Path("configs/production_strategy_csi500_top100_v1.json"))
    parser.add_argument("--catalog", type=Path, default=Path("A_stock_database/lake/catalog/a_share.duckdb"))
    parser.add_argument("--predictions", type=Path, default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/weekly_comparison/predictions/blend_50_50_csi500.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/rank_tilt_experiment"))
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
