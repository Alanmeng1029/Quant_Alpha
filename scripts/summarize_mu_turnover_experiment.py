#!/usr/bin/env python3
"""Summarize and validate the no-risk mu-minus-turnover experiment."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERIODS = {
    "full": None,
    "2024-2026": "2024-01-01",
    "2025-2026": "2025-01-01",
}


def compound(values: pd.Series) -> float:
    return float((1.0 + values).prod() - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--signals", nargs="+", default=["h1", "h5"])
    return parser.parse_args()


def summarize_period(df: pd.DataFrame, start: str | None) -> dict[str, float | int | str]:
    part = df if start is None else df.loc[df["execution_date"] >= start]
    net_nav = (1.0 + part["net_return"]).cumprod()
    benchmark_nav = (1.0 + part["csi500_return"]).cumprod()
    relative_nav = net_nav / benchmark_nav
    active = part["active_return"]
    years = len(part) / 252.0
    net_total = compound(part["net_return"])
    benchmark_total = compound(part["csi500_return"])
    relative_total = (1.0 + net_total) / (1.0 + benchmark_total) - 1.0
    return {
        "days": len(part),
        "gross_total_return": compound(part["gross_return"]),
        "net_total_return": net_total,
        "csi500_total_return": benchmark_total,
        "relative_total_return": relative_total,
        "annualized_relative_return": (1.0 + relative_total) ** (1.0 / years) - 1.0,
        "annualized_net_return": (1.0 + net_total) ** (1.0 / years) - 1.0,
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252.0)),
        "relative_max_drawdown": float((relative_nav / relative_nav.cummax() - 1.0).min()),
        "max_drawdown": float((net_nav / net_nav.cummax() - 1.0).min()),
        "average_buy_turnover": float(part["buy_turnover"].mean()),
        "average_sell_turnover": float(part["sell_turnover"].mean()),
        "average_transaction_cost": float(part["transaction_cost"].mean()),
        "gross_minus_net_total_return": compound(part["gross_return"]) - net_total,
    }


def main() -> None:
    args = parse_args()
    root = args.root
    rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    navs: dict[str, pd.Series] = {}
    relative_navs: dict[str, pd.Series] = {}

    cases = [(model, signal) for model in ("105", "125") for signal in args.signals]
    for model, horizon in cases:
        case = root / model / horizon
        daily = pd.read_parquet(case / "backtest" / "portfolio_daily.parquet")
        daily["execution_date"] = pd.to_datetime(daily["execution_date"])
        label = f"{model}/{horizon.upper()}"
        navs[label] = daily.set_index("execution_date")["nav"]
        indexed = daily.set_index("execution_date")
        relative_navs[label] = (
            (1.0 + indexed["net_return"]).cumprod()
            / (1.0 + indexed["csi500_return"]).cumprod()
        )

        for period, start in PERIODS.items():
            rows.append({"model": model, "horizon": horizon, "period": period, **summarize_period(daily, start)})

        weights = pd.read_parquet(case / "target_weights.parquet")
        grouped = weights.groupby("execution_date")["target_weight"].agg(["sum", "max", "count"])
        validations.append(
            {
                "model": model,
                "horizon": horizon,
                "days": len(grouped),
                "min_weight_sum": float(grouped["sum"].min()),
                "max_weight_sum": float(grouped["sum"].max()),
                "max_single_weight": float(grouped["max"].max()),
                "min_holding_count": int(grouped["count"].min()),
                "max_holding_count": int(grouped["count"].max()),
                "nan_weight_count": int(weights["target_weight"].isna().sum()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "comparison.csv", index=False)
    pd.DataFrame(validations).to_csv(root / "constraint_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, nav in navs.items():
        ax.plot(nav.index, nav / nav.iloc[0], label=label, linewidth=1.5)
    ax.set_title("No-risk optimizer: mu minus realized turnover cost")
    ax.set_ylabel("Net NAV (normalized)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "net_nav_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, nav in relative_navs.items():
        ax.plot(nav.index, nav, label=label, linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("No-risk optimizer: net NAV relative to CSI 500")
    ax.set_ylabel("Portfolio net NAV / CSI 500 NAV")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "relative_nav_vs_csi500.png", dpi=180)
    plt.close(fig)

    compact = summary.loc[:, [
        "model", "horizon", "period", "net_total_return", "csi500_total_return",
        "relative_total_return", "annualized_relative_return", "information_ratio",
        "relative_max_drawdown", "max_drawdown", "average_buy_turnover",
        "gross_minus_net_total_return",
    ]]
    print(compact.to_string(index=False))
    print("\nconstraint validation")
    print(pd.DataFrame(validations).to_string(index=False))


if __name__ == "__main__":
    main()
