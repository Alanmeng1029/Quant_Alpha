#!/usr/bin/env python3
"""Compare two rolling-refit prediction/backtest artifacts on common dates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


TRADING_DAYS = 252.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-backtest", type=Path, required=True)
    parser.add_argument("--candidate-backtest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="quarterly_3y")
    parser.add_argument("--candidate-name", default="monthly_3y")
    return parser.parse_args()


def period_metrics(frame: pl.DataFrame) -> dict[str, float | int]:
    net = frame["net_return"].to_numpy()
    benchmark = frame["benchmark_return"].to_numpy()
    active = net - benchmark
    net_total = float(np.prod(1.0 + net) - 1.0)
    benchmark_total = float(np.prod(1.0 + benchmark) - 1.0)
    annualized = float((1.0 + net_total) ** (TRADING_DAYS / len(net)) - 1.0)
    volatility = float(np.std(net, ddof=1) * np.sqrt(TRADING_DAYS))
    active_std = float(np.std(active, ddof=1))
    information_ratio = float(np.mean(active) / active_std * np.sqrt(TRADING_DAYS))
    nav = np.cumprod(1.0 + net)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    return {
        "days": len(net),
        "net_total_return": net_total,
        "benchmark_total_return": benchmark_total,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "information_ratio": information_ratio,
        "max_drawdown": float(drawdown.min()),
        "average_buy_turnover": float(frame["buy_turnover"].mean()),
        "average_sell_turnover": float(frame["sell_turnover"].mean()),
        "total_transaction_cost": float(frame["transaction_cost"].sum()),
    }


def load_backtest(path: Path, name: str) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .select(
            "execution_date",
            "net_return",
            "benchmark_return",
            "buy_turnover",
            "sell_turnover",
            "transaction_cost",
        )
        .rename({column: f"{name}_{column}" for column in (
            "net_return", "benchmark_return", "buy_turnover", "sell_turnover", "transaction_cost"
        )})
    )


def markdown_table(frame: pl.DataFrame, percent_columns: set[str]) -> str:
    columns = frame.columns
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.iter_rows(named=True):
        rendered = []
        for column in columns:
            value = values[column]
            if isinstance(value, float):
                rendered.append(f"{value:.2%}" if column in percent_columns else f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = load_backtest(args.baseline_backtest, "baseline")
    candidate = load_backtest(args.candidate_backtest, "candidate")
    common = baseline.join(candidate, on="execution_date", how="inner").sort("execution_date")
    if common.height != baseline.height or common.height != candidate.height:
        raise ValueError("backtests do not have identical execution dates")

    summaries = []
    annual = []
    for label, prefix in ((args.baseline_name, "baseline"), (args.candidate_name, "candidate")):
        selected = common.select(
            "execution_date",
            pl.col(f"{prefix}_net_return").alias("net_return"),
            pl.col(f"{prefix}_benchmark_return").alias("benchmark_return"),
            pl.col(f"{prefix}_buy_turnover").alias("buy_turnover"),
            pl.col(f"{prefix}_sell_turnover").alias("sell_turnover"),
            pl.col(f"{prefix}_transaction_cost").alias("transaction_cost"),
        )
        summaries.append({"model": label, **period_metrics(selected)})
        for year, year_frame in selected.with_columns(pl.col("execution_date").dt.year().alias("year")).group_by("year", maintain_order=True):
            annual.append({"model": label, "year": year[0], **period_metrics(year_frame)})

    summary = pl.DataFrame(summaries)
    annual_frame = pl.DataFrame(annual)
    summary.write_csv(args.output / "summary.csv")
    annual_frame.write_csv(args.output / "annual_metrics.csv")

    prediction = (
        pl.scan_parquet(args.baseline_predictions)
        .select("trade_date", "ts_code", pl.col("pred_h1").alias("baseline_h1"), pl.col("pred_h5").alias("baseline_h5"))
        .join(
            pl.scan_parquet(args.candidate_predictions).select(
                "trade_date", "ts_code", pl.col("pred_h1").alias("candidate_h1"), pl.col("pred_h5").alias("candidate_h5")
            ),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        .collect()
        .with_columns(
            (((pl.col("trade_date").dt.year() * 12 + pl.col("trade_date").dt.month() - (2021 * 12 + 4)) % 3).alias("quarter_month_offset")),
            pl.col("baseline_h1").rank(method="ordinal", descending=True).over("trade_date").alias("baseline_rank"),
            pl.col("candidate_h1").rank(method="ordinal", descending=True).over("trade_date").alias("candidate_rank"),
        )
    )
    similarity = (
        prediction.group_by("trade_date")
        .agg(
            pl.col("quarter_month_offset").first(),
            pl.corr("baseline_h1", "candidate_h1", method="spearman").alias("h1_rank_correlation"),
            pl.corr("baseline_h5", "candidate_h5", method="spearman").alias("h5_rank_correlation"),
            ((pl.col("baseline_rank") <= 100) & (pl.col("candidate_rank") <= 100)).sum().truediv(100).alias("top100_overlap"),
        )
        .sort("trade_date")
    )
    similarity.write_csv(args.output / "daily_prediction_similarity.csv")
    similarity_by_offset = similarity.group_by("quarter_month_offset").agg(
        pl.len().alias("days"),
        pl.mean("h1_rank_correlation"),
        pl.mean("h5_rank_correlation"),
        pl.mean("top100_overlap"),
    ).sort("quarter_month_offset")
    similarity_by_offset.write_csv(args.output / "prediction_similarity_by_quarter_month.csv")

    dates = common["execution_date"].to_list()
    baseline_net = common["baseline_net_return"].to_numpy()
    candidate_net = common["candidate_net_return"].to_numpy()
    benchmark = common["baseline_benchmark_return"].to_numpy()
    baseline_nav = np.cumprod(1.0 + baseline_net)
    candidate_nav = np.cumprod(1.0 + candidate_net)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    baseline_dd = baseline_nav / np.maximum.accumulate(baseline_nav) - 1.0
    candidate_dd = candidate_nav / np.maximum.accumulate(candidate_nav) - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].plot(dates, baseline_nav, label=args.baseline_name)
    axes[0].plot(dates, candidate_nav, label=args.candidate_name)
    axes[0].plot(dates, benchmark_nav, label="CSI500", alpha=0.7)
    axes[0].set_title("Net asset value")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].fill_between(dates, baseline_dd, 0, alpha=0.35, label=args.baseline_name)
    axes[1].fill_between(dates, candidate_dd, 0, alpha=0.35, label=args.candidate_name)
    axes[1].set_title("Drawdown")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    annual_returns = annual_frame.pivot(index="year", on="model", values="net_total_return").sort("year")
    years = annual_returns["year"].to_list()
    positions = np.arange(len(years))
    width = 0.38
    axes[2].bar(positions - width / 2, annual_returns[args.baseline_name], width, label=args.baseline_name)
    axes[2].bar(positions + width / 2, annual_returns[args.candidate_name], width, label=args.candidate_name)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(positions, years)
    axes[2].set_title("Calendar-year net return")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(args.output / "comparison.png", dpi=170)
    plt.close(fig)

    baseline_summary, candidate_summary = summaries
    difference = {
        key: candidate_summary[key] - baseline_summary[key]
        for key in baseline_summary
        if key not in {"model", "days"}
    }
    report = f"""# Rolling LightGBM comparison: {args.candidate_name} versus {args.baseline_name}

Both variants use identical factors, 756 trading-day training windows, label lag,
portfolio optimizer, transaction costs, execution rules, and common OOS dates.

## Full-period result

{markdown_table(summary, {"net_total_return", "benchmark_total_return", "annualized_return", "annualized_volatility", "max_drawdown", "average_buy_turnover", "average_sell_turnover", "total_transaction_cost"})}

Candidate-minus-baseline differences:

```json
{json.dumps(difference, ensure_ascii=False, indent=2)}
```

## Calendar-year result

{markdown_table(annual_frame.select("model", "year", "net_total_return", "information_ratio", "max_drawdown", "average_buy_turnover"), {"net_total_return", "max_drawdown", "average_buy_turnover"})}

## Prediction similarity by month within the baseline quarter

Offset 0 is the quarter-start month, when both protocols train on the same data.
Offsets 1 and 2 are the months where the monthly protocol refreshes but the baseline
continues using its quarter-start model.

{markdown_table(similarity_by_offset, {"top100_overlap"})}

![Comparison](comparison.png)
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summaries, "prediction_similarity": similarity_by_offset.to_dicts()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
