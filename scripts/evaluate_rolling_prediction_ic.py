#!/usr/bin/env python3
"""Evaluate rolling prediction IC with the production trainer's exact labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


HORIZONS = ("h1", "h5", "h10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-index-code", default="000905.SH")
    return parser.parse_args()


def sql_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def load_scored(
    connection: duckdb.DuckDBPyConnection,
    predictions: Path,
    model: str,
    benchmark: str,
) -> pl.DataFrame:
    prediction_path = sql_quote(predictions.resolve())
    benchmark = sql_quote(benchmark)
    query = f"""
    WITH calendar AS (
      SELECT trade_date,row_number() over(order by trade_date) n
      FROM observed_calendar WHERE is_observed_market_day
    ), prediction AS (
      SELECT trade_date,ts_code,pred_h1,pred_h5,pred_h10
      FROM read_parquet('{prediction_path}')
    )
    SELECT p.trade_date,p.ts_code,p.pred_h1,p.pred_h5,p.pred_h10,
      CASE WHEN d1.qfq_open>0 AND d2.qfq_open>0 AND i1.open>0 AND i2.open>0
        AND d1.amount_cny>0 AND d2.amount_cny>0
        AND d1.observation_status='complete_trading' AND d2.observation_status='complete_trading'
        THEN d2.qfq_open/d1.qfq_open-i2.open/i1.open END label_h1,
      CASE WHEN d1.qfq_open>0 AND d6.qfq_open>0 AND i1.open>0 AND i6.open>0
        AND d1.amount_cny>0 AND d6.amount_cny>0
        AND d1.observation_status='complete_trading' AND d6.observation_status='complete_trading'
        THEN d6.qfq_open/d1.qfq_open-i6.open/i1.open END label_h5,
      CASE WHEN d1.qfq_open>0 AND d11.qfq_open>0 AND i1.open>0 AND i11.open>0
        AND d1.amount_cny>0 AND d11.amount_cny>0
        AND d1.observation_status='complete_trading' AND d11.observation_status='complete_trading'
        THEN d11.qfq_open/d1.qfq_open-i11.open/i1.open END label_h10
    FROM prediction p
    JOIN calendar c ON c.trade_date=p.trade_date
    LEFT JOIN calendar ce ON ce.n=c.n+1
    LEFT JOIN calendar c2 ON c2.n=c.n+2
    LEFT JOIN calendar c6 ON c6.n=c.n+6
    LEFT JOIN calendar c11 ON c11.n=c.n+11
    LEFT JOIN daily_qfq d1 ON d1.ts_code=p.ts_code AND d1.trade_date=ce.trade_date
    LEFT JOIN daily_qfq d2 ON d2.ts_code=p.ts_code AND d2.trade_date=c2.trade_date
    LEFT JOIN daily_qfq d6 ON d6.ts_code=p.ts_code AND d6.trade_date=c6.trade_date
    LEFT JOIN daily_qfq d11 ON d11.ts_code=p.ts_code AND d11.trade_date=c11.trade_date
    LEFT JOIN index_daily i1 ON i1.index_code='{benchmark}' AND i1.trade_date=ce.trade_date
    LEFT JOIN index_daily i2 ON i2.index_code='{benchmark}' AND i2.trade_date=c2.trade_date
    LEFT JOIN index_daily i6 ON i6.index_code='{benchmark}' AND i6.trade_date=c6.trade_date
    LEFT JOIN index_daily i11 ON i11.index_code='{benchmark}' AND i11.trade_date=c11.trade_date
    ORDER BY p.trade_date,p.ts_code
    """
    return pl.from_arrow(connection.execute(query).to_arrow_table()).with_columns(pl.lit(model).alias("model"))


def daily_ic(scored: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for horizon in HORIZONS:
        prediction = f"pred_{horizon}"
        label = f"label_{horizon}"
        valid = scored.filter(
            pl.col(prediction).is_finite()
            & pl.col(label).is_not_null()
            & pl.col(label).is_finite()
        )
        frames.append(
            valid.group_by("model", "trade_date")
            .agg(
                pl.len().alias("observations"),
                pl.corr(prediction, label).alias("pearson_ic"),
                pl.corr(prediction, label, method="spearman").alias("rank_ic"),
            )
            .with_columns(pl.lit(horizon).alias("horizon"))
        )
    return pl.concat(frames).sort("model", "horizon", "trade_date")


def summarize(frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(groups, maintain_order=True)
        .agg(
            pl.len().alias("days"),
            pl.col("observations").sum(),
            pl.col("rank_ic").mean().alias("mean_rank_ic"),
            pl.col("rank_ic").std().alias("std_rank_ic"),
            pl.col("pearson_ic").mean().alias("mean_pearson_ic"),
            (pl.col("rank_ic") > 0).mean().alias("positive_rank_ic_ratio"),
        )
        .with_columns(
            (pl.col("mean_rank_ic") / pl.col("std_rank_ic")).alias("rank_icir_unannualized"),
            (pl.col("mean_rank_ic") / pl.col("std_rank_ic") * np.sqrt(252.0)).alias("rank_icir_annualized"),
        )
        .sort(groups)
    )


def markdown_table(frame: pl.DataFrame) -> str:
    columns = frame.columns
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.iter_rows(named=True):
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def block_bootstrap_mean_ci(
    values: np.ndarray,
    block_days: int = 20,
    samples: int = 5_000,
    seed: int = 20260922,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < block_days:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(len(values) - block_days + 1)
    block_count = int(np.ceil(len(values) / block_days))
    means = np.empty(samples)
    for sample in range(samples):
        chosen = rng.choice(starts, size=block_count, replace=True)
        draw = np.concatenate([values[start : start + block_days] for start in chosen])[: len(values)]
        means[sample] = draw.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if len(args.predictions) != len(args.name) or not args.name:
        raise ValueError("provide one --name for every --predictions artifact")
    args.output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(args.catalog), read_only=True)
    try:
        scored = pl.concat(
            [load_scored(connection, path, name, args.benchmark_index_code) for path, name in zip(args.predictions, args.name, strict=True)]
        )
    finally:
        connection.close()

    daily = daily_ic(scored)
    daily.write_parquet(args.output / "daily_ic.parquet", compression="zstd")
    daily.write_csv(args.output / "daily_ic.csv")
    summary = summarize(daily, ["model", "horizon"])
    summary.write_csv(args.output / "summary.csv")

    annual_daily = daily.with_columns(pl.col("trade_date").dt.year().alias("year"))
    annual = summarize(annual_daily, ["model", "horizon", "year"])
    annual.write_csv(args.output / "annual_ic.csv")

    anchor = daily["trade_date"].min()
    offset_daily = daily.with_columns(
        (
            (pl.col("trade_date").dt.year() * 12 + pl.col("trade_date").dt.month())
            - (anchor.year * 12 + anchor.month)
        ).mod(3).alias("quarter_month_offset")
    )
    by_offset = summarize(offset_daily, ["model", "horizon", "quarter_month_offset"])
    by_offset.write_csv(args.output / "ic_by_quarter_month.csv")

    paired = daily.select("model", "horizon", "trade_date", "rank_ic").pivot(
        index=["horizon", "trade_date"], on="model", values="rank_ic"
    )
    if len(args.name) == 2:
        paired = paired.with_columns((pl.col(args.name[1]) - pl.col(args.name[0])).alias("candidate_minus_baseline"))
        paired_rows = []
        for horizon in HORIZONS:
            values = paired.filter(pl.col("horizon") == horizon)["candidate_minus_baseline"].drop_nulls().to_numpy()
            low, high = block_bootstrap_mean_ci(values)
            paired_rows.append({
                "horizon": horizon,
                "days": len(values),
                "mean_rank_ic_difference": float(values.mean()),
                "std_rank_ic_difference": float(values.std(ddof=1)),
                "candidate_win_ratio": float((values > 0).mean()),
                "bootstrap_95pct_low": low,
                "bootstrap_95pct_high": high,
            })
        paired_summary = pl.DataFrame(paired_rows)
    else:
        paired_summary = pl.DataFrame()
    paired.write_csv(args.output / "paired_daily_rank_ic.csv")
    paired_summary.write_csv(args.output / "paired_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, horizon in zip(axes[0], ("h1", "h5"), strict=True):
        for model in args.name:
            selected = daily.filter((pl.col("model") == model) & (pl.col("horizon") == horizon)).sort("trade_date")
            axis.plot(selected["trade_date"], selected["rank_ic"].cum_sum(), label=model)
        axis.set_title(f"Cumulative daily Rank IC — {horizon.upper()}")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis, horizon in zip(axes[1], ("h1", "h5"), strict=True):
        selected = annual.filter(pl.col("horizon") == horizon)
        years = sorted(selected["year"].unique().to_list())
        positions = np.arange(len(years))
        width = 0.8 / len(args.name)
        for index, model in enumerate(args.name):
            values = selected.filter(pl.col("model") == model).sort("year")["mean_rank_ic"]
            axis.bar(positions + (index - (len(args.name) - 1) / 2) * width, values, width, label=model)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, years)
        axis.set_title(f"Calendar-year mean Rank IC — {horizon.upper()}")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.savefig(args.output / "ic_comparison.png", dpi=170)
    plt.close(fig)

    report = f"""# 月度与季度三年窗 LightGBM IC 对比

标签完全复用正式 Rust 训练器口径：信号日 T，T+1 开盘进入，分别在
T+2、T+6、T+11 开盘计算 H1/H5/H10 个股相对 CSI500 的超额收益。

## 全样本

{markdown_table(summary.select("model", "horizon", "days", "mean_rank_ic", "rank_icir_annualized", "mean_pearson_ic", "positive_rank_ic_ratio"))}

## 月度减季度的配对日度 Rank IC

{markdown_table(paired_summary)}

## 分年

{markdown_table(annual.select("model", "horizon", "year", "mean_rank_ic", "rank_icir_annualized", "positive_rank_ic_ratio"))}

## 季度内月份

`quarter_month_offset=0` 为季度首月，两个方案使用相同模型；1和2为月度方案新增重训的月份。

{markdown_table(by_offset.select("model", "horizon", "quarter_month_offset", "days", "mean_rank_ic", "rank_icir_annualized"))}

![IC comparison](ic_comparison.png)
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": summary.to_dicts(), "paired": paired_summary.to_dicts()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
