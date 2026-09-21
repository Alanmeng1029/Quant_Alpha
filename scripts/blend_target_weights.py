#!/usr/bin/env python3
"""Blend multiple target-weight ledgers into one netted production target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


KEYS = ["trade_date", "execution_date", "ts_code"]


def blend_targets(
    sources: list[Path], allocations: list[float], invested_weight: float = 0.98
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    if len(sources) != len(allocations) or not sources:
        raise ValueError("provide one allocation for every target-weight source")
    if any(weight <= 0 for weight in allocations):
        raise ValueError("allocations must be positive")
    if abs(sum(allocations) - 1.0) > 1e-10:
        raise ValueError("allocations must sum to one")

    contributions: list[pl.DataFrame] = []
    reference_dates: set[tuple[object, object]] | None = None
    source_summaries: list[dict[str, object]] = []
    for source, allocation in zip(sources, allocations, strict=True):
        frame = (
            pl.read_parquet(source)
            .select(KEYS + ["target_weight"])
            .with_columns(
                pl.col("trade_date").cast(pl.Date),
                pl.col("execution_date").cast(pl.Date),
                pl.col("target_weight").cast(pl.Float64),
            )
        )
        if frame.select(pl.struct(KEYS).is_duplicated().any()).item():
            raise ValueError(f"duplicate date/code targets in {source}")
        if frame.filter(~pl.col("target_weight").is_finite() | (pl.col("target_weight") < 0)).height:
            raise ValueError(f"invalid target weights in {source}")
        daily = frame.group_by("trade_date").agg(
            pl.col("target_weight").sum().alias("weight_sum")
        )
        if daily.filter((pl.col("weight_sum") - invested_weight).abs() > 1e-8).height:
            raise ValueError(f"daily target weight does not equal {invested_weight} in {source}")
        dates = set(frame.select("trade_date", "execution_date").unique().iter_rows())
        if reference_dates is None:
            reference_dates = dates
        elif dates != reference_dates:
            raise ValueError("all sleeves must contain identical trade/execution dates")
        contributions.append(
            frame.with_columns((pl.col("target_weight") * allocation).alias("target_weight"))
        )
        source_summaries.append(
            {"target_weights": str(source), "allocation": allocation}
        )

    blended = (
        pl.concat(contributions)
        .group_by(KEYS)
        .agg(pl.col("target_weight").sum())
        .filter(pl.col("target_weight") > 1e-12)
        .sort(KEYS)
    )
    daily = (
        blended.group_by("trade_date")
        .agg(
            pl.col("execution_date").first(),
            pl.col("target_weight").sum().alias("sum_weight"),
            pl.len().alias("holding_count"),
            pl.col("target_weight").max().alias("max_weight"),
            (1.0 / pl.col("target_weight").pow(2).sum()).alias(
                "effective_holding_count"
            ),
        )
        .sort("trade_date")
    )
    if daily.filter((pl.col("sum_weight") - invested_weight).abs() > 1e-8).height:
        raise RuntimeError("blended daily weights do not preserve the invested weight")
    summary: dict[str, object] = {
        "strategy": "target-level netted capital blend",
        "sources": source_summaries,
        "invested_weight": invested_weight,
        "days": daily.height,
        "rows": blended.height,
        "mean_target_holdings": float(daily["holding_count"].mean()),
        "median_target_holdings": float(daily["holding_count"].median()),
        "mean_effective_holdings": float(daily["effective_holding_count"].mean()),
        "mean_max_weight": float(daily["max_weight"].mean()),
        "max_weight": float(daily["max_weight"].max()),
    }
    return blended, daily, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-weights", type=Path, action="append", required=True)
    parser.add_argument("--allocation", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invested-weight", type=float, default=0.98)
    args = parser.parse_args()

    targets, daily, summary = blend_targets(
        args.target_weights, args.allocation, args.invested_weight
    )
    args.output.mkdir(parents=True, exist_ok=True)
    targets.write_parquet(args.output / "target_weights.parquet", compression="zstd")
    daily.write_parquet(args.output / "optimizer_daily.parquet", compression="zstd")
    (args.output / "optimizer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
