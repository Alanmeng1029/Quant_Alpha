#!/usr/bin/env python3
"""Import chenditc/investment_data CSI1000 intervals as month-end snapshots."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path

import polars as pl


INDEX_CODE = "000852.SH"
SOURCE_NAME = "chenditc/investment_data"
SOURCE_METHOD = "qlib_interval_month_end_snapshot"


def qlib_to_ts_code(symbol: str) -> str:
    if symbol.startswith("SH"):
        return f"{symbol[2:]}.SH"
    if symbol.startswith("SZ"):
        return f"{symbol[2:]}.SZ"
    if symbol.startswith("BJ"):
        return f"{symbol[2:]}.BJ"
    raise ValueError(f"Unsupported Qlib symbol: {symbol}")


def load_intervals(path: Path, start: date) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["qlib_symbol", "effective_from", "effective_to"],
        schema_overrides={"qlib_symbol": pl.String},
    ).with_columns(
        pl.col("effective_from").str.to_date(strict=True),
        pl.col("effective_to").str.to_date(strict=True),
        pl.col("qlib_symbol").map_elements(qlib_to_ts_code, return_dtype=pl.String).alias("ts_code"),
    )
    frame = frame.filter(pl.col("effective_to") >= pl.lit(start))
    duplicate_intervals = frame.group_by(["ts_code", "effective_from", "effective_to"]).len().filter(pl.col("len") > 1)
    if duplicate_intervals.height:
        raise RuntimeError(f"Duplicate source intervals: {duplicate_intervals.head().to_dicts()}")
    return frame


def month_ends(calendar_path: Path, start: date, source_end: date) -> pl.DataFrame:
    dates = (
        pl.read_parquet(calendar_path)
        .select("trade_date")
        .filter(pl.col("trade_date") >= pl.lit(start))
        .with_columns(pl.col("trade_date").dt.year().alias("year"), pl.col("trade_date").dt.month().alias("month"))
        .group_by(["year", "month"])
        .agg(pl.col("trade_date").max().alias("as_of_date"))
        .sort("as_of_date")
    )
    if source_end > dates["as_of_date"].max():
        current = pl.DataFrame(
            {"year": [source_end.year], "month": [source_end.month], "as_of_date": [source_end]}
        )
        dates = pl.concat([dates, current], how="vertical_relaxed").unique(["year", "month"], keep="last").sort("as_of_date")
    return dates


def build_snapshots(intervals: pl.DataFrame, dates: pl.DataFrame) -> pl.DataFrame:
    snapshots = (
        dates.join(intervals, how="cross")
        .filter(
            (pl.col("effective_from") <= pl.col("as_of_date"))
            & (pl.col("effective_to") >= pl.col("as_of_date"))
        )
        .select(
            pl.lit(INDEX_CODE).alias("index_code"),
            "as_of_date",
            "ts_code",
            pl.lit(None, dtype=pl.String).alias("name"),
            "effective_from",
            "effective_to",
            pl.lit(SOURCE_NAME).alias("source_name"),
            pl.lit(SOURCE_METHOD).alias("source_method"),
        )
        .sort(["as_of_date", "ts_code"])
    )
    duplicates = snapshots.group_by(["as_of_date", "ts_code"]).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise RuntimeError(f"Duplicate snapshot memberships: {duplicates.head().to_dicts()}")
    counts = snapshots.group_by("as_of_date").len().sort("as_of_date")
    if counts.is_empty() or counts["len"].min() < 1000 or counts["len"].max() > 1001:
        raise RuntimeError(f"Unexpected CSI1000 month-end counts: {counts.filter(pl.col('len') != 1000).to_dicts()}")
    return snapshots


def install(snapshots: pl.DataFrame, root: Path) -> dict[str, object]:
    destination_root = root / "lake" / "canonical" / "reference" / "index_constituents"
    backup_root = root / "lake" / "backups" / "index_constituents" / f"csi1000-{datetime.now():%Y%m%dT%H%M%S}"
    installed: list[int] = []
    for year in sorted(snapshots["as_of_date"].dt.year().unique().to_list()):
        destination = destination_root / f"year={year}" / "constituents.parquet"
        existing = pl.read_parquet(destination) if destination.exists() else snapshots.head(0)
        combined = pl.concat(
            [existing.filter(pl.col("index_code") != INDEX_CODE), snapshots.filter(pl.col("as_of_date").dt.year() == year)],
            how="vertical_relaxed",
        ).sort(["as_of_date", "index_code", "ts_code"])
        if combined.select(pl.struct(["index_code", "as_of_date", "ts_code"]).n_unique()).item() != combined.height:
            raise RuntimeError(f"Duplicate primary keys after merging year {year}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = backup_root / f"year={year}" / "constituents.parquet"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            combined.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        installed.append(year)
    return {"years_installed": installed, "backup_root": str(backup_root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("A_stock_database"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    args = parser.parse_args()
    intervals = load_intervals(args.source, args.start)
    source_end = intervals["effective_to"].max()
    dates = month_ends(
        args.data_root / "lake" / "canonical" / "reference" / "observed_calendar.parquet",
        args.start,
        source_end,
    )
    snapshots = build_snapshots(intervals, dates)
    result = install(snapshots, args.data_root)
    counts = snapshots.group_by("as_of_date").len()
    result.update(
        index_code=INDEX_CODE,
        source=SOURCE_NAME,
        start=str(snapshots["as_of_date"].min()),
        end=str(snapshots["as_of_date"].max()),
        snapshot_rows=snapshots.height,
        month_ends=counts.height,
        min_members=counts["len"].min(),
        max_members=counts["len"].max(),
        non_1000_months=counts.filter(pl.col("len") != 1000).sort("as_of_date").to_dicts(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
