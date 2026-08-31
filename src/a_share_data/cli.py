from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl


DEFAULT_DATABASE_DIR = "A_stock_database"
EXPECTED_BARS_PER_DAY = 241


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def minute_source(self) -> Path:
        return self.root / "minute"

    @property
    def adjustment_source(self) -> Path:
        return self.root / "复权因子" / "复权因子_前复权"

    @property
    def stock_list(self) -> Path:
        return self.root / "股票列表_沪深.csv"

    @property
    def lake(self) -> Path:
        return self.root / "lake"

    @property
    def canonical_minute(self) -> Path:
        return self.lake / "canonical" / "minute"

    @property
    def canonical_daily(self) -> Path:
        return self.lake / "canonical" / "daily_aggregated"

    @property
    def canonical_adjustment(self) -> Path:
        return self.lake / "canonical" / "adjustment"

    @property
    def reference(self) -> Path:
        return self.lake / "canonical" / "reference"

    @property
    def trading_universe(self) -> Path:
        return self.reference / "trading_universe"

    @property
    def index_constituents(self) -> Path:
        return self.reference / "index_constituents"

    @property
    def index_daily(self) -> Path:
        return self.lake / "canonical" / "index_daily"

    @property
    def quality(self) -> Path:
        return self.lake / "quality"

    @property
    def staging(self) -> Path:
        return self.lake / "staging"

    @property
    def backups(self) -> Path:
        return self.lake / "backups"

    @property
    def catalog(self) -> Path:
        return self.lake / "catalog" / "a_share.duckdb"

    def ensure_layout(self) -> None:
        for path in (
            self.canonical_minute,
            self.canonical_daily,
            self.canonical_adjustment,
            self.reference,
            self.trading_universe,
            self.index_constituents,
            self.index_daily,
            self.quality,
            self.staging,
            self.backups,
            self.lake / "derived" / "factors",
            self.catalog.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def sql_literal(value: str) -> str:
    return value.replace("'", "''")


def connect(temp_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("SET preserve_insertion_order = false")
    # Full-history validation includes distinct-key checks across tens of
    # billions of bars.  Keep its working set bounded and let DuckDB spill to
    # the lake staging area rather than competing with the operating system.
    conn.execute("SET memory_limit = '8GB'")
    conn.execute("SET threads = 2")
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        conn.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    return conn


def parse_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def years_in_range(start: str | None, end: str | None, source: Path) -> list[int]:
    source_years = sorted(
        int(match.group(1))
        for child in source.glob("*_1min")
        if (match := re.fullmatch(r"(\d{4})_1min", child.name))
    )
    if not source_years:
        raise FileNotFoundError(f"No yearly minute directories under {source}")
    start_year = int(start[:4]) if start else source_years[0]
    end_year = int(end[:4]) if end else source_years[-1]
    return [year for year in source_years if start_year <= year <= end_year]


def canonical_years_in_range(paths: Paths, start: str | None, end: str | None) -> list[int]:
    """Return years currently represented in the canonical minute lake."""
    available = sorted(
        int(match.group(1))
        for child in paths.canonical_minute.glob("year=*")
        if (match := re.fullmatch(r"year=(\d{4})", child.name))
    )
    if not available:
        raise FileNotFoundError("No canonical minute year partitions found")
    start_year = int(start[:4]) if start else available[0]
    end_year = int(end[:4]) if end else available[-1]
    return [year for year in available if start_year <= year <= end_year]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quick_file_metadata(path: Path, include_checksum: bool) -> dict[str, object]:
    stat = path.stat()
    row: dict[str, object] = {
        "source_file": str(path.resolve()),
        "source_year": int(path.parent.name[:4]),
        "source_symbol": path.stem.rsplit("_", 1)[0],
        "byte_size": stat.st_size,
        "modified_at_ns": stat.st_mtime_ns,
        "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
        "sha256": sha256_file(path) if include_checksum else None,
        "inventoried_at": utc_now(),
    }
    return row


def write_polars_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    frame.write_parquet(temp, compression="zstd", statistics=True)
    os.replace(temp, path)


def cmd_inventory(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    files = sorted(paths.minute_source.glob("*_1min/*.csv"))
    if args.source_dir:
        source_dir = Path(args.source_dir)
        files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No minute CSV files found")
    records = [quick_file_metadata(path, args.checksum) for path in files]
    frame = pl.DataFrame(records)
    output = paths.quality / "source_inventory.parquet"
    write_polars_parquet(frame, output)
    total_bytes = sum(record["byte_size"] for record in records)
    print(json.dumps({"files": len(records), "bytes": total_bytes, "output": str(output)}, ensure_ascii=False))


def minute_raw_cte(source_glob: Path, start: str | None, end: str | None) -> str:
    where_dates = []
    if start:
        where_dates.append(f"CAST(parsed_datetime AS DATE) >= DATE '{sql_literal(start)}'")
    if end:
        where_dates.append(f"CAST(parsed_datetime AS DATE) <= DATE '{sql_literal(end)}'")
    date_filter = " AND ".join(where_dates) if where_dates else "TRUE"
    return f"""
    WITH raw AS (
        SELECT
            trim(\"时间\") AS raw_datetime,
            lower(trim(\"代码\")) AS source_symbol,
            try_cast(\"开盘价\" AS DOUBLE) AS open,
            try_cast(\"最高价\" AS DOUBLE) AS high,
            try_cast(\"最低价\" AS DOUBLE) AS low,
            try_cast(\"收盘价\" AS DOUBLE) AS close,
            try_cast(\"成交量\" AS BIGINT) AS volume_lot,
            try_cast(\"成交额\" AS DOUBLE) AS amount_cny,
            filename AS source_file
        FROM read_csv_auto(
            '{sql_path(source_glob)}',
            header = true,
            filename = true,
            union_by_name = true,
            all_varchar = true,
            strict_mode = true
        )
    ), parsed AS (
        SELECT *,
            coalesce(
                try_strptime(replace(raw_datetime, '/', '-'), '%Y-%m-%d %H:%M:%S'),
                try_strptime(replace(raw_datetime, '/', '-'), '%Y-%m-%d %H:%M')
            ) AS parsed_datetime
        FROM raw
    ), normalized AS (
        SELECT *,
            CASE
                WHEN left(source_symbol, 2) = 'sh' THEN substr(source_symbol, 3, 6) || '.SH'
                WHEN left(source_symbol, 2) = 'sz' THEN substr(source_symbol, 3, 6) || '.SZ'
                ELSE NULL
            END AS ts_code,
            CASE
                WHEN parsed_datetime IS NULL THEN NULL
                WHEN extract(hour FROM parsed_datetime) = 9
                     AND extract(minute FROM parsed_datetime) >= 30
                    THEN extract(minute FROM parsed_datetime) - 30
                WHEN extract(hour FROM parsed_datetime) IN (10, 11)
                    THEN extract(hour FROM parsed_datetime) * 60 + extract(minute FROM parsed_datetime) - 570
                WHEN extract(hour FROM parsed_datetime) = 13
                     AND extract(minute FROM parsed_datetime) >= 1
                    THEN 121 + extract(minute FROM parsed_datetime) - 1
                WHEN extract(hour FROM parsed_datetime) = 14
                    THEN 180 + extract(minute FROM parsed_datetime)
                WHEN extract(hour FROM parsed_datetime) = 15
                     AND extract(minute FROM parsed_datetime) = 0
                    THEN 240
                ELSE NULL
            END AS minute_index
        FROM parsed
    )
    """, date_filter


def source_invalid_query(source_glob: Path, start: str | None, end: str | None) -> str:
    cte, date_filter = minute_raw_cte(source_glob, start, end)
    return f"""
    {cte}
    SELECT
        coalesce(ts_code, source_symbol) AS instrument,
        raw_datetime,
        source_file,
        CASE
            WHEN parsed_datetime IS NULL THEN 'unparseable_datetime'
            WHEN ts_code IS NULL THEN 'unsupported_exchange_or_symbol'
            WHEN minute_index IS NULL THEN 'outside_expected_session'
            WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                THEN 'unparseable_price'
            WHEN volume_lot IS NULL OR amount_cny IS NULL THEN 'unparseable_volume_or_amount'
            WHEN open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 THEN 'non_positive_price'
            WHEN high < greatest(open, close) OR low > least(open, close) THEN 'invalid_ohlc'
            WHEN volume_lot < 0 OR amount_cny < 0 THEN 'negative_volume_or_amount'
            ELSE NULL
        END AS reason,
        '{utc_now()}' AS quarantined_at
    FROM normalized
    WHERE {date_filter}
      AND (
        parsed_datetime IS NULL OR ts_code IS NULL OR minute_index IS NULL
        OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
        OR volume_lot IS NULL OR amount_cny IS NULL
        OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
        OR high < greatest(open, close) OR low > least(open, close)
        OR volume_lot < 0 OR amount_cny < 0
      )
    """


def source_valid_query(source_glob: Path, start: str | None, end: str | None) -> str:
    cte, date_filter = minute_raw_cte(source_glob, start, end)
    return f"""
    {cte}
    SELECT
        ts_code,
        parsed_datetime AS datetime,
        CAST(parsed_datetime AS DATE) AS trade_date,
        minute_index::UTINYINT AS minute_index,
        open,
        high,
        low,
        close,
        volume_lot,
        volume_lot * 100 AS volume_share,
        amount_cny,
        year(CAST(parsed_datetime AS DATE))::INTEGER AS year,
        lpad(month(CAST(parsed_datetime AS DATE))::VARCHAR, 2, '0') AS month
    FROM normalized
    WHERE {date_filter}
      AND parsed_datetime IS NOT NULL
      AND ts_code IS NOT NULL
      AND minute_index IS NOT NULL
      AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL
      AND volume_lot IS NOT NULL AND amount_cny IS NOT NULL
      AND open > 0 AND high > 0 AND low > 0 AND close > 0
      AND high >= greatest(open, close) AND low <= least(open, close)
      AND volume_lot >= 0 AND amount_cny >= 0
    """


def find_partition_dirs(root: Path) -> Iterable[Path]:
    for year_dir in root.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            yield from month_dir.glob("trade_date=*")


def has_mixed_newlines(path: Path) -> bool:
    """DuckDB rejects a CSV that mixes CRLF and LF record endings."""
    data = path.read_bytes()
    return b"\r\n" in data and b"\n" in data.replace(b"\r\n", b"")


def prepare_csv_source(source_dir: Path, run_root: Path) -> tuple[Path, int]:
    """Create a read-only staging view, normalizing only mixed-newline files."""
    prepared = run_root / "prepared_sources" / source_dir.name
    prepared.mkdir(parents=True, exist_ok=True)
    normalized = 0
    for source_file in sorted(source_dir.glob("*.csv")):
        destination = prepared / source_file.name
        if has_mixed_newlines(source_file):
            destination.write_bytes(source_file.read_bytes().replace(b"\r\n", b"\n"))
            normalized += 1
        else:
            destination.symlink_to(source_file.resolve())
    return prepared, normalized


def install_minute_partitions(paths: Paths, staged_root: Path, replace: bool, run_id: str) -> tuple[int, int]:
    installed = 0
    skipped = 0
    for source in sorted(find_partition_dirs(staged_root)):
        relative = source.relative_to(staged_root)
        destination = paths.canonical_minute / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not replace:
                skipped += 1
                continue
            backup = paths.backups / "minute" / run_id / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(backup))
        shutil.move(str(source), str(destination))
        installed += 1
    return installed, skipped


def cmd_backfill_minute(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    if args.source_dir:
        source_dirs = [Path(args.source_dir)]
    else:
        source_dirs = [paths.minute_source / f"{year}_1min" for year in years_in_range(args.start, args.end, paths.minute_source)]
    run_id = f"minute-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_root = paths.staging / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    conn = connect(paths.staging / ".duckdb_tmp")
    try:
        conn.execute(f"SET threads = {args.threads}")
        total_installed = total_skipped = total_invalid = 0
        normalized_source_files = 0
        for source_dir in source_dirs:
            if not list(source_dir.glob("*.csv")):
                raise FileNotFoundError(f"No CSV files in {source_dir}")
            prepared_source, normalized_count = prepare_csv_source(source_dir, run_root)
            normalized_source_files += normalized_count
            source_glob = prepared_source / "*.csv"
            invalid_path = paths.quality / "quarantined_records" / f"{source_dir.name}-{run_id}.parquet"
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_query = source_invalid_query(source_glob, args.start, args.end)
            conn.execute(f"COPY ({invalid_query}) TO '{sql_path(invalid_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            invalid_count = conn.execute(f"SELECT count(*) FROM read_parquet('{sql_path(invalid_path)}')").fetchone()[0]
            total_invalid += invalid_count

            staged_output = run_root / source_dir.name
            staged_output.mkdir(parents=True, exist_ok=True)
            valid_query = source_valid_query(source_glob, args.start, args.end)
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM ({valid_query}) AS canonical_rows
                    ORDER BY trade_date, ts_code, datetime
                ) TO '{sql_path(staged_output)}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880,
                 PARTITION_BY (year, month, trade_date),
                 WRITE_PARTITION_COLUMNS false, FILENAME_PATTERN 'part_{{uuid}}')
                """
            )
            installed, skipped = install_minute_partitions(paths, staged_output, args.replace, run_id)
            total_installed += installed
            total_skipped += skipped
        print(json.dumps({"run_id": run_id, "installed_partitions": total_installed, "skipped_partitions": total_skipped, "quarantined_rows": total_invalid, "normalized_newline_files": normalized_source_files}, ensure_ascii=False))
    finally:
        conn.close()
        shutil.rmtree(run_root, ignore_errors=True)


def minute_glob(paths: Paths) -> Path:
    return paths.canonical_minute / "year=*" / "month=*" / "trade_date=*" / "*.parquet"


def create_empty_parquet(conn: duckdb.DuckDBPyConnection, path: Path, columns: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY (SELECT {columns} WHERE false) TO '{sql_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def build_instruments(paths: Paths, conn: duckdb.DuckDBPyConnection) -> None:
    bounds = conn.execute(
        f"""
        SELECT ts_code, min(trade_date) AS first_observed_date, max(trade_date) AS last_observed_date
        FROM read_parquet('{sql_path(minute_glob(paths))}', hive_partitioning = true)
        GROUP BY ts_code
        ORDER BY ts_code
        """
    ).fetch_arrow_table()
    observed = pl.from_arrow(bounds)
    if paths.stock_list.exists():
        listed = pl.read_csv(paths.stock_list, encoding="utf8-lossy", infer_schema_length=10000)
        rename = {"TS代码": "ts_code", "股票代码": "symbol", "股票名称": "name", "交易所代码": "exchange", "上市日期": "list_date", "所属行业": "industry", "市场类型": "market"}
        listed = listed.rename({key: value for key, value in rename.items() if key in listed.columns})
        wanted = [name for name in ("ts_code", "symbol", "name", "exchange", "list_date", "industry", "market") if name in listed.columns]
        listed = listed.select(wanted)
        frame = observed.join(listed, on="ts_code", how="left")
    else:
        frame = observed
    frame = frame.with_columns(pl.lit("minute_source+stock_list").alias("source"))
    write_polars_parquet(frame, paths.reference / "instruments.parquet")


def install_file(source: Path, destination: Path, backup_root: Path, replace: bool, run_id: str) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not replace:
            return False
        backup = backup_root / run_id / destination.relative_to(destination.parents[2])
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(backup))
    shutil.move(str(source), str(destination))
    return True


def cmd_build_daily(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    minute_files = list(paths.canonical_minute.glob("year=*/month=*/trade_date=*/*.parquet"))
    if not minute_files:
        raise FileNotFoundError("No canonical minute Parquet found; run backfill-minute first")
    run_id = f"daily-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_root = paths.staging / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    conn = connect(paths.staging / ".duckdb_tmp")
    try:
        minute_source = sql_path(minute_glob(paths))
        conn.execute(
            f"""
            CREATE TABLE daily_observed AS
            SELECT
                ts_code,
                trade_date,
                arg_min(open, datetime) AS open,
                max(high) AS high,
                min(low) AS low,
                arg_max(close, datetime) AS close,
                sum(volume_lot) AS volume_lot,
                sum(volume_share) AS volume_share,
                sum(amount_cny) AS amount_cny,
                avg(close) AS twap_close,
                count(*)::INTEGER AS bar_count,
                sum(CASE WHEN volume_lot = 0 THEN 1 ELSE 0 END)::INTEGER AS zero_volume_bar_count,
                min(datetime) AS first_bar_time,
                max(datetime) AS last_bar_time
            FROM read_parquet('{minute_source}', hive_partitioning = true)
            GROUP BY ts_code, trade_date
            """
        )
        conn.execute(
            """
            CREATE TABLE daily_returns AS
            SELECT *,
                lag(close) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_observed_close,
                CASE
                    WHEN lag(close) OVER (PARTITION BY ts_code ORDER BY trade_date) > 0
                    THEN close / lag(close) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1
                    ELSE NULL
                END AS raw_close_return
            FROM daily_observed
            """
        )
        conn.execute(
            """
            CREATE TABLE coverage AS
            WITH calendar AS (SELECT DISTINCT trade_date FROM daily_observed),
            bounds AS (
                SELECT ts_code, min(trade_date) AS first_date, max(trade_date) AS last_date
                FROM daily_observed GROUP BY ts_code
            )
            SELECT bounds.ts_code, calendar.trade_date
            FROM bounds
            JOIN calendar ON calendar.trade_date BETWEEN bounds.first_date AND bounds.last_date
            """
        )
        conn.execute(
            """
            CREATE TABLE daily_final AS
            SELECT
                c.ts_code,
                c.trade_date,
                d.open, d.high, d.low, d.close,
                d.volume_lot, d.volume_share, d.amount_cny,
                CASE WHEN d.volume_share > 0 THEN d.amount_cny / d.volume_share ELSE NULL END AS vwap,
                d.twap_close,
                d.prev_observed_close,
                d.raw_close_return,
                d.bar_count,
                d.zero_volume_bar_count,
                d.first_bar_time,
                d.last_bar_time,
                coalesce(d.bar_count = 241, false) AS is_full_session,
                CASE
                    WHEN d.ts_code IS NULL THEN 'missing'
                    WHEN d.bar_count = 241 AND d.volume_lot = 0 THEN 'complete_zero_volume'
                    WHEN d.bar_count = 241 THEN 'complete_trading'
                    ELSE 'partial_observation'
                END AS observation_status
            FROM coverage c
            LEFT JOIN daily_returns d USING (ts_code, trade_date)
            """
        )
        selected_years = canonical_years_in_range(paths, args.start, args.end)
        staged_daily = run_root / "daily"
        staged_coverage = run_root / "coverage"
        staged_daily.mkdir(parents=True, exist_ok=True)
        staged_coverage.mkdir(parents=True, exist_ok=True)
        installed = skipped = 0
        for year in selected_years:
            daily_file = staged_daily / f"year={year}" / "daily.parquet"
            coverage_file = staged_coverage / f"year={year}" / "coverage.parquet"
            daily_file.parent.mkdir(parents=True, exist_ok=True)
            coverage_file.parent.mkdir(parents=True, exist_ok=True)
            conn.execute(
                f"COPY (SELECT * FROM daily_final WHERE year(trade_date) = {year} ORDER BY trade_date, ts_code) TO '{sql_path(daily_file)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
            )
            conn.execute(
                f"COPY (SELECT ts_code, trade_date, observation_status, bar_count, is_full_session FROM daily_final WHERE year(trade_date) = {year} ORDER BY trade_date, ts_code) TO '{sql_path(coverage_file)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            if install_file(daily_file, paths.canonical_daily / f"year={year}" / "daily.parquet", paths.backups / "daily", args.replace, run_id):
                installed += 1
            else:
                skipped += 1
            install_file(coverage_file, paths.quality / "instrument_day_coverage" / f"year={year}" / "coverage.parquet", paths.backups / "coverage", args.replace, run_id)
        calendar_file = run_root / "observed_calendar.parquet"
        conn.execute(
            f"""
            COPY (
                SELECT
                    trade_date,
                    true AS is_observed_market_day,
                    count(*)::INTEGER AS instrument_day_count,
                    sum(CASE WHEN observation_status = 'complete_trading' THEN 1 ELSE 0 END)::INTEGER AS complete_trading_count
                FROM daily_final
                GROUP BY trade_date
                ORDER BY trade_date
            ) TO '{sql_path(calendar_file)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        install_file(calendar_file, paths.reference / "observed_calendar.parquet", paths.backups / "reference", True, run_id)
        build_instruments(paths, conn)
        print(json.dumps({"run_id": run_id, "daily_years_installed": installed, "daily_years_skipped": skipped}, ensure_ascii=False))
    finally:
        conn.close()
        shutil.rmtree(run_root, ignore_errors=True)


def adjustment_glob(paths: Paths) -> Path:
    return paths.canonical_adjustment / "snapshot=*" / "year=*" / "*.parquet"


def cmd_ingest_adjustment(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    source_dir = Path(args.source_dir) if args.source_dir else paths.adjustment_source
    if not source_dir.exists():
        raise FileNotFoundError(f"Adjustment source directory does not exist: {source_dir}")
    snapshot = parse_date(args.snapshot_date)
    assert snapshot is not None
    run_id = f"adjustment-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_root = paths.staging / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    conn = connect(paths.staging / ".duckdb_tmp")
    try:
        source_glob = source_dir / "*.csv"
        query = f"""
        WITH raw AS (
            SELECT
                upper(trim(\"股票代码\")) AS ts_code,
                try_cast(\"交易日期\" AS DATE) AS trade_date,
                try_cast(\"复权因子\" AS DOUBLE) AS vendor_qfq_ratio
            FROM read_csv_auto('{sql_path(source_glob)}', header = true, union_by_name = true, all_varchar = true, strict_mode = true)
        )
        SELECT
            DATE '{snapshot}' AS snapshot_date,
            ts_code,
            trade_date,
            vendor_qfq_ratio,
            CASE
                WHEN ts_code IS NULL OR trade_date IS NULL OR vendor_qfq_ratio IS NULL THEN 'invalid_parse'
                WHEN vendor_qfq_ratio <= 0 THEN 'invalid_nonpositive'
                ELSE 'valid'
            END AS validation_status,
            year(trade_date)::INTEGER AS year
        FROM raw
        WHERE trade_date >= DATE '2018-01-01'
          AND ts_code ~ '^[0-9]{{6}}\\.(SH|SZ)$'
        """
        staged = run_root / "adjustment"
        staged.mkdir(parents=True, exist_ok=True)
        conn.execute(
            f"""
            COPY ({query}) TO '{sql_path(staged)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880,
             PARTITION_BY (year), WRITE_PARTITION_COLUMNS false, FILENAME_PATTERN 'part_{{uuid}}')
            """
        )
        destination_root = paths.canonical_adjustment / f"snapshot={snapshot}"
        if destination_root.exists() and not args.replace:
            raise FileExistsError(f"Snapshot already exists: {destination_root}; pass --replace to archive and rebuild it")
        if destination_root.exists():
            backup = paths.backups / "adjustment" / run_id / destination_root.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination_root), str(backup))
        shutil.move(str(staged), str(destination_root))
        invalid = conn.execute(
            f"SELECT count(*) FROM read_parquet('{sql_path(destination_root / 'year=*' / '*.parquet')}', hive_partitioning = true) WHERE validation_status <> 'valid'"
        ).fetchone()[0]
        print(json.dumps({"snapshot_date": snapshot, "destination": str(destination_root), "invalid_rows": invalid}, ensure_ascii=False))
    finally:
        conn.close()
        shutil.rmtree(run_root, ignore_errors=True)


def daily_glob(paths: Paths) -> Path:
    return paths.canonical_daily / "year=*" / "daily.parquet"


def universe_glob(paths: Paths) -> Path:
    return paths.trading_universe / "year=*" / "universe.parquet"


def index_constituent_glob(paths: Paths) -> Path:
    return paths.index_constituents / "year=*" / "constituents.parquet"


def cmd_ingest_index_events(args: argparse.Namespace) -> None:
    """Materialize month-end CSI300/CSI500 snapshots from public event history."""
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    source_dir = Path(args.source_dir)
    expected = {"csi300.csv": "000300.SH", "csi500.csv": "000905.SH"}
    if not all((source_dir / filename).exists() for filename in expected):
        raise FileNotFoundError("--source-dir must contain history/csi300.csv and history/csi500.csv")
    calendar_file = paths.reference / "observed_calendar.parquet"
    if not calendar_file.exists():
        raise FileNotFoundError("No observed calendar found; run build-daily first")
    event_frames = []
    for filename, index_code in expected.items():
        frame = pl.read_csv(source_dir / filename, encoding="utf8-lossy").with_columns(
            pl.lit(index_code).alias("index_code"),
            pl.col("symbol").str.slice(2, 6).alias("symbol"),
            pl.when(pl.col("symbol").str.starts_with("SH")).then(pl.lit(".SH")).otherwise(pl.lit(".SZ")).alias("exchange"),
            pl.col("opt-in").str.to_date(strict=True).alias("effective_from"),
            pl.col("opt-out").replace("", None).str.to_date(strict=False).alias("effective_to"),
        ).select(["index_code", "symbol", "exchange", "name", "effective_from", "effective_to"])
        event_frames.append(frame)
    events = pl.concat(event_frames).with_columns((pl.col("symbol") + pl.col("exchange")).alias("ts_code"))
    conn = connect(paths.staging / ".duckdb_tmp")
    run_id = f"index-events-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_root = paths.staging / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        conn.register("events", events.to_arrow())
        calendar = sql_path(calendar_file)
        conn.execute(
            f"""
            CREATE TABLE monthly_constituents AS
            WITH month_ends AS (
                SELECT max(trade_date) AS as_of_date
                FROM read_parquet('{calendar}')
                GROUP BY year(trade_date), month(trade_date)
            )
            SELECT e.index_code, m.as_of_date, e.ts_code, e.name,
                e.effective_from, e.effective_to,
                'unliftedq/index-constitution' AS source_name,
                'event_reconstruction' AS source_method
            FROM month_ends m JOIN events e
              ON e.effective_from <= m.as_of_date
             AND (e.effective_to IS NULL OR e.effective_to > m.as_of_date)
            WHERE m.as_of_date >= DATE '2018-01-01'
            QUALIFY row_number() OVER (
                PARTITION BY e.index_code, m.as_of_date, e.ts_code
                ORDER BY e.effective_from DESC, e.effective_to DESC NULLS FIRST
            ) = 1
            """
        )
        installed = 0
        for year in canonical_years_in_range(paths, "2018-01-01", None):
            staged = run_root / f"year={year}" / "constituents.parquet"
            staged.parent.mkdir(parents=True, exist_ok=True)
            conn.execute(f"COPY (SELECT * FROM monthly_constituents WHERE year(as_of_date)={year} ORDER BY as_of_date,index_code,ts_code) TO '{sql_path(staged)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            if install_file(staged, paths.index_constituents / f"year={year}" / "constituents.parquet", paths.backups / "index_constituents", args.replace, run_id):
                installed += 1
        count = conn.execute("SELECT count(*) FROM monthly_constituents").fetchone()[0]
        print(json.dumps({"run_id": run_id, "snapshots": count, "years_installed": installed, "covered_indices": list(expected.values())}, ensure_ascii=False))
    finally:
        conn.close()
        shutil.rmtree(run_root, ignore_errors=True)


def cmd_fetch_index_daily(args: argparse.Namespace) -> None:
    """Download unadjusted official-price index bars from the free BaoStock API."""
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("Install baostock before running fetch-index-daily") from exc
    paths = Paths(Path(args.data_root)); paths.ensure_layout()
    login = bs.login()
    if login.error_code != "0": raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    try:
        frames = []
        for source_code, index_code, name in (("sh.000300", "000300.SH", "CSI300"), ("sh.000905", "000905.SH", "CSI500")):
            rs = bs.query_history_k_data_plus(source_code, "date,open,high,low,close,preclose,volume,amount,pctChg", start_date=args.start, end_date=args.end, frequency="d", adjustflag="3")
            if rs.error_code != "0": raise RuntimeError(f"BaoStock {source_code}: {rs.error_msg}")
            rows=[]
            while rs.next(): rows.append(rs.get_row_data())
            frame=pl.DataFrame(rows, schema=["trade_date","open","high","low","close","prev_close","volume_share","amount_cny","pct_change"], orient="row").with_columns(
                pl.lit(index_code).alias("index_code"), pl.lit(name).alias("index_name"),
                pl.col("trade_date").str.to_date(), *[pl.col(x).cast(pl.Float64, strict=False) for x in ("open","high","low","close","prev_close","volume_share","amount_cny","pct_change")],
            ).with_columns((pl.col("close") / pl.col("prev_close") - 1).alias("close_return"))
            frames.append(frame)
        output=pl.concat(frames).sort(["index_code","trade_date"])
        write_polars_parquet(output, paths.index_daily / "index_daily.parquet")
        print(json.dumps({"rows": output.height, "first_date": str(output["trade_date"].min()), "last_date": str(output["trade_date"].max()), "source": "BaoStock"}, ensure_ascii=False))
    finally:
        bs.logout()


def cmd_build_universe(args: argparse.Namespace) -> None:
    """Build one eligible-stock membership set for every observed trade date."""
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    if not list(paths.canonical_daily.glob("year=*/daily.parquet")):
        raise FileNotFoundError("No daily aggregated Parquet found")
    if not list(paths.canonical_adjustment.glob("snapshot=*/year=*/*.parquet")):
        raise FileNotFoundError("No adjustment Parquet found")
    run_id = f"universe-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_root = paths.staging / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    conn = connect(paths.staging / ".duckdb_tmp")
    try:
        daily = sql_path(daily_glob(paths))
        adjustment = sql_path(adjustment_glob(paths))
        conn.execute(
            f"""
            CREATE TABLE eligible_universe AS
            WITH latest_adjustment AS (
                SELECT snapshot_date, ts_code, trade_date, vendor_qfq_ratio, validation_status
                FROM read_parquet('{adjustment}', hive_partitioning = true)
                QUALIFY row_number() OVER (
                    PARTITION BY ts_code, trade_date ORDER BY snapshot_date DESC
                ) = 1
            )
            SELECT
                'qfq_valid_complete_trading_v1' AS universe_name,
                d.trade_date,
                d.ts_code,
                a.snapshot_date AS adjustment_snapshot_date,
                a.vendor_qfq_ratio,
                d.bar_count,
                d.observation_status,
                true AS is_eligible,
                'valid_qfq_and_complete_trading' AS eligibility_reason
            FROM read_parquet('{daily}', hive_partitioning = true) d
            JOIN latest_adjustment a USING (ts_code, trade_date)
            WHERE d.observation_status = 'complete_trading'
              AND a.validation_status = 'valid'
              AND isfinite(a.vendor_qfq_ratio)
              AND a.vendor_qfq_ratio > 0
            """
        )
        installed = skipped = 0
        for year in canonical_years_in_range(paths, args.start, args.end):
            staged = run_root / f"year={year}" / "universe.parquet"
            staged.parent.mkdir(parents=True, exist_ok=True)
            conn.execute(
                f"COPY (SELECT * FROM eligible_universe WHERE year(trade_date) = {year} "
                f"ORDER BY trade_date, ts_code) TO '{sql_path(staged)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
            )
            destination = paths.trading_universe / f"year={year}" / "universe.parquet"
            if install_file(staged, destination, paths.backups / "trading_universe", args.replace, run_id):
                installed += 1
            else:
                skipped += 1
        members = conn.execute("SELECT count(*) FROM eligible_universe").fetchone()[0]
        print(json.dumps({"run_id": run_id, "years_installed": installed, "years_skipped": skipped, "eligible_memberships": members}, ensure_ascii=False))
    finally:
        conn.close()
        shutil.rmtree(run_root, ignore_errors=True)


def cmd_build_catalog(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    if not list(paths.canonical_minute.glob("year=*/month=*/trade_date=*/*.parquet")):
        raise FileNotFoundError("No canonical minute Parquet found")
    if not list(paths.canonical_daily.glob("year=*/daily.parquet")):
        raise FileNotFoundError("No daily aggregated Parquet found")
    if not list(paths.canonical_adjustment.glob("snapshot=*/year=*/*.parquet")):
        raise FileNotFoundError("No adjustment Parquet found")
    if not (paths.reference / "observed_calendar.parquet").exists():
        raise FileNotFoundError("No observed calendar found")
    conn = duckdb.connect(str(paths.catalog))
    try:
        minute = sql_path(minute_glob(paths))
        daily = sql_path(daily_glob(paths))
        adjustment = sql_path(adjustment_glob(paths))
        calendar = sql_path(paths.reference / "observed_calendar.parquet")
        coverage = sql_path(paths.quality / "instrument_day_coverage" / "year=*" / "coverage.parquet")
        instruments = sql_path(paths.reference / "instruments.parquet")
        universe = sql_path(universe_glob(paths))
        index_constituents = sql_path(index_constituent_glob(paths))
        index_daily = sql_path(paths.index_daily / "index_daily.parquet")
        validation = sql_path(paths.quality / "validation_results.parquet")
        conn.execute(f"CREATE OR REPLACE VIEW minute_bars AS SELECT * FROM read_parquet('{minute}', hive_partitioning = true)")
        conn.execute(f"CREATE OR REPLACE VIEW daily_aggregated AS SELECT * FROM read_parquet('{daily}', hive_partitioning = true)")
        if (paths.index_daily / "index_daily.parquet").exists():
            conn.execute(f"CREATE OR REPLACE VIEW index_daily AS SELECT * FROM read_parquet('{index_daily}')")
        else:
            conn.execute("CREATE OR REPLACE VIEW index_daily AS SELECT CAST(NULL AS VARCHAR) AS index_code, CAST(NULL AS DATE) AS trade_date, CAST(NULL AS DOUBLE) AS close_return WHERE false")
        conn.execute(f"CREATE OR REPLACE VIEW adjustment_snapshots AS SELECT * FROM read_parquet('{adjustment}', hive_partitioning = true)")
        conn.execute(
            """
            CREATE OR REPLACE VIEW latest_adjustment AS
            SELECT snapshot_date, ts_code, trade_date, vendor_qfq_ratio, validation_status
            FROM adjustment_snapshots
            QUALIFY row_number() OVER (PARTITION BY ts_code, trade_date ORDER BY snapshot_date DESC) = 1
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE VIEW daily_qfq AS
            SELECT
                d.*,
                a.snapshot_date AS adjustment_snapshot_date,
                a.vendor_qfq_ratio,
                CASE WHEN a.validation_status = 'valid' THEN d.open * a.vendor_qfq_ratio END AS qfq_open,
                CASE WHEN a.validation_status = 'valid' THEN d.high * a.vendor_qfq_ratio END AS qfq_high,
                CASE WHEN a.validation_status = 'valid' THEN d.low * a.vendor_qfq_ratio END AS qfq_low,
                CASE WHEN a.validation_status = 'valid' THEN d.close * a.vendor_qfq_ratio END AS qfq_close,
                CASE WHEN a.validation_status = 'valid' THEN d.vwap * a.vendor_qfq_ratio END AS qfq_vwap,
                CASE WHEN a.validation_status = 'valid' THEN d.twap_close * a.vendor_qfq_ratio END AS qfq_twap,
                CASE
                    WHEN d.close IS NOT NULL
                     AND lag(d.close * a.vendor_qfq_ratio) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) > 0
                    THEN d.close * a.vendor_qfq_ratio
                       / lag(d.close * a.vendor_qfq_ratio) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) - 1
                END AS qfq_return
            FROM daily_aggregated d
            LEFT JOIN latest_adjustment a USING (ts_code, trade_date)
            """
        )
        conn.execute("""
            CREATE OR REPLACE VIEW daily_excess_returns AS
            SELECT d.ts_code, d.trade_date, d.raw_close_return, q.qfq_return,
                b.csi300_return, b.csi500_return,
                q.qfq_return - b.csi300_return AS excess_return_vs_csi300,
                q.qfq_return - b.csi500_return AS excess_return_vs_csi500
            FROM daily_aggregated d
            LEFT JOIN daily_qfq q USING (ts_code, trade_date)
            LEFT JOIN (
                SELECT trade_date,
                  max(CASE WHEN index_code='000300.SH' THEN close_return END) AS csi300_return,
                  max(CASE WHEN index_code='000905.SH' THEN close_return END) AS csi500_return
                FROM index_daily GROUP BY trade_date
            ) b USING (trade_date)
        """)
        conn.execute(f"CREATE OR REPLACE VIEW observed_calendar AS SELECT * FROM read_parquet('{calendar}')")
        conn.execute(f"CREATE OR REPLACE VIEW instrument_day_coverage AS SELECT * FROM read_parquet('{coverage}', hive_partitioning = true)")
        conn.execute(f"CREATE OR REPLACE VIEW instruments AS SELECT * FROM read_parquet('{instruments}')")
        if list(paths.trading_universe.glob("year=*/universe.parquet")):
            conn.execute(f"CREATE OR REPLACE VIEW trading_universe AS SELECT * FROM read_parquet('{universe}', hive_partitioning = true)")
        else:
            conn.execute("CREATE OR REPLACE VIEW trading_universe AS SELECT CAST(NULL AS VARCHAR) AS universe_name, CAST(NULL AS DATE) AS trade_date, CAST(NULL AS VARCHAR) AS ts_code, CAST(NULL AS BOOLEAN) AS is_eligible WHERE false")
        if list(paths.index_constituents.glob("year=*/constituents.parquet")):
            conn.execute(f"CREATE OR REPLACE VIEW index_monthly_constituents AS SELECT * FROM read_parquet('{index_constituents}', hive_partitioning = true)")
            conn.execute("""
                CREATE OR REPLACE VIEW index_trading_universe AS
                SELECT c.index_code, u.trade_date, u.ts_code, c.as_of_date AS constituent_snapshot_date
                FROM trading_universe u
                JOIN index_monthly_constituents c
                 ON c.ts_code = u.ts_code
                 AND c.as_of_date = (
                    SELECT max(c2.as_of_date) FROM index_monthly_constituents c2
                    WHERE c2.index_code = c.index_code
                      AND c2.as_of_date <= u.trade_date
                 )
            """)
        else:
            conn.execute("CREATE OR REPLACE VIEW index_monthly_constituents AS SELECT CAST(NULL AS VARCHAR) AS index_code, CAST(NULL AS DATE) AS as_of_date, CAST(NULL AS VARCHAR) AS ts_code WHERE false")
            conn.execute("CREATE OR REPLACE VIEW index_trading_universe AS SELECT CAST(NULL AS VARCHAR) AS index_code, CAST(NULL AS DATE) AS trade_date, CAST(NULL AS VARCHAR) AS ts_code WHERE false")
        if Path(validation.replace("''", "'")).exists():
            conn.execute(f"CREATE OR REPLACE VIEW data_quality_issues AS SELECT * FROM read_parquet('{validation}')")
        else:
            conn.execute("CREATE OR REPLACE VIEW data_quality_issues AS SELECT CAST(NULL AS VARCHAR) AS check_name, CAST(NULL AS VARCHAR) AS severity, CAST(NULL AS BIGINT) AS violations, CAST(NULL AS VARCHAR) AS details, CAST(NULL AS TIMESTAMP) AS checked_at WHERE false")
        conn.execute(
            """
            CREATE OR REPLACE VIEW data_status AS
            SELECT 'minute_bars' AS dataset, min(trade_date) AS first_date, max(trade_date) AS last_date, count(*) AS rows FROM minute_bars
            UNION ALL
            SELECT 'daily_aggregated', min(trade_date), max(trade_date), count(*) FROM daily_aggregated
            UNION ALL
            SELECT 'latest_adjustment', min(trade_date), max(trade_date), count(*) FROM latest_adjustment
            UNION ALL
            SELECT 'trading_universe', min(trade_date), max(trade_date), count(*) FROM trading_universe
            """
        )
        print(json.dumps({"catalog": str(paths.catalog), "views": 14}, ensure_ascii=False))
    finally:
        conn.close()


def cmd_validate(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    paths.ensure_layout()
    if not list(paths.canonical_minute.glob("year=*/month=*/trade_date=*/*.parquet")):
        raise FileNotFoundError("No canonical minute Parquet found")
    conn = connect(paths.staging / ".duckdb_tmp")
    try:
        minute = sql_path(minute_glob(paths))
        # A global COUNT(DISTINCT ...) over the full minute lake is needlessly
        # memory hungry.  A datetime belongs to exactly one date partition, so
        # validating the primary key file-by-file is equivalent and bounded.
        def duplicate_rows(files: Iterable[Path], columns: str) -> int:
            total = 0
            for parquet_file in files:
                source = sql_path(parquet_file)
                total += conn.execute(
                    f"SELECT coalesce(sum(row_count - 1), 0) FROM ("
                    f"SELECT count(*) AS row_count FROM read_parquet('{source}') "
                    f"GROUP BY {columns} HAVING count(*) > 1)"
                ).fetchone()[0]
            return int(total)

        minute_pk_violations = duplicate_rows(
            paths.canonical_minute.glob("year=*/month=*/trade_date=*/*.parquet"),
            "ts_code, datetime",
        )
        checks: list[tuple[str, str, str, str]] = [
            ("minute_primary_key", "error", str(minute_pk_violations), "duplicate (ts_code, datetime)"),
            ("minute_ohlc", "error", f"SELECT count(*) FROM read_parquet('{minute}', hive_partitioning = true) WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR high < greatest(open, close) OR low > least(open, close)", "invalid minute OHLC"),
            ("minute_session", "error", f"SELECT count(*) FROM read_parquet('{minute}', hive_partitioning = true) WHERE minute_index NOT BETWEEN 0 AND 240", "minute index outside 0..240"),
            ("minute_volume_unit", "error", f"SELECT count(*) FROM read_parquet('{minute}', hive_partitioning = true) WHERE volume_share <> volume_lot * 100 OR volume_lot < 0 OR amount_cny < 0", "volume/amount unit violation"),
        ]
        if list(paths.canonical_daily.glob("year=*/daily.parquet")):
            daily = sql_path(daily_glob(paths))
            daily_pk_violations = duplicate_rows(
                paths.canonical_daily.glob("year=*/daily.parquet"),
                "ts_code, trade_date",
            )
            checks.extend([
                ("daily_primary_key", "error", str(daily_pk_violations), "duplicate (ts_code, trade_date)"),
                ("daily_ohlc", "error", f"SELECT count(*) FROM read_parquet('{daily}', hive_partitioning = true) WHERE observation_status <> 'missing' AND (open <= 0 OR high < greatest(open, close) OR low > least(open, close))", "invalid aggregate daily OHLC"),
                (
                    "daily_minute_reaggregation",
                    "error",
                    f"""
                    WITH reaggregated AS (
                        SELECT ts_code, trade_date,
                            arg_min(open, datetime) AS open, max(high) AS high,
                            min(low) AS low, arg_max(close, datetime) AS close,
                            sum(volume_lot) AS volume_lot,
                            sum(volume_share) AS volume_share,
                            sum(amount_cny) AS amount_cny,
                            count(*)::INTEGER AS bar_count
                        FROM read_parquet('{minute}', hive_partitioning = true)
                        GROUP BY ts_code, trade_date
                    )
                    SELECT count(*)
                    FROM read_parquet('{daily}', hive_partitioning = true) d
                    FULL OUTER JOIN reaggregated r USING (ts_code, trade_date)
                    WHERE d.observation_status <> 'missing' AND (
                        d.open IS DISTINCT FROM r.open OR d.high IS DISTINCT FROM r.high
                        OR d.low IS DISTINCT FROM r.low OR d.close IS DISTINCT FROM r.close
                        OR d.volume_lot IS DISTINCT FROM r.volume_lot
                        OR d.volume_share IS DISTINCT FROM r.volume_share
                        OR d.amount_cny IS DISTINCT FROM r.amount_cny
                        OR d.bar_count IS DISTINCT FROM r.bar_count
                    )
                    """,
                    "daily OHLCV differs from canonical-minute reaggregation",
                ),
            ])
        if list(paths.canonical_adjustment.glob("snapshot=*/year=*/*.parquet")):
            adjustment = sql_path(adjustment_glob(paths))
            checks.append(("adjustment_validity", "error", f"SELECT count(*) FROM read_parquet('{adjustment}', hive_partitioning = true) WHERE validation_status <> 'valid'", "invalid vendor qfq ratio"))
            if list(paths.canonical_daily.glob("year=*/daily.parquet")):
                checks.append(("adjustment_coverage", "warning", f"SELECT count(*) FROM read_parquet('{daily}', hive_partitioning = true) d LEFT JOIN (SELECT * FROM read_parquet('{adjustment}', hive_partitioning = true) QUALIFY row_number() OVER (PARTITION BY ts_code, trade_date ORDER BY snapshot_date DESC) = 1) a USING(ts_code, trade_date) WHERE d.observation_status <> 'missing' AND (a.validation_status IS NULL OR a.validation_status <> 'valid')", "observed daily bar without valid adjustment ratio"))
        rows: list[dict[str, object]] = []
        for name, severity, query, details in checks:
            violations = int(query) if name in {"minute_primary_key", "daily_primary_key"} else conn.execute(query).fetchone()[0]
            rows.append({"check_name": name, "severity": severity, "violations": violations, "details": details, "checked_at": utc_now()})
        frame = pl.DataFrame(rows)
        write_polars_parquet(frame, paths.quality / "validation_results.parquet")
        print(frame)
        if any(row["severity"] == "error" and row["violations"] for row in rows):
            raise SystemExit("Validation failed: see lake/quality/validation_results.parquet")
    finally:
        conn.close()


def cmd_update_day(args: argparse.Namespace) -> None:
    args.start = args.trade_date
    args.end = args.trade_date
    if not args.source_dir:
        raise FileNotFoundError("update-day requires --source-dir pointing to the received CSV package")
    args.replace = True
    args.threads = getattr(args, "threads", 4)
    cmd_backfill_minute(args)
    cmd_build_daily(args)
    cmd_build_universe(args)


def cmd_refresh_month(args: argparse.Namespace) -> None:
    try:
        month_start = datetime.strptime(args.month, "%Y-%m").date().replace(day=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Month must be YYYY-MM") from exc
    if month_start.month == 12:
        month_end = month_start.replace(year=month_start.year + 1, month=1)
    else:
        month_end = month_start.replace(month=month_start.month + 1)
    from datetime import timedelta

    args.start = month_start.isoformat()
    args.end = (month_end - timedelta(days=1)).isoformat()
    if not args.source_dir:
        raise FileNotFoundError("refresh-month requires --source-dir pointing to the received CSV package")
    args.replace = True
    args.threads = getattr(args, "threads", 4)
    cmd_backfill_minute(args)
    cmd_build_daily(args)
    cmd_build_universe(args)


def cmd_status(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.data_root))
    if not paths.catalog.exists():
        raise FileNotFoundError("Catalog does not exist; run build-catalog first")
    conn = duckdb.connect(str(paths.catalog), read_only=True)
    try:
        print(conn.execute("SELECT * FROM data_status ORDER BY dataset").fetchdf().to_string(index=False))
        print("\nLatest validation:")
        print(conn.execute("SELECT check_name, severity, violations, checked_at FROM data_quality_issues ORDER BY check_name").fetchdf().to_string(index=False))
    finally:
        conn.close()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=DEFAULT_DATABASE_DIR, help="Existing A_stock_database directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and maintain the local A-share minute-data lake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Record source-file metadata")
    add_common_arguments(inventory)
    inventory.add_argument("--checksum", action="store_true", help="Calculate SHA-256 for every source CSV (slow)")
    inventory.add_argument("--source-dir", help="Override a yearly source directory, useful for tests")
    inventory.set_defaults(handler=cmd_inventory)

    backfill = subparsers.add_parser("backfill-minute", help="Convert minute CSV to date-partitioned Parquet")
    add_common_arguments(backfill)
    backfill.add_argument("--start", type=parse_date)
    backfill.add_argument("--end", type=parse_date)
    backfill.add_argument("--replace", action="store_true", help="Archive and replace already-built date partitions")
    backfill.add_argument("--source-dir", help="Override a yearly source directory, useful for tests")
    backfill.add_argument("--threads", type=int, default=4, help="DuckDB conversion threads; ordered output avoids tiny Parquet files")
    backfill.set_defaults(handler=cmd_backfill_minute)

    daily = subparsers.add_parser("build-daily", help="Aggregate canonical minute Parquet into daily OHLCV")
    add_common_arguments(daily)
    daily.add_argument("--start", type=parse_date)
    daily.add_argument("--end", type=parse_date)
    daily.add_argument("--replace", action="store_true", help="Archive and replace existing annual daily files")
    daily.set_defaults(handler=cmd_build_daily)

    universe = subparsers.add_parser("build-universe", help="Build daily eligible-stock memberships from complete trading and valid qfq factors")
    add_common_arguments(universe)
    universe.add_argument("--start", type=parse_date)
    universe.add_argument("--end", type=parse_date)
    universe.add_argument("--replace", action="store_true", help="Archive and replace existing annual universe files")
    universe.set_defaults(handler=cmd_build_universe)

    index_events = subparsers.add_parser("ingest-index-events", help="Build month-end CSI300/CSI500 constituent snapshots from event-history CSVs")
    add_common_arguments(index_events)
    index_events.add_argument("--source-dir", required=True, help="Directory containing csi300.csv and csi500.csv")
    index_events.add_argument("--replace", action="store_true", help="Archive and replace existing annual constituent snapshots")
    index_events.set_defaults(handler=cmd_ingest_index_events)

    index_daily = subparsers.add_parser("fetch-index-daily", help="Download CSI300 and CSI500 daily bars from BaoStock")
    add_common_arguments(index_daily)
    index_daily.add_argument("--start", default="2018-01-01")
    index_daily.add_argument("--end", default=datetime.now().date().isoformat())
    index_daily.set_defaults(handler=cmd_fetch_index_daily)

    adjustment = subparsers.add_parser("ingest-adjustment", help="Import a supplier front-adjustment snapshot")
    add_common_arguments(adjustment)
    adjustment.add_argument("--snapshot-date", required=True, type=parse_date)
    adjustment.add_argument("--source-dir", help="Override adjustment CSV directory")
    adjustment.add_argument("--replace", action="store_true", help="Archive and replace an existing snapshot")
    adjustment.set_defaults(handler=cmd_ingest_adjustment)

    catalog = subparsers.add_parser("build-catalog", help="Create DuckDB SQL views over Parquet")
    add_common_arguments(catalog)
    catalog.set_defaults(handler=cmd_build_catalog)

    validate = subparsers.add_parser("validate", help="Run data-quality checks")
    add_common_arguments(validate)
    validate.set_defaults(handler=cmd_validate)

    update = subparsers.add_parser("update-day", help="Rebuild one trade date from the raw annual CSV")
    add_common_arguments(update)
    update.add_argument("--trade-date", required=True, type=parse_date)
    update.add_argument("--source-dir", help="Directory containing the newly received minute CSV files")
    update.set_defaults(handler=cmd_update_day)

    refresh = subparsers.add_parser("refresh-month", help="Rebuild one month from the raw annual CSV")
    add_common_arguments(refresh)
    refresh.add_argument("--month", required=True, help="YYYY-MM")
    refresh.add_argument("--source-dir", help="Directory containing the newly received minute CSV files")
    refresh.set_defaults(handler=cmd_refresh_month)

    status = subparsers.add_parser("status", help="Show catalog coverage and latest validation")
    add_common_arguments(status)
    status.set_defaults(handler=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, FileExistsError, duckdb.Error, pl.exceptions.PolarsError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
