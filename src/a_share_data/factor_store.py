"""Official daily-factor store, intentionally separate from research Parquet.

Research artifacts remain immutable Parquet files under ``derived/factors``.
Only a manually registered and activated factor may be written to this DuckDB
store.  The module does not calculate factors: callers provide one validated
daily cross-section at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


DEFAULT_STORE_PATH = Path("A_stock_database/lake/production/factor_store.duckdb")
VALUE_COLUMNS = ("trade_date", "ts_code", "factor_value")
_FACTOR_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class FactorStoreError(RuntimeError):
    """Raised when an official-factor lifecycle or data contract is violated."""


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _value_table_name(factor_id: str) -> str:
    if not _FACTOR_ID.fullmatch(factor_id):
        raise FactorStoreError("factor_id must start with a letter and contain only letters, digits, and underscores")
    # Hashing makes table names stable while preventing a long research ID from
    # leaking implementation details into the physical schema.
    return "official_factor_" + hashlib.sha256(factor_id.encode("utf-8")).hexdigest()[:16]


def _as_frame(values: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(values, pl.DataFrame):
        raise FactorStoreError("factor values must be supplied as a Polars DataFrame")
    if set(values.columns) != set(VALUE_COLUMNS):
        raise FactorStoreError(f"factor values must contain exactly {list(VALUE_COLUMNS)}")
    try:
        return values.select(
            pl.col("trade_date").cast(pl.Date, strict=True),
            pl.col("ts_code").cast(pl.String, strict=True),
            pl.col("factor_value").cast(pl.Float64, strict=True),
        )
    except pl.exceptions.PolarsError as exc:
        raise FactorStoreError("factor values must have date, string, and float-compatible columns") from exc


def _validate_values(values: pl.DataFrame, *, target_date: date | None = None) -> pl.DataFrame:
    frame = _as_frame(values)
    if frame.is_empty():
        raise FactorStoreError("factor values must not be empty")
    invalid = frame.select(
        pl.col("trade_date").is_null().any().alias("null_date"),
        pl.col("ts_code").is_null().or_(pl.col("ts_code").str.len_chars() == 0).any().alias("invalid_code"),
        (~pl.col("factor_value").is_finite()).any().alias("nonfinite_value"),
        pl.struct(VALUE_COLUMNS[:2]).n_unique().ne(pl.len()).alias("duplicate_key"),
    ).row(0, named=True)
    if any(invalid.values()):
        labels = ", ".join(key for key, value in invalid.items() if value)
        raise FactorStoreError(f"invalid factor values: {labels}")
    if target_date is not None:
        dates = frame.get_column("trade_date").unique().to_list()
        if dates != [target_date]:
            raise FactorStoreError(f"write-day source must contain only {target_date.isoformat()}")
    return frame.sort(["trade_date", "ts_code"])


class OfficialFactorStore:
    """Transactional store for approved, daily-frequency production factors."""

    def __init__(self, path: Path | str = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.path))

    @staticmethod
    def _initialize(connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS official_factor_registry (
                factor_id VARCHAR PRIMARY KEY,
                value_table VARCHAR NOT NULL UNIQUE,
                status VARCHAR NOT NULL CHECK (status IN ('registered', 'active', 'retired')),
                output_frequency VARCHAR NOT NULL CHECK (output_frequency = 'daily'),
                research_manifest_path VARCHAR NOT NULL,
                research_manifest_sha256 VARCHAR NOT NULL,
                formula VARCHAR,
                source_file VARCHAR,
                source_file_sha256 VARCHAR,
                approved_by VARCHAR NOT NULL,
                approval_note VARCHAR NOT NULL,
                registered_at TIMESTAMP NOT NULL,
                activated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS official_factor_runs (
                run_id VARCHAR PRIMARY KEY,
                factor_id VARCHAR NOT NULL,
                operation VARCHAR NOT NULL,
                input_start_date DATE,
                input_end_date DATE,
                source_path VARCHAR,
                source_sha256 VARCHAR,
                row_count BIGINT,
                status VARCHAR NOT NULL CHECK (status IN ('succeeded', 'failed')),
                error_message VARCHAR,
                run_metadata JSON,
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS official_factor_watermarks (
                factor_id VARCHAR PRIMARY KEY,
                last_complete_trade_date DATE NOT NULL,
                last_run_id VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            """
        )

    def initialize(self) -> None:
        connection = self._connect()
        try:
            self._initialize(connection)
        finally:
            connection.close()

    @staticmethod
    def _manifest(path: Path, factor_id: str) -> dict[str, Any]:
        if not path.is_file():
            raise FactorStoreError(f"research manifest does not exist: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FactorStoreError(f"invalid research manifest: {path}") from exc
        if manifest.get("factor_id") != factor_id:
            raise FactorStoreError("research manifest factor_id does not match the requested official factor")
        return manifest

    @staticmethod
    def _record_run(
        connection: duckdb.DuckDBPyConnection,
        *,
        run_id: str,
        factor_id: str,
        operation: str,
        start: date | None,
        end: date | None,
        source_path: Path | None,
        source_sha256: str | None,
        rows: int | None,
        status: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO official_factor_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, factor_id, operation, start, end,
                str(source_path) if source_path else None, source_sha256, rows, status,
                error, json.dumps(metadata or {}, ensure_ascii=False), now, now,
            ],
        )

    def _record_failed_operation(
        self,
        *,
        factor_id: str,
        operation: str,
        error: Exception,
        start: date | None = None,
        end: date | None = None,
        source_path: Path | None = None,
    ) -> None:
        """Best-effort audit trail for failures that happen before a transaction."""
        connection = self._connect()
        try:
            self._initialize(connection)
            self._record_run(
                connection, run_id=uuid.uuid4().hex, factor_id=factor_id, operation=operation,
                start=start, end=end, source_path=source_path,
                source_sha256=_sha256(source_path) if source_path and source_path.is_file() else None,
                rows=None, status="failed", error=str(error),
            )
        finally:
            connection.close()

    @staticmethod
    def _registry_row(connection: duckdb.DuckDBPyConnection, factor_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT factor_id, value_table, status, research_manifest_sha256 FROM official_factor_registry WHERE factor_id = ?",
            [factor_id],
        ).fetchone()
        if row is None:
            raise FactorStoreError(f"factor is not registered: {factor_id}")
        return {"factor_id": row[0], "value_table": row[1], "status": row[2], "research_manifest_sha256": row[3]}

    def register_factor(
        self,
        factor_id: str,
        research_manifest: Path | str,
        approved_by: str,
        approval_note: str,
    ) -> dict[str, Any]:
        manifest_path = Path(research_manifest)
        manifest = self._manifest(manifest_path, factor_id)
        if not approved_by.strip() or not approval_note.strip():
            raise FactorStoreError("approved_by and approval_note must not be empty")
        table = _value_table_name(factor_id)
        source_file = Path(manifest["source_file"]) if manifest.get("source_file") else None
        source_file_sha256 = _sha256(source_file) if source_file and source_file.is_file() else None
        run_id = uuid.uuid4().hex
        connection = self._connect()
        try:
            self._initialize(connection)
            connection.execute("BEGIN")
            if connection.execute("SELECT 1 FROM official_factor_registry WHERE factor_id = ?", [factor_id]).fetchone():
                raise FactorStoreError(f"factor is already registered: {factor_id}")
            connection.execute(
                f"""
                CREATE TABLE {_quote_identifier(table)} (
                    trade_date DATE NOT NULL,
                    ts_code VARCHAR NOT NULL,
                    factor_value DOUBLE NOT NULL,
                    run_id VARCHAR NOT NULL,
                    loaded_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (trade_date, ts_code)
                )
                """
            )
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO official_factor_registry VALUES (?, ?, 'registered', 'daily', ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                [factor_id, table, str(manifest_path), _sha256(manifest_path), manifest.get("formula"), manifest.get("source_file"), source_file_sha256, approved_by, approval_note, now],
            )
            self._record_run(connection, run_id=run_id, factor_id=factor_id, operation="register", start=None, end=None,
                             source_path=manifest_path, source_sha256=_sha256(manifest_path), rows=0, status="succeeded")
            connection.execute("COMMIT")
            return {"factor_id": factor_id, "value_table": table, "status": "registered", "run_id": run_id}
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def activate_factor(self, factor_id: str) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        connection = self._connect()
        try:
            self._initialize(connection)
            connection.execute("BEGIN")
            registry = self._registry_row(connection, factor_id)
            if registry["status"] == "retired":
                raise FactorStoreError(f"retired factor cannot be activated: {factor_id}")
            connection.execute("UPDATE official_factor_registry SET status = 'active', activated_at = ? WHERE factor_id = ?", [_utc_now(), factor_id])
            self._record_run(connection, run_id=run_id, factor_id=factor_id, operation="activate", start=None, end=None,
                             source_path=None, source_sha256=None, rows=0, status="succeeded")
            connection.execute("COMMIT")
            return {"factor_id": factor_id, "status": "active", "run_id": run_id}
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _write_values(
        self,
        *,
        factor_id: str,
        values: pl.DataFrame,
        operation: str,
        source_path: Path | None,
        source_sha256: str | None,
        metadata: dict[str, Any] | None,
        replace_dates: bool,
    ) -> dict[str, Any]:
        frame = _validate_values(values)
        start, end = frame.get_column("trade_date").min(), frame.get_column("trade_date").max()
        run_id = uuid.uuid4().hex
        connection = self._connect()
        try:
            self._initialize(connection)
            connection.execute("BEGIN")
            registry = self._registry_row(connection, factor_id)
            if operation == "write_day" and registry["status"] != "active":
                raise FactorStoreError(f"only active factors may receive daily writes: {factor_id}")
            table = _quote_identifier(registry["value_table"])
            if not replace_dates:
                overlap = connection.execute(
                    f"SELECT count(*) FROM {table} WHERE trade_date BETWEEN ? AND ?", [start, end]
                ).fetchone()[0]
                if overlap:
                    raise FactorStoreError("history import overlaps existing official values")
            else:
                connection.execute(f"DELETE FROM {table} WHERE trade_date BETWEEN ? AND ?", [start, end])
            connection.register("official_factor_staging", frame.to_arrow())
            connection.execute(
                f"INSERT INTO {table} (trade_date, ts_code, factor_value, run_id, loaded_at) "
                "SELECT trade_date, ts_code, factor_value, ?, ? FROM official_factor_staging",
                [run_id, _utc_now()],
            )
            self._record_run(connection, run_id=run_id, factor_id=factor_id, operation=operation, start=start, end=end,
                             source_path=source_path, source_sha256=source_sha256, rows=frame.height, status="succeeded", metadata=metadata)
            watermark = connection.execute("SELECT last_complete_trade_date FROM official_factor_watermarks WHERE factor_id = ?", [factor_id]).fetchone()
            if watermark is None:
                connection.execute("INSERT INTO official_factor_watermarks VALUES (?, ?, ?, ?)", [factor_id, end, run_id, _utc_now()])
            elif end > watermark[0]:
                connection.execute("UPDATE official_factor_watermarks SET last_complete_trade_date = ?, last_run_id = ?, updated_at = ? WHERE factor_id = ?", [end, run_id, _utc_now(), factor_id])
            connection.unregister("official_factor_staging")
            connection.execute("COMMIT")
            return {"factor_id": factor_id, "operation": operation, "run_id": run_id, "rows": frame.height, "start": str(start), "end": str(end)}
        except Exception as exc:
            try:
                connection.execute("ROLLBACK")
                self._record_run(connection, run_id=run_id, factor_id=factor_id, operation=operation, start=start, end=end,
                                 source_path=source_path, source_sha256=source_sha256, rows=None, status="failed", error=str(exc), metadata=metadata)
            except duckdb.Error:
                pass
            raise
        finally:
            connection.close()

    def import_history(
        self,
        factor_id: str,
        source: Path | str,
        start: date | str,
        end: date | str,
    ) -> dict[str, Any]:
        source_path = Path(source)
        if not source_path.is_file():
            error = FactorStoreError(f"research factor Parquet does not exist: {source_path}")
            self._record_failed_operation(factor_id=factor_id, operation="import_history", error=error, source_path=source_path)
            raise error
        start_date = _parse_date(start) if isinstance(start, str) else start
        end_date = _parse_date(end) if isinstance(end, str) else end
        if start_date > end_date:
            raise FactorStoreError("history start must not be after end")
        manifest_path = source_path.with_name("manifest.json")
        self._manifest(manifest_path, factor_id)
        connection = self._connect()
        try:
            self._initialize(connection)
            registry = self._registry_row(connection, factor_id)
            if registry["status"] == "retired":
                raise FactorStoreError(f"retired factor cannot import history: {factor_id}")
            if registry["research_manifest_sha256"] != _sha256(manifest_path):
                raise FactorStoreError("research manifest changed after official registration; register a new factor version")
        finally:
            connection.close()
        try:
            values = pl.read_parquet(source_path).filter(
                (pl.col("trade_date") >= pl.lit(start_date)) & (pl.col("trade_date") <= pl.lit(end_date))
            )
        except Exception as exc:
            self._record_failed_operation(factor_id=factor_id, operation="import_history", error=exc, start=start_date, end=end_date, source_path=source_path)
            raise
        return self._write_values(factor_id=factor_id, values=values, operation="import_history", source_path=source_path,
                                  source_sha256=_sha256(source_path), metadata={"requested_start": str(start_date), "requested_end": str(end_date)}, replace_dates=False)

    def write_day(
        self,
        factor_id: str,
        trade_date: date | str,
        values: pl.DataFrame,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = _parse_date(trade_date) if isinstance(trade_date, str) else trade_date
        try:
            frame = _validate_values(values, target_date=target)
        except FactorStoreError as exc:
            self._record_failed_operation(factor_id=factor_id, operation="write_day", error=exc, start=target, end=target)
            raise
        return self._write_values(factor_id=factor_id, values=frame, operation="write_day", source_path=None,
                                  source_sha256=None, metadata=run_metadata, replace_dates=True)

    def read_values(self, factor_id: str, start: date | str | None = None, end: date | str | None = None) -> pl.DataFrame:
        connection = self._connect()
        try:
            self._initialize(connection)
            registry = self._registry_row(connection, factor_id)
            clauses, parameters = [], []
            if start is not None:
                clauses.append("trade_date >= ?"); parameters.append(_parse_date(start) if isinstance(start, str) else start)
            if end is not None:
                clauses.append("trade_date <= ?"); parameters.append(_parse_date(end) if isinstance(end, str) else end)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            query = f"SELECT trade_date, ts_code, factor_value FROM {_quote_identifier(registry['value_table'])}{where} ORDER BY trade_date, ts_code"
            return pl.from_arrow(connection.execute(query, parameters).to_arrow_table())
        finally:
            connection.close()

    def status(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            self._initialize(connection)
            return connection.execute(
                """
                SELECT r.factor_id, r.status, r.value_table, r.approved_by, r.registered_at,
                       r.activated_at, w.last_complete_trade_date, w.last_run_id
                FROM official_factor_registry r
                LEFT JOIN official_factor_watermarks w USING (factor_id)
                ORDER BY r.factor_id
                """
            ).fetchdf().to_dict(orient="records")
        finally:
            connection.close()


def _store_path(value: str | None) -> Path:
    return Path(value) if value else DEFAULT_STORE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage approved daily factors independently from research Parquet")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create an empty official factor store")
    init.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    register = commands.add_parser("register", help="Manually register a screened research factor")
    register.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH); register.add_argument("--factor-id", required=True)
    register.add_argument("--research-manifest", required=True, type=Path); register.add_argument("--approved-by", required=True); register.add_argument("--approval-note", required=True)
    history = commands.add_parser("import-history", help="Import an approved date range from a research Parquet")
    history.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH); history.add_argument("--factor-id", required=True); history.add_argument("--source", required=True, type=Path)
    history.add_argument("--start", required=True, type=_parse_date); history.add_argument("--end", required=True, type=_parse_date)
    activate = commands.add_parser("activate", help="Allow a registered factor to receive daily writes")
    activate.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH); activate.add_argument("--factor-id", required=True)
    write = commands.add_parser("write-day", help="Atomically replace one active factor's daily cross-section")
    write.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH); write.add_argument("--factor-id", required=True); write.add_argument("--trade-date", required=True, type=_parse_date); write.add_argument("--source", required=True, type=Path)
    status = commands.add_parser("status", help="Show approved factor lifecycle status")
    status.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = OfficialFactorStore(args.store)
    if args.command == "init":
        store.initialize(); print(json.dumps({"store": str(args.store), "status": "initialized"}, ensure_ascii=False)); return
    if args.command == "register":
        result = store.register_factor(args.factor_id, args.research_manifest, args.approved_by, args.approval_note)
    elif args.command == "import-history":
        result = store.import_history(args.factor_id, args.source, args.start, args.end)
    elif args.command == "activate":
        result = store.activate_factor(args.factor_id)
    elif args.command == "write-day":
        result = store.write_day(args.factor_id, args.trade_date, pl.read_parquet(args.source), {"source": str(args.source), "source_sha256": _sha256(args.source)})
    else:
        print(json.dumps(store.status(), ensure_ascii=False, default=str, indent=2)); return
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
