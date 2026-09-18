"""Raw-price daily factor bootstrap for configurable index universes.

The module is deliberately separate from ``factors.py``: it cannot overwrite
the active qfq artifacts and it never reads a qfq field.  It is the Python
reference oracle for the Rust production port; its manifest makes that boundary
explicit until the byte-for-byte Rust formula implementation is enabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from a_share_data import factors
from a_share_data import polars_factor_engine
from a_share_data.dual_sleeve import CSI500, CSI1000


def _atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    frame.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, path)


def _membership(catalog: Path, start: str | None, end: str | None, index_codes: tuple[str, ...]) -> pl.DataFrame:
    if not index_codes or any(
        not code or any(not (character.isascii() and (character.isalnum() or character == ".")) for character in code)
        for code in index_codes
    ):
        raise ValueError(f"invalid index code list: {index_codes!r}")
    where = []
    if start:
        where.append(f"trade_date >= DATE '{start}'")
    if end:
        where.append(f"trade_date <= DATE '{end}'")
    clause = (" AND " + " AND ".join(where)) if where else ""
    index_values = ",".join(f"'{code}'" for code in index_codes)
    sql = f"""SELECT c.index_code, cal.trade_date, c.ts_code
               FROM observed_calendar cal
               JOIN index_monthly_constituents c ON c.index_code IN ({index_values})
                AND c.as_of_date = (SELECT max(c2.as_of_date) FROM index_monthly_constituents c2
                                     WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date)
               WHERE cal.is_observed_market_day{clause}"""
    with duckdb.connect(str(catalog), read_only=True) as conn:
        frame = pl.from_arrow(conn.execute(sql).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    # Eligibility is applied only by the raw daily panel below; this query is
    # point-in-time membership and never inherits qfq availability.
    pairs = frame.group_by(["trade_date", "ts_code"]).agg(pl.col("index_code").sort().alias("indices"))
    priority = [code for code in (CSI500, CSI1000) if code in index_codes]
    priority.extend(code for code in index_codes if code not in priority)
    sleeve = pl.lit(priority[-1])
    for code in reversed(priority[:-1]):
        sleeve = pl.when(pl.col("indices").list.contains(code)).then(pl.lit(code)).otherwise(sleeve)
    return pairs.with_columns(sleeve.alias("sleeve")).select("trade_date", "ts_code", "sleeve")


def raw_factor_id(qfq_id: str) -> str:
    return qfq_id.replace("_qfq_v1", "_raw_v1")


def build(catalog: Path, output_root: Path, ids_file: Path, start: str | None, end: str | None,
          index_codes: tuple[str, ...] = (CSI500, CSI1000)) -> dict[str, object]:
    ids = tuple(line.strip() for line in ids_file.read_text().splitlines() if line.strip() and not line.startswith("#"))
    panel = polars_factor_engine.load_calendar_panel(catalog, start, end, lookback_sessions=252, price_basis="raw")
    members = _membership(catalog, start, end, index_codes)
    written: list[str] = []
    for raw_id in ids:
        qfq_id = raw_id.replace("_raw_v1", "_qfq_v1")
        family, number, *_ = qfq_id.split("_")
        definition = factors._definition(family, int(number.removeprefix("alpha")))
        if definition.polars_status != "implemented":
            raise RuntimeError(f"{qfq_id} has no Polars reference evaluator")
        frame = polars_factor_engine.evaluate_factor(qfq_id, panel).collect().sort(["trade_date", "ts_code"])
        if start:
            frame = frame.filter(pl.col("trade_date") >= pl.lit(start).str.to_date())
        if end:
            frame = frame.filter(pl.col("trade_date") <= pl.lit(end).str.to_date())
        frame = frame.join(members, on=["trade_date", "ts_code"], how="inner").drop("sleeve")
        if frame.is_empty():
            raise RuntimeError(f"{qfq_id}: no raw eligible index-universe rows")
        target = output_root / raw_id.removesuffix("_v1") / "v1" / "factor.parquet"
        # A daily run computes only its requested dates (with the 252-session
        # warm-up above) and replaces only those keys.  It must never truncate
        # a completed historical artifact.
        if target.exists():
            previous = pl.read_parquet(target)
            if start:
                previous = previous.filter(pl.col("trade_date") < pl.lit(start).str.to_date())
            if end:
                previous = previous.filter(pl.col("trade_date") > pl.lit(end).str.to_date())
            frame = pl.concat([previous, frame]).unique(["trade_date", "ts_code"], keep="last").sort(["trade_date", "ts_code"])
        _atomic(frame, target)
        manifest = {
            "factor_id": raw_id, "source_factor_id": qfq_id, "price_basis": "raw", "input_fields": ["daily_aggregated.open", "high", "low", "close", "amount_cny/volume_share", "volume_share"],
            "storage_universe": f"point-in-time {' union '.join(index_codes)}", "calculation_universe": "all valid raw daily observations", "runtime": "python_reference_oracle_pending_rust_port",
            "rows": frame.height, "start": str(frame["trade_date"].min()), "end": str(frame["trade_date"].max()), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (target.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        written.append(raw_id)
    return {"factor_count": len(written), "factor_ids": written, "output_root": str(output_root)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--factor-ids-file", type=Path, required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--index-codes", default=f"{CSI500},{CSI1000}")
    args = p.parse_args()
    index_codes = tuple(code.strip() for code in args.index_codes.split(",") if code.strip())
    if not index_codes:
        raise SystemExit("--index-codes must contain at least one code")
    print(json.dumps(build(args.catalog, args.output_root, args.factor_ids_file, args.start, args.end, index_codes), ensure_ascii=False))


if __name__ == "__main__":
    main()
