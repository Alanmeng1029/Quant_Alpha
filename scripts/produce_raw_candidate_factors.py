"""Produce raw CSI300+CSI500+CSI1000 factors for completed input dates.

This command does not download data, retrain models, select securities, or
change the active strategy.  ``--plan`` performs the same input checks without
writing factor artifacts, which makes it suitable for a future scheduler.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
INDEXES = ("000300.SH", "000905.SH", "000852.SH")
FACTOR_SET = "o2o_raw_daily60_minute45_csi300_csi500_csi1000_v2"


def dates(catalog: Path, start: str | None, end: str | None) -> list[str]:
    with duckdb.connect(str(catalog), read_only=True) as conn:
        rows = conn.execute("""SELECT trade_date::VARCHAR FROM observed_calendar
            WHERE is_observed_market_day AND (? IS NULL OR trade_date>=?::DATE)
              AND (? IS NULL OR trade_date<=?::DATE) ORDER BY trade_date""", [start, start, end, end]).fetchall()
    return [x[0] for x in rows]


def validate_day(catalog: Path, minute_root: Path, day: str) -> dict[str, int]:
    with duckdb.connect(str(catalog), read_only=True) as conn:
        expected, raw_ok = conn.execute("""WITH members AS (
            SELECT DISTINCT c.ts_code FROM index_monthly_constituents c
            WHERE c.index_code IN ('000300.SH','000905.SH','000852.SH') AND c.as_of_date=(
              SELECT max(c2.as_of_date) FROM index_monthly_constituents c2
              WHERE c2.index_code=c.index_code AND c2.as_of_date<=?::DATE))
            SELECT count(*),count(*) FILTER(WHERE d.open>0 AND d.high>0 AND d.low>0 AND d.close>0
              AND d.amount_cny>0 AND d.volume_share>0 AND d.observation_status='complete_trading')
            FROM members m LEFT JOIN daily_aggregated d ON d.ts_code=m.ts_code AND d.trade_date=?::DATE""", [day, day]).fetchone()
    partition = minute_root / f"year={day[:4]}" / f"month={day[5:7]}" / f"trade_date={day}"
    if not list(partition.glob("*.parquet")):
        raise RuntimeError(f"{day}: missing minute partition")
    glob = str(partition / "*.parquet").replace("'", "''")
    with duckdb.connect(str(catalog), read_only=True) as conn:
        full_minute = conn.execute(f"""WITH members AS (
            SELECT DISTINCT c.ts_code FROM index_monthly_constituents c
            WHERE c.index_code IN ('000300.SH','000905.SH','000852.SH') AND c.as_of_date=(
              SELECT max(c2.as_of_date) FROM index_monthly_constituents c2
              WHERE c2.index_code=c.index_code AND c2.as_of_date<=?::DATE)),
            bars AS (SELECT ts_code,count(*) n FROM read_parquet('{glob}') GROUP BY ts_code)
            SELECT count(*) FROM members m JOIN bars b USING(ts_code) WHERE b.n=241""", [day]).fetchone()[0]
    # Constituent history begins at the first available month-end snapshot.
    # Earlier calendar days are legitimate warm-up/replay days with no output
    # universe, not input failures.
    if expected == 0:
        return {"members": 0, "raw_daily": 0, "full_minute": 0}
    # Suspended, not-yet-listed and otherwise incomplete members are excluded
    # by the raw eligibility contract. Every raw-eligible member must have a
    # complete 241-bar minute session, but not every index member must trade.
    if raw_ok == 0 or full_minute < raw_ok:
        raise RuntimeError(f"{day}: incomplete input (members={expected}, raw={raw_ok}, full_minute={full_minute})")
    return {"members": expected, "raw_daily": raw_ok, "full_minute": full_minute}


def invoke(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    catalog = ROOT / "A_stock_database/lake/catalog/a_share.duckdb"
    minute_root = ROOT / "A_stock_database/lake/canonical/minute"
    days = dates(catalog, args.start, args.end or args.start)
    if not days:
        raise SystemExit("no observed trade date in requested range")
    result = {"factor_set": FACTOR_SET, "status": "planned" if args.plan else "running", "dates": days,
              "coverage": {day: validate_day(catalog, minute_root, day) for day in days},
              "generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    if args.plan:
        print(json.dumps(result, ensure_ascii=False)); return
    invoke([sys.executable, "-m", "a_share_data.raw_daily_factors", "--catalog", str(catalog),
        "--output-root", "A_stock_database/lake/derived/factors_raw_o2o_csi300_csi500_csi1000_v2",
        "--factor-ids-file", "configs/candidate_factors_daily_o2o_candidate60_raw_v1.txt", "--start", days[0], "--end", days[-1],
        "--index-codes", ",".join(INDEXES)])
    groups = (("ohlcv_candidates_v1", "A_stock_database/lake/derived/minute_factors/ohlcv_candidates_v1_raw_csi300_csi500_csi1000_v2", 2),
              ("core24", "A_stock_database/lake/derived/minute_factors/core24_raw_csi300_csi500_csi1000_v2", 2),
              ("ohlcv_candidates_v3", "A_stock_database/lake/derived/minute_factors/ohlcv_candidates_v3_raw_csi300_csi500_csi1000_v2", 1))
    for name, output, threads in groups:
        invoke(["cargo", "run", "--release", "-p", "quant-minute-factor", "--", "build", "--catalog", str(catalog),
          "--minute-root", str(minute_root), "--output", output, "--factor-set", name, "--index-codes", ",".join(INDEXES),
          "--raw-eligible-universe", "--start", days[0], "--end", days[-1], "--block-days", "60", "--jobs", str(args.jobs),
          "--threads-per-job", str(threads), "--memory-limit-mb", "12000"])
    result["status"] = "complete"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
