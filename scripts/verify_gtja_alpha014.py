"""Independent acceptance checks for the first real factor/backtest run."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int | float:
    return connection.execute(query).fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GTJA Alpha14 factor and Rust output")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--factor", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    con = duckdb.connect(str(args.catalog), read_only=True)
    try:
        factor = str(args.factor.resolve()).replace("'", "''")
        result = str(args.result.resolve()).replace("'", "''")
        factor_check = f"""
        WITH cal AS (
          SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS index
          FROM observed_calendar WHERE is_observed_market_day
        ), expected AS (
          SELECT q.trade_date, q.ts_code, q.qfq_close - p.qfq_close AS factor_value
          FROM daily_qfq q JOIN cal c USING (trade_date) JOIN cal pcal ON pcal.index = c.index - 5
          JOIN daily_qfq p ON p.ts_code=q.ts_code AND p.trade_date=pcal.trade_date
          WHERE q.qfq_close > 0 AND p.qfq_close > 0
        ), actual AS (SELECT * FROM read_parquet('{factor}'))
        SELECT
          (SELECT count(*) FROM (SELECT * FROM expected EXCEPT SELECT * FROM actual)) +
          (SELECT count(*) FROM (SELECT * FROM actual EXCEPT SELECT * FROM expected)) AS set_difference,
          (SELECT max(abs(e.factor_value-a.factor_value)) FROM expected e JOIN actual a USING(trade_date,ts_code)) AS max_error
        """
        set_difference, max_error = con.execute(factor_check).fetchone()
        duplicate_ic = scalar(con, f"SELECT count(*) FROM (SELECT horizon, return_kind, trade_date, count(*) n FROM read_parquet('{result}/daily_ic.parquet') GROUP BY 1,2,3 HAVING n > 1)")
        cost_identity = scalar(con, f"SELECT count(*) FROM read_parquet('{result}/portfolio_daily.parquet') WHERE abs(net_return - (gross_return - transaction_cost)) > 1e-12")
        invalid_factor = scalar(con, f"SELECT count(*) FROM read_parquet('{factor}') WHERE NOT isfinite(factor_value)")
    finally:
        con.close()
    failures = {
        "factor_set_difference": set_difference,
        "factor_max_error": max_error,
        "duplicate_daily_ic_keys": duplicate_ic,
        "portfolio_cost_identity_violations": cost_identity,
        "nonfinite_factor_values": invalid_factor,
    }
    print(failures)
    if set_difference or max_error > 1e-12 or duplicate_ic or cost_identity or invalid_factor:
        raise SystemExit("Acceptance verification failed")


if __name__ == "__main__":
    main()
