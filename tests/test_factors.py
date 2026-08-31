from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

from a_share_data.factors import build_gtja_alpha014


class FactorTest(unittest.TestCase):
    def test_gtja_alpha014_uses_market_calendar_lag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "A_stock_database"
            catalog = root / "lake" / "catalog" / "a_share.duckdb"
            catalog.parent.mkdir(parents=True)
            days = [date(2024, 1, 2) + timedelta(days=index) for index in range(8)]
            # One stock lacks the third observed market day. Its 5-session lag
            # must be missing rather than incorrectly using five prior rows.
            prices = [
                {"trade_date": day, "ts_code": "000001.SZ", "qfq_close": float(index + 10)}
                for index, day in enumerate(days)
            ] + [
                {"trade_date": day, "ts_code": "000002.SZ", "qfq_close": float(index + 20)}
                for index, day in enumerate(days) if index != 2
            ]
            con = duckdb.connect(str(catalog))
            try:
                con.register("prices", pl.DataFrame(prices).to_arrow())
                con.register("calendar", pl.DataFrame({"trade_date": days, "is_observed_market_day": [True] * len(days)}).to_arrow())
                con.execute("CREATE TABLE daily_qfq AS SELECT * FROM prices")
                con.execute("CREATE TABLE observed_calendar AS SELECT * FROM calendar")
            finally:
                con.close()
            result = build_gtja_alpha014(root, "2024-01-02", "2024-01-09")
            frame = pl.read_parquet(root / "lake" / "derived" / "factors" / "gtja_alpha014_qfq" / "v1" / "factor.parquet")
            self.assertEqual(result["rows"], 5)
            self.assertEqual(frame.filter(pl.col("ts_code") == "000001.SZ")["factor_value"].to_list(), [5.0, 5.0, 5.0])
            self.assertEqual(frame.filter(pl.col("ts_code") == "000002.SZ")["factor_value"].to_list(), [5.0, 5.0])
            manifest = json.loads((root / "lake" / "derived" / "factors" / "gtja_alpha014_qfq" / "v1" / "manifest.json").read_text())
            self.assertEqual(manifest["factor_id"], "gtja_alpha014_qfq_v1")


if __name__ == "__main__":
    unittest.main()
