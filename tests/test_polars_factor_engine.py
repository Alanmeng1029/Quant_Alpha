from __future__ import annotations

import unittest
from datetime import date, timedelta

import polars as pl

from a_share_data.polars_factor_engine import delay, factor_wq014, rank


class PolarsFactorPrimitiveTest(unittest.TestCase):
    def test_rank_excludes_nulls_from_cross_section_denominator(self) -> None:
        frame = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2)] * 3,
                "ts_code": ["A", "B", "C"],
                "value": [1.0, 2.0, None],
            }
        )
        values = frame.lazy().select(rank(pl.col("value")).alias("rank")).collect()["rank"].to_list()
        self.assertEqual(values[:2], [0.5, 1.0])
        self.assertIsNone(values[2])

    def test_delay_is_scoped_to_each_security(self) -> None:
        frame = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2), date(2024, 1, 3)] * 2,
                "ts_code": ["A", "A", "B", "B"],
                "value": [10.0, 11.0, 20.0, 21.0],
            }
        )
        result = (
            frame.lazy()
            .sort(["ts_code", "trade_date"])
            .select("ts_code", "trade_date", delay(pl.col("value")).alias("previous"))
            .collect()
        )
        self.assertEqual(result["previous"].to_list(), [None, 10.0, None, 20.0])

    def test_wq014_produces_finite_values_after_rolling_warmup(self) -> None:
        days = [date(2024, 1, 2) + timedelta(days=index) for index in range(16)]
        records = []
        for code, offset in [("A", 0.0), ("B", 10.0)]:
            for index, day in enumerate(days):
                close = 10.0 + offset + index * (1.0 if code == "A" else 1.2)
                records.append(
                    {
                        "trade_date": day,
                        "ts_code": code,
                        "qfq_open": close - 0.2,
                        "qfq_high": close + 0.3,
                        "qfq_low": close - 0.5,
                        "qfq_close": close,
                        "qfq_vwap": close - 0.1,
                        "volume_share": 100.0 + offset + index * (2.0 if code == "A" else 3.0),
                    }
                )
        result = factor_wq014(pl.DataFrame(records).lazy().sort(["ts_code", "trade_date"])).collect()
        self.assertGreater(result.height, 0)
        self.assertTrue(result["factor_value"].is_finite().all())


if __name__ == "__main__":
    unittest.main()
