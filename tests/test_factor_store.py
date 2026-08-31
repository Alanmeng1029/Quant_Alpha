from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

from a_share_data.factor_store import FactorStoreError, OfficialFactorStore


FACTOR_ID = "screened_intraday_factor_v1"


def frame(day: date, values: list[tuple[str, float]]) -> pl.DataFrame:
    return pl.DataFrame({"trade_date": [day] * len(values), "ts_code": [code for code, _ in values], "factor_value": [value for _, value in values]})


class OfficialFactorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = OfficialFactorStore(self.root / "factor_store.duckdb")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"factor_id": FACTOR_ID, "formula": "test", "source_file": "research.py"}), encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def register(self, factor_id: str = FACTOR_ID) -> None:
        manifest = self.manifest
        if factor_id != FACTOR_ID:
            manifest = self.root / f"{factor_id}.json"
            manifest.write_text(json.dumps({"factor_id": factor_id}), encoding="utf-8")
        self.store.register_factor(factor_id, manifest, "tester", "screened")

    def test_only_registered_and_active_factors_accept_daily_writes(self) -> None:
        values = frame(date(2026, 8, 28), [("000001.SZ", 1.0), ("000002.SZ", 2.0)])
        with self.assertRaises(FactorStoreError):
            self.store.write_day(FACTOR_ID, "2026-08-28", values)
        self.register()
        with self.assertRaises(FactorStoreError):
            self.store.write_day(FACTOR_ID, "2026-08-28", values)
        self.store.activate_factor(FACTOR_ID)
        self.store.write_day(FACTOR_ID, "2026-08-28", values)
        self.assertEqual(self.store.read_values(FACTOR_ID).height, 2)

    def test_daily_rewrite_replaces_the_whole_cross_section_without_duplicates(self) -> None:
        self.register(); self.store.activate_factor(FACTOR_ID)
        first = frame(date(2026, 8, 28), [("000001.SZ", 1.0), ("000002.SZ", 2.0)])
        second = frame(date(2026, 8, 28), [("000001.SZ", 3.0)])
        self.store.write_day(FACTOR_ID, "2026-08-28", first)
        self.store.write_day(FACTOR_ID, "2026-08-28", second)
        result = self.store.read_values(FACTOR_ID)
        self.assertEqual(result.to_dicts(), [{"trade_date": date(2026, 8, 28), "ts_code": "000001.SZ", "factor_value": 3.0}])

    def test_history_import_checks_manifest_and_rejects_overlaps(self) -> None:
        source = self.root / "factor.parquet"
        values = pl.concat([frame(date(2026, 8, 27), [("000001.SZ", 1.0)]), frame(date(2026, 8, 28), [("000001.SZ", 2.0)])])
        values.write_parquet(source)
        self.register()
        self.store.import_history(FACTOR_ID, source, "2026-08-27", "2026-08-28")
        self.assertEqual(self.store.read_values(FACTOR_ID).height, 2)
        with self.assertRaisesRegex(FactorStoreError, "overlaps"):
            self.store.import_history(FACTOR_ID, source, "2026-08-28", "2026-08-28")

    def test_history_import_refuses_a_revised_research_manifest(self) -> None:
        source = self.root / "factor.parquet"
        frame(date(2026, 8, 28), [("000001.SZ", 1.0)]).write_parquet(source)
        self.register()
        self.manifest.write_text(json.dumps({"factor_id": FACTOR_ID, "formula": "revised"}), encoding="utf-8")
        with self.assertRaisesRegex(FactorStoreError, "changed after official registration"):
            self.store.import_history(FACTOR_ID, source, "2026-08-28", "2026-08-28")

    def test_invalid_daily_frame_does_not_replace_existing_values_or_watermark(self) -> None:
        self.register(); self.store.activate_factor(FACTOR_ID)
        self.store.write_day(FACTOR_ID, "2026-08-28", frame(date(2026, 8, 28), [("000001.SZ", 1.0)]))
        self.store.write_day(FACTOR_ID, "2026-08-29", frame(date(2026, 8, 29), [("000001.SZ", 2.0)]))
        bad = frame(date(2026, 8, 28), [("000001.SZ", float("nan"))])
        with self.assertRaises(FactorStoreError):
            self.store.write_day(FACTOR_ID, "2026-08-28", bad)
        self.assertEqual(self.store.read_values(FACTOR_ID).filter(pl.col("trade_date") == date(2026, 8, 28))["factor_value"].to_list(), [1.0])
        self.assertEqual(str(self.store.status()[0]["last_complete_trade_date"])[:10], "2026-08-29")

    def test_daily_write_rejects_rows_for_any_other_date(self) -> None:
        self.register(); self.store.activate_factor(FACTOR_ID)
        with self.assertRaisesRegex(FactorStoreError, "only 2026-08-28"):
            self.store.write_day(FACTOR_ID, "2026-08-28", frame(date(2026, 8, 29), [("000001.SZ", 1.0)]))
        self.assertTrue(self.store.read_values(FACTOR_ID).is_empty())

    def test_registered_factors_have_independent_tables(self) -> None:
        other = "screened_intraday_factor_v2"
        self.register(); self.register(other)
        status = {row["factor_id"]: row for row in self.store.status()}
        self.assertNotEqual(status[FACTOR_ID]["value_table"], status[other]["value_table"])


if __name__ == "__main__":
    unittest.main()
