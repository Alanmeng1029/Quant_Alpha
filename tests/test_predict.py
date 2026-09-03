from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np
import polars as pl

from a_share_data.predict import CORE40, LABEL_LAG, TRAIN_DAYS, project_capped_simplex, read_factor_ids, rolling_windows, winsorize_labels
from a_share_data.ensemble import SelectionSettings, select_features
from a_share_data.factors import _restrict_storage_universe


class PredictionProtocolTests(unittest.TestCase):
    def test_core40_is_fixed_and_complete(self) -> None:
        self.assertEqual(len(CORE40), 40)
        self.assertEqual(CORE40[0], "gtja_alpha001_qfq_v1")
        self.assertEqual(CORE40[-1], "wq_alpha020_qfq_v1")

    def test_window_has_exact_756_days_and_six_day_label_gap(self) -> None:
        days: list[str] = []
        current = date(2018, 1, 1)
        while len(days) < 1_200:
            if current.weekday() < 5:
                days.append(current.isoformat())
            current += timedelta(days=1)
        signal, train = rolling_windows(days)[0]
        signal_idx = days.index(signal)
        self.assertEqual(len(train), TRAIN_DAYS)
        self.assertEqual(days[signal_idx - LABEL_LAG - 1], train[-1])
        self.assertEqual(days[signal_idx - LABEL_LAG - TRAIN_DAYS], train[0])

    def test_winsorization_is_cross_sectional(self) -> None:
        values = [0.0] * 98 + [-10.0, 10.0]
        frame = pl.DataFrame({"trade_date": [date(2024, 1, 2)] * 100, "excess_h1": values})
        out = winsorize_labels(frame, "excess_h1")
        self.assertGreater(out["excess_h1"].min(), -10.0)
        self.assertLess(out["excess_h1"].max(), 10.0)

    def test_factor_id_file_ignores_comments_and_rejects_duplicates(self) -> None:
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factors.txt"
            path.write_text("# selected\nwq_alpha001_qfq_v1\n\ngtja_alpha001_qfq_v1 # keep\n")
            self.assertEqual(read_factor_ids(path), ("wq_alpha001_qfq_v1", "gtja_alpha001_qfq_v1"))
            path.write_text("wq_alpha001_qfq_v1\nwq_alpha001_qfq_v1\n")
            with self.assertRaises(ValueError):
                read_factor_ids(path)

    def test_capped_simplex_preserves_budget_and_cap(self) -> None:
        result = project_capped_simplex(np.array([10.0, 1.0, 1.0]), .5)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertLessEqual(float(result.max()), .5)
        self.assertAlmostEqual(float(result[0]), .5)

    def test_rolling_selection_filters_low_coverage_and_duplicate_features(self) -> None:
        rows = []
        for day in range(8):
            for code in range(5):
                value = float(code - 2)
                rows.append({"trade_date": date(2024, 1, 2) + timedelta(days=day), "ts_code": str(code), "good": value, "duplicate": value, "sparse": value if code == 0 else None, "target": value})
        selected, diagnostics = select_features(pl.DataFrame(rows), ("good", "duplicate", "sparse"), "target", SelectionSettings(min_coverage=.8, min_abs_icir=0, max_features=3, max_abs_correlation=.8))
        self.assertIn("good", selected)
        self.assertNotIn("sparse", selected)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(diagnostics), 2)

    def test_factor_storage_keeps_only_daily_index_union_members(self) -> None:
        day = date(2024, 1, 2)
        values = pl.DataFrame({"trade_date": [day, day, day], "ts_code": ["A", "B", "C"], "factor_value": [1.0, 2.0, 3.0]})
        members = pl.DataFrame({"trade_date": [day, day], "ts_code": ["A", "C"]})
        result = _restrict_storage_universe(values, members)
        self.assertEqual(result["ts_code"].to_list(), ["A", "C"])

if __name__ == "__main__":
    unittest.main()
