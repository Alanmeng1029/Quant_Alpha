from __future__ import annotations

import unittest
from datetime import date, timedelta

import polars as pl

from a_share_data.predict import CORE40, LABEL_LAG, TRAIN_DAYS, read_factor_ids, rolling_windows, winsorize_labels


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

if __name__ == "__main__":
    unittest.main()
