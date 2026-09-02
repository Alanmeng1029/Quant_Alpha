from __future__ import annotations

import unittest
from datetime import date, timedelta

import polars as pl

from a_share_data.predict import CORE40, LABEL_LAG, TRAIN_DAYS, build_top_fraction_targets, read_factor_ids, rolling_windows, winsorize_labels


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

    def test_top_fraction_targets_are_equal_weight_and_tie_stable(self) -> None:
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.parquet"
            rows = [
                {"trade_date": date(2024, 1, 2), "execution_date": date(2024, 1, 3), "ts_code": code, "alpha_daily": alpha}
                for code, alpha in [("C", 1.0), ("A", 2.0), ("B", 2.0), ("D", 0.0), ("E", -1.0)]
            ]
            pl.DataFrame(rows).write_parquet(predictions)
            output = root / "targets.parquet"
            result = build_top_fraction_targets(predictions, output, .40)
            selected = pl.read_parquet(output)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(selected["ts_code"].to_list(), ["A", "B"])
            self.assertEqual(selected["target_weight"].to_list(), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
