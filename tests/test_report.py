from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from a_share_data.report import render, render_batch


class ReportTest(unittest.TestCase):
    def test_render_creates_pdf_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"factor_id": "test", "universe": "csi300", "start": "2024-01-01", "end": "2024-01-03", "rows": 4, "transaction_cost_bps": 10.0}), encoding="utf-8")
            pl.DataFrame({"horizon": [1, 1, 1], "return_kind": ["raw", "raw", "raw"], "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"], "pearson_ic": [.1, .2, -.1], "rank_ic": [.1, .2, -.1]}).write_parquet(root / "daily_ic.parquet")
            pl.DataFrame({"horizon": [1] * 10, "return_kind": ["raw"] * 10, "trade_date": ["2024-01-01"] * 10, "group_number": list(range(1, 11)), "mean_return": [0.001] * 10}).write_parquet(root / "group_returns.parquet")
            pl.DataFrame({"execution_mode": ["open", "open", "vwap", "vwap"], "execution_date": ["2024-01-01", "2024-01-02"] * 2, "nav": [1.0, 1.01, 1.0, 1.02]}).write_parquet(root / "portfolio_daily.parquet")
            render(root)
            self.assertGreater((root / "report.pdf").stat().st_size, 1000)
            self.assertIn("test", (root / "report.html").read_text(encoding="utf-8"))

    def test_batch_metrics_use_current_return_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "factor" / "csi500"
            task.mkdir(parents=True)
            (task / "summary.json").write_text(json.dumps({"factor_id": "factor", "universe": "csi500"}), encoding="utf-8")
            rows = []
            for kind in ("close_to_close_raw", "vwap_to_vwap_raw", "twap_to_twap_raw"):
                for horizon in (1, 5, 10, 20):
                    rows.extend({"horizon": horizon, "return_kind": kind, "trade_date": f"2024-01-{day:02d}", "rank_ic": .1} for day in range(1, 4))
            pl.DataFrame(rows).write_parquet(task / "daily_ic.parquet")
            render_batch(root, render_reports=False, jobs=1)
            metrics = pl.read_parquet(root / "batch_metrics.parquet")
            self.assertEqual(metrics.height, 12)
            self.assertEqual(set(metrics["label"].to_list()), {"Close", "VWAP", "TWAP"})
            self.assertIn("close_h1_rank_ic", pl.read_parquet(root / "batch_summary.parquet").columns)


if __name__ == "__main__":
    unittest.main()
