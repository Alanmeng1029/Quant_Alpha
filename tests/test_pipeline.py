from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def run_cli(self, root: Path, *args: str) -> None:
        command = [sys.executable, "-m", "a_share_data", args[0], "--data-root", str(root), *args[1:]]
        env = {**__import__("os").environ, "PYTHONPATH": str(PROJECT / "src")}
        subprocess.run(command, check=True, cwd=PROJECT, env=env, text=True)

    def test_small_end_to_end_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "A_stock_database"
            source = root / "minute" / "2026_1min"
            factors = root / "复权因子" / "复权因子_前复权"
            source.mkdir(parents=True)
            factors.mkdir(parents=True)
            (root / "股票列表_沪深.csv").write_text(
                "TS代码,股票代码,股票名称,交易所代码,上市日期\n000001.SZ,000001,样本,SZSE,20100101\n",
                encoding="utf-8-sig",
            )
            with (source / "sz000001_2026.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["时间", "代码", "名称", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "涨幅", "振幅"])
                writer.writerows([
                    ["2026-01-05 09:30:00", "sz000001", "样本", 10, 10, 10, 10, 1, 1000, 0, 0],
                    ["2026-01-05 15:00:00", "sz000001", "样本", 10, 11, 11, 10, 2, 2200, 10, 10],
                    ["2026/01/06 09:30", "sz000001", "样本", 11, 11, 11, 11, 1, 1100, 0, 0],
                    ["2026/01/06 15:00", "sz000001", "样本", 11, 12, 12, 11, 3, 3600, 9, 9],
                ])
            (factors / "000001.SZ.csv").write_text(
                "股票代码,交易日期,复权因子\n000001.SZ,2026-01-05,0.5\n000001.SZ,2026-01-06,0.5\n",
                encoding="utf-8-sig",
            )
            self.run_cli(root, "backfill-minute", "--source-dir", str(source))
            self.run_cli(root, "build-daily", "--start", "2026-01-01", "--end", "2026-01-31")
            self.run_cli(root, "ingest-adjustment", "--snapshot-date", "2026-01-06")
            self.run_cli(root, "build-universe")
            self.run_cli(root, "validate")
            self.run_cli(root, "build-catalog")
            import duckdb

            conn = duckdb.connect(str(root / "lake" / "catalog" / "a_share.duckdb"), read_only=True)
            try:
                value = conn.execute("SELECT close, volume_share, qfq_close, qfq_vwap FROM daily_qfq WHERE trade_date = DATE '2026-01-06'").fetchone()
                self.assertEqual(value, (12.0, 400, 6.0, 5.875))
                # The fixture has only two bars per day, so it is correctly
                # excluded by the full-session eligibility requirement.
                self.assertEqual(conn.execute("SELECT count(*) FROM trading_universe").fetchone()[0], 0)
            finally:
                conn.close()

            # Both maintenance commands may safely rebuild existing data.
            self.run_cli(root, "update-day", "--trade-date", "2026-01-06", "--source-dir", str(source))
            self.run_cli(root, "refresh-month", "--month", "2026-01", "--source-dir", str(source))


if __name__ == "__main__":
    unittest.main()
