from datetime import date

import polars as pl

from scripts.import_chenditc_csi1000 import build_snapshots, month_ends, qlib_to_ts_code


def test_qlib_to_ts_code() -> None:
    assert qlib_to_ts_code("SH600000") == "600000.SH"
    assert qlib_to_ts_code("SZ000001") == "000001.SZ"


def test_interval_end_is_inclusive() -> None:
    intervals = pl.DataFrame(
        {
            "qlib_symbol": ["SZ000001"],
            "effective_from": [date(2018, 1, 1)],
            "effective_to": [date(2018, 1, 31)],
            "ts_code": ["000001.SZ"],
        }
    )
    dates = pl.DataFrame({"year": [2018], "month": [1], "as_of_date": [date(2018, 1, 31)]})
    # The production builder requires index-sized snapshots; repeat distinct symbols.
    intervals = pl.concat(
        [intervals.with_columns(pl.lit(f"{number:06d}.SZ").alias("ts_code")) for number in range(1000)]
    )
    result = build_snapshots(intervals, dates)
    assert result.height == 1000
    assert result["as_of_date"].unique().to_list() == [date(2018, 1, 31)]


def test_source_end_extends_stale_calendar(tmp_path) -> None:
    calendar = tmp_path / "calendar.parquet"
    pl.DataFrame({"trade_date": [date(2026, 8, 28)]}).write_parquet(calendar)
    result = month_ends(calendar, date(2018, 1, 1), date(2026, 9, 17))
    assert result["as_of_date"].to_list() == [date(2026, 8, 28), date(2026, 9, 17)]
