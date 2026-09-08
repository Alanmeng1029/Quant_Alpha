from __future__ import annotations

import numpy as np
import polars as pl
from datetime import date, timedelta

from a_share_data.research_oos import TRAIN_DAYS, block_bootstrap_difference, daily_normalize, next_trading_date_map, validation_folds


def test_inner_folds_are_date_purged_and_cover_last_189_days():
    dates = [str(date(2020, 1, 1) + timedelta(days=index)) for index in range(TRAIN_DAYS)]
    folds = validation_folds(dates)
    assert [len(valid) for _, valid in folds] == [63, 63, 63]
    assert folds[0][1][0] == dates[-189]
    for train, valid in folds:
        assert dates.index(train[-1]) + 6 < dates.index(valid[0])


def test_daily_normalization_is_cross_sectional_only():
    frame = pl.DataFrame({"trade_date": ["2021-01-01"] * 3 + ["2021-01-04"] * 3, "ts_code": list("ABCDEF"), "raw_h1": [1., 2., 3., 10., 20., 30.], "raw_h5": [3., 2., 1., 30., 20., 10.]}).with_columns(pl.col("trade_date").str.to_date())
    result = daily_normalize(frame, "raw_h1", "raw_h5")
    assert np.allclose(result.group_by("trade_date").agg(pl.col("pred_h1").mean()).get_column("pred_h1"), 0.0)
    assert result.filter(pl.col("ts_code") == "A").get_column("pred_h1").item() < 0


def test_block_bootstrap_is_deterministic():
    days = [date(2021, 1, 1) + timedelta(days=index) for index in range(45) for _ in range(3)]
    values = np.arange(45 * 3)
    base = pl.DataFrame({"trade_date": days, "raw_h1": values, "excess_h1": values})
    candidate = base.with_columns((pl.col("raw_h1") * -1).alias("raw_h1"))
    first = block_bootstrap_difference(base, candidate, "h1", repeats=20, block_days=20)
    second = block_bootstrap_difference(base, candidate, "h1", repeats=20, block_days=20)
    assert first == second


def test_execution_date_uses_next_observed_trading_date_not_calendar_day():
    calendar = [date(2021, 1, 7), date(2021, 1, 8), date(2021, 1, 11)]
    assert next_trading_date_map(calendar, [date(2021, 1, 8)]) == {date(2021, 1, 8): date(2021, 1, 11)}
