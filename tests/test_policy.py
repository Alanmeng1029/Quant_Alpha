from datetime import date

import polars as pl

from a_share_data.policy import (
    LimitedReplacementConfig,
    _desired_replacements,
    _round_lot,
    _run_account,
    _score_frame,
)


def test_score_ties_sort_by_code() -> None:
    frame = pl.DataFrame({"ts_code": ["000002.SZ", "000001.SZ"], "pred_h1": [1.0, 1.0], "pred_h5": [2.0, 2.0]})
    assert _score_frame(frame, 0.5).get_column("ts_code").to_list() == ["000001.SZ", "000002.SZ"]


def test_replacements_are_capped_and_require_exit_rank() -> None:
    config = LimitedReplacementConfig(target_holdings=5, entry_rank=5, exit_rank=6, max_daily_replacements=2)
    ranked = ["A", "B", "C", "D", "E", "F", "G", "H"]
    sells, buys = _desired_replacements(ranked, {"D", "E", "F", "G", "H"}, config)
    assert sells == ["H", "G"]
    assert buys == ["A", "B"]
    assert _desired_replacements(ranked, {"A", "B", "C", "D", "E"}, config) == ([], [])


def test_lot_rounding_never_rounds_up() -> None:
    assert _round_lot(1_099.0, 10.0, 100) == 100.0
    assert _round_lot(999.0, 10.0, 100) == 0.0


def test_unchanged_members_are_not_rebalanced_daily() -> None:
    config = LimitedReplacementConfig(
        target_holdings=2, entry_rank=2, exit_rank=3, max_daily_replacements=1,
        max_weight=0.9, rebalance_to_weight=0.8, min_new_weight=0.01,
        cash_reserve=0.02, daily_buy_budget=0.1, daily_sell_budget=0.1,
        initial_capital=10_000.0,
    )
    signal_days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    execution_days = [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    predictions = pl.DataFrame({
        "trade_date": [signal_days[0]] * 2 + [signal_days[1]] * 2 + [signal_days[2]] * 2,
        "execution_date": [execution_days[0]] * 2 + [execution_days[1]] * 2 + [execution_days[2]] * 2,
        "ts_code": ["A", "B"] * 3,
        "pred_h1": [2.0, 1.0] * 3,
        "pred_h5": [2.0, 1.0] * 3,
    })
    quotes = {(day, code): (10.0 if day == execution_days[0] else (12.0 if code == "A" else 10.0), 1.0, True) for day in execution_days for code in ("A", "B")}
    csi500 = {day: 100.0 for day in execution_days}
    daily, orders, _, holdings, _ = _run_account(predictions, quotes, csi500, config, 2.1, 7.1)
    assert daily.height == 2
    assert orders.filter((pl.col("reason") == "rank_entry") & (pl.col("status") == "filled")).is_empty()
    first_weights = holdings.filter(pl.col("execution_date") == execution_days[0]).sort("ts_code").get_column("weight").to_list()
    second_weights = holdings.filter(pl.col("execution_date") == execution_days[1]).sort("ts_code").get_column("weight").to_list()
    assert first_weights != second_weights
