from datetime import date

import polars as pl
import pytest

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


def test_weekly_policy_only_makes_rank_replacements_on_last_signal_of_week() -> None:
    config = LimitedReplacementConfig(
        target_holdings=2, entry_rank=2, exit_rank=3, max_daily_replacements=1,
        max_weight=0.9, rebalance_to_weight=0.8, min_new_weight=0.01,
        cash_reserve=0.02, daily_buy_budget=0.5, daily_sell_budget=0.5,
        initial_capital=10_000.0, rebalance_frequency="weekly",
    )
    signal_days = [date(2024, 1, day) for day in (1, 2, 3, 4, 5, 8)]
    execution_days = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9)]
    rows = []
    for index, (signal, execution) in enumerate(zip(signal_days, execution_days)):
        ranked = ["A", "B", "C", "D"] if index == 0 else ["C", "D", "A", "B"]
        rows.extend({"trade_date": signal, "execution_date": execution, "ts_code": code,
                     "pred_h1": float(4 - rank), "pred_h5": float(4 - rank)}
                    for rank, code in enumerate(ranked))
    predictions = pl.DataFrame(rows)
    quotes = {(day, code): (10.0, 1.0, True) for day in execution_days for code in ("A", "B", "C", "D")}
    _, orders, _, _, _ = _run_account(
        predictions, quotes, {day: 100.0 for day in execution_days}, config, 2.1, 7.1)
    replacements = orders.filter(pl.col("reason") == "rank_exit")
    assert replacements.height == 1
    assert replacements["signal_date"].to_list() == [date(2024, 1, 5)]


def test_policy_rejects_calendar_day_execution(tmp_path):
    import duckdb
    import pytest
    from a_share_data.policy import run_limited_replacement_policy
    catalog = tmp_path / 'calendar.duckdb'
    conn = duckdb.connect(str(catalog))
    conn.execute('CREATE TABLE observed_calendar(trade_date DATE, is_observed_market_day BOOLEAN)')
    conn.execute("INSERT INTO observed_calendar VALUES ('2021-04-02',true),('2021-04-06',true)")
    conn.close()
    path = tmp_path / 'predictions.parquet'
    pl.DataFrame({'trade_date':[date(2021,4,2)], 'execution_date':[date(2021,4,3)]}).write_parquet(path)
    with pytest.raises(ValueError, match='next trading day'):
        run_limited_replacement_policy(catalog, path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_dynamic_top10_initial_holdings():
    codes = [f'S{i:03}' for i in range(30)]
    days = [date(2024,1,3), date(2024,1,4)]
    predictions = pl.DataFrame([{'trade_date':date(2024,1,2+i), 'execution_date':d, 'ts_code':c, 'pred_h1':float(j), 'pred_h5':float(j)} for i,d in enumerate(days) for j,c in enumerate(codes)])
    config = LimitedReplacementConfig(target_fraction=.1, max_weight=.5, rebalance_to_weight=.45)
    quotes = {(d,c):(10.,1.,True) for d in days for c in codes}
    daily, _, _, holdings, _ = _run_account(predictions,quotes,{d:100. for d in days},config,2.1,7.1)
    assert daily['holding_count'][0] == 3
    assert set(holdings['ts_code']) == {'S027','S028','S029'}


def test_initial_build_respects_explicit_sleeve_nav_targets():
    days = [date(2024, 1, 3), date(2024, 1, 4)]
    codes = ["A", "B", "C", "D"]
    sleeves = {"A": "500", "B": "500", "C": "1000", "D": "1000"}
    predictions = pl.DataFrame([
        {"trade_date": date(2024, 1, 2 + index), "execution_date": day, "ts_code": code,
         "pred_h1": float(4 - rank), "pred_h5": float(4 - rank), "sleeve": sleeves[code]}
        for index, day in enumerate(days) for rank, code in enumerate(codes)
    ])
    config = LimitedReplacementConfig(
        target_holdings=4, entry_rank=4, exit_rank=4, max_daily_replacements=1,
        max_weight=.5, rebalance_to_weight=.45, min_new_weight=.005,
        initial_capital=100_000, sleeve_nav_targets={"500": .80, "1000": .18},
        sleeve_target_holdings={"500": 2, "1000": 2},
    )
    quotes = {(day, code): (10.0, 1.0, True) for day in days for code in codes}
    _, _, _, holdings, _ = _run_account(predictions, quotes, {day: 100.0 for day in days}, config, 2.1, 7.1)
    weights = holdings.with_columns(pl.col("ts_code").replace_strict(sleeves).alias("sleeve")).group_by("sleeve").agg(pl.col("weight").sum())
    actual = dict(weights.iter_rows())
    assert actual["500"] == pytest.approx(0.80, abs=.011)
    assert actual["1000"] == pytest.approx(0.18, abs=.011)
