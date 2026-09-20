"""Real-holdings, limited-replacement policy for daily A-share signals.

The policy deliberately trades a small, deterministic stock list rather than
rebalancing a target-weight vector every day.  It owns the decision and account
loop so the next decision always sees the shares and cash that were actually
produced by lot rounding and transaction costs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl


INFEASIBLE_EXECUTION_CODES = frozenset({"000937.SZ"})


@dataclass(frozen=True)
class LimitedReplacementConfig:
    target_holdings: int = 80
    entry_rank: int = 80
    exit_rank: int = 96
    max_daily_replacements: int = 5
    max_weight: float = 0.03
    rebalance_to_weight: float = 0.028
    min_new_weight: float = 0.005
    cash_reserve: float = 0.02
    daily_buy_budget: float = 0.10
    daily_sell_budget: float = 0.10
    h1_weight: float = 0.50
    entry_sizing: str = "equal"
    rank_tilt: float = 0.15
    lot_size: int = 100
    initial_capital: float = 10_000_000.0
    buy_bps: float = 2.0
    sell_bps: float = 2.0
    target_fraction: float | None = None
    rebalance_frequency: str = "daily"
    sleeve_nav_targets: dict[str, float] | None = None
    sleeve_target_holdings: dict[str, int] | None = None
    sleeve_rebalance_tolerance: float = 0.0025

    def validate(self) -> None:
        if self.target_fraction is not None and not 0 < self.target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        if not (self.target_holdings > 0 and self.entry_rank >= self.target_holdings and self.exit_rank >= self.entry_rank):
            raise ValueError("ranking limits must satisfy target_holdings <= entry_rank <= exit_rank")
        if not (0 < self.max_daily_replacements <= self.target_holdings):
            raise ValueError("max_daily_replacements must be between 1 and target_holdings")
        if not (0 < self.rebalance_to_weight <= self.max_weight < 1):
            raise ValueError("invalid single-name caps")
        if not (0 < self.min_new_weight <= self.rebalance_to_weight):
            raise ValueError("min_new_weight must not exceed rebalance_to_weight")
        if not (0 <= self.cash_reserve < 1 and 0 < self.daily_buy_budget <= 1 and 0 < self.daily_sell_budget <= 1):
            raise ValueError("invalid cash reserve or daily budgets")
        if not (0 <= self.h1_weight <= 1 and self.entry_sizing in {"equal", "rank_tilt", "cash_balanced"} and 0 <= self.rank_tilt < 1 and self.lot_size > 0 and self.initial_capital > 0 and self.buy_bps >= 0 and self.sell_bps >= 0):
            raise ValueError("invalid execution parameters")
        if self.rebalance_frequency not in {"daily", "weekly"}:
            raise ValueError("rebalance_frequency must be daily or weekly")
        if (self.sleeve_nav_targets is None) != (self.sleeve_target_holdings is None):
            raise ValueError("sleeve targets and holding counts must be configured together")
        if self.sleeve_nav_targets is not None:
            if set(self.sleeve_nav_targets) != set(self.sleeve_target_holdings or {}):
                raise ValueError("sleeve target keys must match")
            if any(value <= 0 for value in self.sleeve_nav_targets.values()):
                raise ValueError("sleeve NAV targets must be positive")
            if sum(self.sleeve_nav_targets.values()) > 1.0 - self.cash_reserve + 1e-12:
                raise ValueError("sleeve NAV targets exceed investable NAV")
            if sum((self.sleeve_target_holdings or {}).values()) != self.target_holdings:
                raise ValueError("sleeve holding counts must sum to target_holdings")
            if self.sleeve_rebalance_tolerance < 0:
                raise ValueError("sleeve rebalance tolerance must be non-negative")


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(json.dumps({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, sort_keys=True).encode()).hexdigest()


def _score_frame(frame: pl.DataFrame, h1_weight: float) -> pl.DataFrame:
    """Score by same-day cross-sectional z-score with deterministic tie order."""
    h5_weight = 1.0 - h1_weight
    h1 = frame.get_column("pred_h1").to_numpy().astype(float, copy=False)
    h5 = frame.get_column("pred_h5").to_numpy().astype(float, copy=False)
    def z(values: np.ndarray) -> np.ndarray:
        std = values.std()
        return (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)
    score = h1_weight * z(h1) + h5_weight * z(h5)
    return frame.with_columns(pl.Series("score", score)).sort(["score", "ts_code"], descending=[True, False]).with_row_index("rank0").with_columns((pl.col("rank0") + 1).alias("rank")).drop("rank0")


def _desired_replacements(ranked_codes: list[str], held_codes: set[str], config: LimitedReplacementConfig) -> tuple[list[str], list[str]]:
    """Return (sell, buy) for normal rank-driven replacements only."""
    ranks = {code: index + 1 for index, code in enumerate(ranked_codes)}
    sellable = sorted((code for code in held_codes if ranks.get(code, config.exit_rank + 1) > config.exit_rank), key=lambda code: (-ranks.get(code, config.exit_rank + 1), code))
    candidates = [code for code in ranked_codes[:config.entry_rank] if code not in held_codes]
    count = min(config.max_daily_replacements, len(sellable), len(candidates))
    return sellable[:count], candidates[:count]


def _load_quotes(catalog: Path, first_day: date, last_day: date) -> dict[tuple[date, str], tuple[float, float, bool]]:
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        rows = pl.from_arrow(conn.execute(
            f"""SELECT trade_date, ts_code, open AS raw_open, qfq_open,
                       observation_status = 'complete_trading' AND amount_cny > 0 AS tradable
                FROM daily_qfq
                WHERE trade_date BETWEEN DATE '{first_day}' AND DATE '{last_day}'
                  AND open > 0 AND qfq_open > 0"""
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        conn.close()
    return {(day, code): (raw, qfq / raw, bool(tradable)) for day, code, raw, qfq, tradable in rows.iter_rows()}


def _load_csi500_opens(catalog: Path, first_day: date, last_day: date) -> dict[date, float]:
    """Load the benchmark using the account's open-to-open timing."""
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        rows = pl.from_arrow(conn.execute(
            f"""SELECT trade_date, open
                  FROM index_daily
                  WHERE index_code = '000905.SH'
                    AND trade_date BETWEEN DATE '{first_day}' AND DATE '{last_day}'
                    AND open > 0
                  ORDER BY trade_date"""
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        conn.close()
    return dict(rows.iter_rows())


def _round_lot(value: float, price: float, lot_size: int) -> float:
    return float(np.floor(value / price / lot_size) * lot_size) if value > 0 and price > 0 else 0.0


def _annual_metrics(daily: pl.DataFrame) -> pl.DataFrame:
    if daily.is_empty():
        return pl.DataFrame()
    return daily.with_columns(pl.col("execution_date").dt.year().alias("year")).group_by("year").agg(
        pl.len().alias("days"),
        ((1 + pl.col("net_return")).product() - 1).alias("net_return"),
        ((1 + pl.col("gross_return")).product() - 1).alias("gross_return"),
        ((1 + pl.col("csi500_return")).product() - 1).alias("csi500_return"),
        pl.col("transaction_cost").sum().alias("transaction_cost"),
        pl.col("buy_turnover").mean().alias("average_buy_turnover"),
        pl.col("sell_turnover").mean().alias("average_sell_turnover"),
        pl.col("holding_count").mean().alias("average_holding_count"),
        pl.col("cash_weight").mean().alias("average_cash_weight"),
    ).sort("year")


def _run_account(predictions: pl.DataFrame, quotes: dict[tuple[date, str], tuple[float, float, bool]], csi500_opens: dict[date, float], config: LimitedReplacementConfig, buy_bps: float, sell_bps: float) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Simulate decisions, orders and holdings using actual shares and cash."""
    by_signal = {(key[0] if isinstance(key, tuple) else key): frame for key, frame in predictions.partition_by("trade_date", as_dict=True).items()}
    shares: dict[str, float] = {}
    qfq_ratio: dict[str, float] = {}
    last_value: dict[str, float] = {}
    cash = config.initial_capital
    previous_ending_value = config.initial_capital
    nav = 1.0
    benchmark_nav = 1.0
    daily_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    first = True
    valid_days: list[tuple[date, date, pl.DataFrame]] = []
    for signal_date in sorted(by_signal):
        raw_frame = by_signal[signal_date].filter(pl.col("pred_h1").is_finite() & pl.col("pred_h5").is_finite() & ~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
        if raw_frame.is_empty():
            continue
        execution_day = raw_frame.get_column("execution_date")[0]
        if execution_day is None:
            continue
        valid_days.append((signal_date, execution_day, raw_frame))

    # The final signal has no next opening price to mark the account.  It is
    # therefore intentionally not traded in this open-to-open simulation.
    base_config = config
    weekly_rebalance_dates: set[date] = set()
    if base_config.rebalance_frequency == "weekly":
        by_week: dict[tuple[int, int], date] = {}
        for signal_date, _, _ in valid_days[:-1]:
            calendar = signal_date.isocalendar()
            by_week[(calendar.year, calendar.week)] = signal_date
        weekly_rebalance_dates = set(by_week.values())
    for day_index, (signal_date, execution_day, raw_frame) in enumerate(valid_days[:-1]):
        if base_config.target_fraction is not None:
            count = max(1, int(np.ceil(raw_frame.height * base_config.target_fraction)))
            config = replace(base_config, target_holdings=count, entry_rank=count,
                             exit_rank=max(count, int(np.ceil(count * base_config.exit_rank / base_config.entry_rank))),
                             max_daily_replacements=min(count, base_config.max_daily_replacements))
        frame = _score_frame(raw_frame, config.h1_weight)
        ranks = dict(frame.select("ts_code", "rank").iter_rows())
        sleeve_by_code = dict(frame.select("ts_code", "sleeve").iter_rows()) if "sleeve" in frame.columns else {}
        signal_codes = set(frame.get_column("ts_code").to_list())
        quotes_today = {code: quotes[(execution_day, code)] for code in set(shares) | signal_codes if (execution_day, code) in quotes}
        for code, quantity in list(shares.items()):
            quote = quotes_today.get(code)
            if quote:
                _, ratio, _ = quote
                if code in qfq_ratio:
                    shares[code] = quantity * ratio / qfq_ratio[code]
                qfq_ratio[code] = ratio
        equity = cash + sum(quantity * quotes_today[code][0] if code in quotes_today else last_value.get(code, 0.0) for code, quantity in shares.items())
        if equity <= 0:
            raise RuntimeError(f"non-positive equity on {execution_day}")
        buy_budget = float("inf") if first else equity * config.daily_buy_budget
        sell_budget = float("inf") if first else equity * config.daily_sell_budget
        bought = sold = transaction_cost = 0.0
        active_holdings = set(shares)

        def sell(code: str, quantity: float, reason: str, budget_exception: bool = False) -> bool:
            nonlocal cash, sold, transaction_cost, sell_budget
            quote = quotes_today.get(code)
            if not quote or not quote[2] or quantity <= 0:
                order_rows.append({"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "side": "sell", "reason": reason, "status": "unfilled", "shares": quantity, "notional": None, "budget_exception": budget_exception})
                return False
            price = quote[0]
            quantity = min(quantity, shares.get(code, 0.0))
            notional = quantity * price
            if not budget_exception and notional > sell_budget + 1e-8:
                order_rows.append({"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "side": "sell", "reason": reason, "status": "skipped_sell_budget", "shares": quantity, "notional": notional, "budget_exception": False})
                return False
            fee = notional * sell_bps / 10_000
            cash += notional - fee
            shares[code] = quantity = shares.get(code, 0.0) - quantity
            if quantity <= 1e-10:
                shares.pop(code, None); qfq_ratio.pop(code, None); last_value.pop(code, None)
            sold += notional; transaction_cost += fee
            if not budget_exception:
                sell_budget -= notional
            event = {"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "side": "sell", "reason": reason, "status": "filled", "shares": notional / price, "notional": notional, "fee": fee, "budget_exception": budget_exception}
            order_rows.append(event); execution_rows.append(event.copy())
            return True

        def buy(code: str, reason: str, requested_notional: float | None = None, exact_amount: bool = False) -> bool:
            nonlocal cash, bought, transaction_cost, buy_budget
            quote = quotes_today.get(code)
            if not quote or not quote[2]:
                order_rows.append({"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "side": "buy", "reason": reason, "status": "unfilled", "shares": None, "notional": None, "budget_exception": False})
                return False
            price = quote[0]
            sleeve = sleeve_by_code.get(code)
            if config.sleeve_nav_targets and sleeve in config.sleeve_nav_targets:
                reference = equity * config.sleeve_nav_targets[sleeve] / config.sleeve_target_holdings[sleeve]
            else:
                reference = equity * (1.0 - config.cash_reserve) / config.target_holdings
            if exact_amount:
                reference = max(0.0, requested_notional or 0.0)
            elif config.entry_sizing == "rank_tilt":
                # A bounded, mean-one tilt: rank 1 receives 1 + tilt times
                # the equal entry amount, rank 80 receives 1 - tilt.  It
                # uses score strength only when opening a position, so it
                # does not reintroduce daily score-chasing rebalances.
                denominator = max(config.entry_rank - 1, 1)
                rank_fraction = (ranks.get(code, config.entry_rank) - 1) / denominator
                reference *= 1.0 + config.rank_tilt * (1.0 - 2.0 * rank_fraction)
            if requested_notional is not None and not exact_amount:
                reference = max(reference, min(requested_notional, equity * config.max_weight))
            # The reserve is a hard account constraint, rather than merely a
            # reference sizing convention.  In particular, lot rounding and
            # rank tilts must not quietly consume it during the initial build.
            affordable = max(0.0, cash - equity * config.cash_reserve) / (1.0 + buy_bps / 10_000)
            minimum = 0.0 if exact_amount and code in shares else equity * config.min_new_weight
            current_value = shares.get(code, 0.0) * price
            cap_room = max(0.0, equity * config.max_weight - current_value)
            notional = min(max(reference, minimum), affordable, buy_budget, cap_room)
            quantity = _round_lot(notional, price, config.lot_size)
            notional = quantity * price
            if notional < minimum - 1e-8:
                # A downward lot round must not turn an otherwise viable entry
                # into a sub-minimum position.  Take the smallest whole lot
                # that satisfies the declared admission rule when the cash and
                # daily budget genuinely support it.
                required_quantity = float(np.ceil(minimum / price / config.lot_size) * config.lot_size)
                required_notional = required_quantity * price
                if required_notional <= affordable + 1e-8 and required_notional <= buy_budget + 1e-8:
                    quantity, notional = required_quantity, required_notional
            if notional < minimum - 1e-8:
                order_rows.append({"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "side": "buy", "reason": reason, "status": "skipped_min_new_weight", "shares": quantity, "notional": notional, "budget_exception": False})
                return False
            fee = notional * buy_bps / 10_000
            cash -= notional + fee
            shares[code] = shares.get(code, 0.0) + quantity
            qfq_ratio[code] = quote[1]
            bought += notional; transaction_cost += fee; buy_budget -= notional
            event = {"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "side": "buy", "reason": reason, "status": "filled", "shares": quantity, "notional": notional, "fee": fee, "budget_exception": False}
            order_rows.append(event); execution_rows.append(event.copy())
            return True

        # Remove constituents that are no longer eligible whenever they trade.
        for code in sorted(active_holdings - signal_codes):
            sell(code, shares.get(code, 0.0), "universe_exit", budget_exception=True)

        # Enforce the cap before ordinary rank replacements.  The 2.8% target
        # avoids repeating tiny risk-reduction orders around a 3% boundary.
        for code in sorted(set(shares)):
            quote = quotes_today.get(code)
            if quote and quote[2] and shares[code] * quote[0] / equity > config.max_weight + 1e-12:
                desired = _round_lot(equity * config.rebalance_to_weight, quote[0], config.lot_size)
                sell(code, max(shares[code] - desired, 0.0), "single_name_cap", budget_exception=True)

        current_codes = set(shares)
        rebalance_today = (first or base_config.rebalance_frequency == "daily" or
                           signal_date in weekly_rebalance_dates)
        if first:
            buy_list = [code for code in frame.get_column("ts_code").to_list()[:config.target_holdings] if code not in current_codes]
            buy_reason = "initial_top_rank"
        elif rebalance_today:
            normal_sells, normal_buys = _desired_replacements(frame.get_column("ts_code").to_list(), current_codes, config)
            # Constituents forced out of the pool consume the same five-entry
            # capacity as ordinary replacements.  Reserve those slots first;
            # otherwise five normal swaps every day could leave a permanent
            # and growing cash vacancy after universe exits.
            preexisting_vacancies = max(0, config.target_holdings - len(current_codes))
            ordinary_slots = max(0, config.max_daily_replacements - preexisting_vacancies)
            normal_sells = normal_sells[:ordinary_slots]
            normal_buys = normal_buys[:ordinary_slots]
            completed_sells = 0
            for code in normal_sells:
                if sell(code, shares.get(code, 0.0), "rank_exit"):
                    completed_sells += 1
            current_codes = set(shares)
            replacements = [code for code in normal_buys[:completed_sells] if code not in current_codes]
            vacancies = max(0, config.target_holdings - len(current_codes) - len(replacements))
            fill_candidates = [code for code in frame.get_column("ts_code").to_list()[:config.entry_rank] if code not in current_codes and code not in replacements]
            buy_list = replacements + fill_candidates[:max(0, min(config.max_daily_replacements - len(replacements), vacancies))]
            buy_reason = "rank_entry"
        else:
            buy_list = []
            buy_reason = "weekly_hold"
        for position, code in enumerate(buy_list):
            requested_notional = None
            if not first and config.entry_sizing == "cash_balanced":
                # Recycle sale proceeds instead of letting gains from exited
                # positions accumulate permanently as idle cash.  Spread the
                # deployable cash over today's remaining admitted names, with
                # the declared single-name cap as a hard ceiling.
                # Include vacancies that cannot be filled today because of
                # the shared replacement limit.  Dividing only by today's
                # buy list overfunds the first few names after a batch index
                # rebalance and leaves no cash for the deferred refills.
                deployable = max(0.0, cash - equity * config.cash_reserve) / (1.0 + buy_bps / 10_000)
                sleeve = sleeve_by_code.get(code)
                if config.sleeve_nav_targets and sleeve in config.sleeve_nav_targets:
                    sleeve_codes = [held for held in shares if sleeve_by_code.get(held) == sleeve]
                    sleeve_value = sum(shares[held] * quotes_today[held][0] for held in sleeve_codes if held in quotes_today)
                    sleeve_capacity = max(0.0, equity * config.sleeve_nav_targets[sleeve] - sleeve_value)
                    sleeve_vacancies = max(1, config.sleeve_target_holdings[sleeve] - len(sleeve_codes))
                    requested_notional = min(deployable, sleeve_capacity) / sleeve_vacancies
                else:
                    remaining = max(config.target_holdings - len(shares), len(buy_list) - position)
                    requested_notional = deployable / remaining if remaining else None
            buy(code, buy_reason, requested_notional)

        if config.sleeve_nav_targets:
            # Keep the capital sleeves close to their declared NAV weights.
            # Partial sizing trades do not change the selected stock list.
            tolerance_value = equity * config.sleeve_rebalance_tolerance

            def sleeve_codes(name: str) -> list[str]:
                return [code for code in shares if sleeve_by_code.get(code) == name and code in quotes_today]

            def sleeve_value(name: str) -> float:
                return sum(shares[code] * quotes_today[code][0] for code in sleeve_codes(name))

            # Trim an overweight sleeve first so its proceeds can fund the
            # underweight sleeve without consuming the cash reserve.
            for sleeve, target_weight in sorted(config.sleeve_nav_targets.items()):
                excess = sleeve_value(sleeve) - equity * target_weight
                if excess <= tolerance_value:
                    continue
                for code in sorted(sleeve_codes(sleeve), key=lambda item: (-ranks.get(item, config.entry_rank + 1), item)):
                    if excess <= tolerance_value:
                        break
                    price = quotes_today[code][0]
                    sellable = max(0.0, shares[code] - config.lot_size)
                    quantity = min(sellable, _round_lot(excess, price, config.lot_size))
                    if quantity > 0 and sell(code, quantity, "sleeve_weight_rebalance", budget_exception=True):
                        excess -= quantity * price

            # Deploy available cash into the underweight sleeve, starting with
            # its most underweight selected names.  The 10% daily buy budget
            # and the 3% single-name cap remain effective.
            for sleeve, target_weight in sorted(config.sleeve_nav_targets.items()):
                deficit = equity * target_weight - sleeve_value(sleeve)
                if deficit <= tolerance_value or buy_budget <= 0:
                    continue
                target_per_name = equity * target_weight / config.sleeve_target_holdings[sleeve]
                candidates = []
                for code in sleeve_codes(sleeve):
                    current_value = shares[code] * quotes_today[code][0]
                    candidates.append((max(0.0, target_per_name - current_value), ranks.get(code, config.entry_rank + 1), code))
                for gap, _, code in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
                    if deficit <= tolerance_value or cash <= equity * config.cash_reserve or buy_budget <= 0:
                        break
                    amount = min(deficit, max(gap, equity * config.sleeve_rebalance_tolerance))
                    before = bought
                    if buy(code, "sleeve_weight_rebalance", amount, exact_amount=True):
                        filled = bought - before
                        deficit -= filled

        # Mark to the next executable open and keep a record of actual shares.
        next_execution = valid_days[day_index + 1][1]
        ending_value = cash
        for code, quantity in shares.items():
            current = quotes_today.get(code)
            future = quotes.get((next_execution, code))
            if current and future:
                adjusted_quantity = quantity * future[1] / current[1]
                ending_value += adjusted_quantity * future[0]
            else:
                ending_value += last_value.get(code, quantity * current[0] if current else 0.0)
        for code, quantity in shares.items():
            if code in quotes_today:
                last_value[code] = quantity * quotes_today[code][0]
                holding_rows.append({"signal_date": signal_date, "execution_date": execution_day, "ts_code": code, "shares": quantity, "raw_open": quotes_today[code][0], "market_value": last_value[code], "weight": last_value[code] / equity})
        gross_return = (ending_value - previous_ending_value + transaction_cost) / previous_ending_value
        net_return = ending_value / previous_ending_value - 1
        nav = ending_value / config.initial_capital
        csi500_open = csi500_opens.get(execution_day)
        csi500_next_open = csi500_opens.get(next_execution)
        benchmark_return = csi500_next_open / csi500_open - 1.0 if csi500_open and csi500_next_open else 0.0
        benchmark_nav *= 1.0 + benchmark_return
        daily_rows.append({"signal_date": signal_date, "execution_date": execution_day, "next_execution_date": next_execution, "gross_return": gross_return, "net_return": net_return, "transaction_cost": transaction_cost / previous_ending_value, "buy_turnover": bought / equity, "sell_turnover": sold / equity, "holding_count": len(shares), "cash": cash, "cash_weight": cash / equity, "equity": ending_value, "nav": nav, "csi500_return": benchmark_return, "csi500_nav": benchmark_nav, "active_return": net_return - benchmark_return})
        previous_ending_value = ending_value
        first = False
    return pl.DataFrame(daily_rows), pl.DataFrame(order_rows), pl.DataFrame(execution_rows), pl.DataFrame(holding_rows), {"final_cash": cash, "final_nav": nav}


def run_limited_replacement_policy(catalog: Path, predictions_path: Path, output: Path, config: LimitedReplacementConfig = LimitedReplacementConfig()) -> dict[str, Any]:
    """Run charged and zero-fee versions of the same real-holdings policy."""
    config.validate()
    predictions = pl.read_parquet(predictions_path).with_columns(pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date)).filter(pl.col("execution_date").is_not_null())
    if predictions.is_empty():
        raise ValueError("predictions are empty")
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        calendar = [row[0] for row in conn.execute("SELECT trade_date FROM observed_calendar WHERE is_observed_market_day ORDER BY trade_date").fetchall()]
    finally:
        conn.close()
    next_days = dict(zip(calendar, calendar[1:]))
    invalid = [(str(signal), str(execution)) for signal, execution in predictions.select("trade_date", "execution_date").unique().iter_rows()
               if next_days.get(signal) != execution]
    if invalid:
        raise ValueError(f"execution_date must be the next trading day: {invalid[:3]}")
    first_day = predictions.get_column("execution_date").min()
    last_day = predictions.get_column("execution_date").max()
    quotes = _load_quotes(catalog, first_day, last_day)
    csi500_opens = _load_csi500_opens(catalog, first_day, last_day)
    missing_benchmark = [str(day) for day in predictions["execution_date"].unique().to_list()
                         if not np.isfinite(csi500_opens.get(day, np.nan)) or csi500_opens.get(day, 0) <= 0]
    if missing_benchmark:
        raise ValueError(f"missing CSI500 execution open: {missing_benchmark[:3]}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"strategy": "limited_replacement_v2", "config": asdict(config), "predictions": str(predictions_path.resolve()), "prediction_fingerprint": _fingerprint(predictions_path), "catalog": str(catalog.resolve()), "execution": "signal at T close; decisions and lot-rounded trades at T+1 open; actual shares and cash feed the next decision", "costs": {"buy_bps": config.buy_bps, "sell_bps": config.sell_bps}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    summaries: dict[str, Any] = {}
    for name, buy_bps, sell_bps in (("charged", config.buy_bps, config.sell_bps), ("zero_cost", 0.0, 0.0)):
        target = output if name == "charged" else output / name
        target.mkdir(parents=True, exist_ok=True)
        daily, orders, executions, holdings, metadata = _run_account(predictions, quotes, csi500_opens, config, buy_bps, sell_bps)
        daily.write_parquet(target / "portfolio_daily.parquet", compression="zstd")
        orders.write_parquet(target / "orders.parquet", compression="zstd")
        executions.write_parquet(target / "executions.parquet", compression="zstd")
        holdings.write_parquet(target / "holdings.parquet", compression="zstd")
        _annual_metrics(daily).write_parquet(target / "annual_metrics.parquet", compression="zstd")
        net = daily.get_column("net_return").to_numpy() if not daily.is_empty() else np.array([])
        active = daily.get_column("active_return").to_numpy() if not daily.is_empty() else np.array([])
        fee_addback = float(np.prod(1.0 + daily.get_column("gross_return").to_numpy()) - 1.0) if daily.height else 0.0
        summary = {"days": daily.height, "net_total_return": float(daily.get_column("nav")[-1] - 1) if daily.height else 0.0, "fee_addback_diagnostic_total_return": fee_addback, "csi500_total_return": float(daily.get_column("csi500_nav")[-1] - 1) if daily.height else 0.0, "average_buy_turnover": float(daily.get_column("buy_turnover").mean()) if daily.height else 0.0, "average_sell_turnover": float(daily.get_column("sell_turnover").mean()) if daily.height else 0.0, "average_holding_count": float(daily.get_column("holding_count").mean()) if daily.height else 0.0, "average_cash_weight": float(daily.get_column("cash_weight").mean()) if daily.height else 0.0, "sum_daily_transaction_cost_rate": float(daily.get_column("transaction_cost").sum()) if daily.height else 0.0, "information_ratio": float(np.mean(active) / np.std(active, ddof=1) * np.sqrt(252)) if len(active) > 1 and np.std(active, ddof=1) > 0 else None, **metadata}
        (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        summaries[name] = summary
    return {"output": str(output), "charged": summaries["charged"], "zero_cost": summaries["zero_cost"]}
