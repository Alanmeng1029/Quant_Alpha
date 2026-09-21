"""Polars-native long-panel factor primitives.

The engine preserves calendar gaps by materializing the observed-market-day ×
security grid lazily before applying any per-security rolling operation.  This
matches the factor contract used by the existing research implementation while
keeping the execution plan columnar and multithreaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable
import ast
import re

import duckdb
import numba
import numpy as np
import polars as pl


KEYS = ("trade_date", "ts_code")
GTJA_REFERENCE = Path("/Users/alanmxy/大学/大学/gtja191Alpha.dos")
WQ_REFERENCE = Path("/Users/alanmxy/大学/大学/alpha101_adjusted.py")


def _over_security(expression: pl.Expr) -> pl.Expr:
    return expression.over("ts_code")


def rank(expression: pl.Expr) -> pl.Expr:
    """Cross-sectional minimum rank scaled to (0, 1] for each trade date."""
    # Pandas ``rank(pct=True)`` divides by the number of non-null observations,
    # not by the whole calendar-expanded universe for the date.
    return (expression.rank(method="min") / expression.count()).over("trade_date")


def scale(expression: pl.Expr, k: float = 1.0) -> pl.Expr:
    """Cross-sectional L1 normalization, matching the WQ Scale operator."""
    denominator = expression.abs().sum().over("trade_date")
    return pl.when(denominator != 0).then(expression * k / denominator).otherwise(None)


def delay(expression: pl.Expr, periods: int = 1) -> pl.Expr:
    return _over_security(expression.shift(periods))


def delta(expression: pl.Expr, periods: int = 1) -> pl.Expr:
    return expression - delay(expression, periods)


def ts_sum(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.rolling_sum(window_size=window, min_samples=window))


def sma(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.rolling_mean(window_size=window, min_samples=window))


def stddev(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.rolling_std(window_size=window, min_samples=window))


def ts_min(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.rolling_min(window_size=window, min_samples=window))


def ts_max(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.rolling_max(window_size=window, min_samples=window))


def ts_rank(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.rolling_rank(window_size=window, method="min").cast(pl.Float64))


def correlation(left: pl.Expr, right: pl.Expr, window: int) -> pl.Expr:
    return _over_security(pl.rolling_corr(left, right, window_size=window, min_samples=window))


def covariance(left: pl.Expr, right: pl.Expr, window: int) -> pl.Expr:
    return _over_security(pl.rolling_cov(left, right, window_size=window, min_samples=window))


def pairwise_min(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return pl.min_horizontal(left, right)


def pairwise_max(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return pl.max_horizontal(left, right)


def ewm_sma(expression: pl.Expr, window: int, weight: int) -> pl.Expr:
    """Chinese SMA(X,N,M): recursive EMA with alpha M/N."""
    return _over_security(expression.ewm_mean(alpha=weight / window, adjust=False, min_samples=1))


def decay_linear(expression: pl.Expr, window: int) -> pl.Expr:
    """Causal linear weighted moving average using native Polars shifts."""
    divisor = window * (window + 1) / 2.0
    weighted = sum(
        delay(expression, lag) * (window - lag)
        for lag in range(window)
    )
    return weighted / divisor


def weighted_average(expression: pl.Expr, weights: tuple[float, ...]) -> pl.Expr:
    return sum(delay(expression, len(weights) - 1 - index) * weight for index, weight in enumerate(weights)) / sum(weights)


def ts_argmax(expression: pl.Expr, window: int) -> pl.Expr:
    maximum = ts_max(expression, window)
    result: pl.Expr = pl.lit(float(window))
    # Earliest occurrence wins, matching NumPy/Pandas argmax within the window.
    for lag in range(window - 1, -1, -1):
        result = pl.when(delay(expression, lag) == maximum).then(float(window - lag)).otherwise(result)
    return pl.when(maximum.is_not_null()).then(result).otherwise(None)


def ts_argmin(expression: pl.Expr, window: int) -> pl.Expr:
    minimum = ts_min(expression, window)
    result: pl.Expr = pl.lit(float(window))
    for lag in range(window - 1, -1, -1):
        result = pl.when(delay(expression, lag) == minimum).then(float(window - lag)).otherwise(result)
    return pl.when(minimum.is_not_null()).then(result).otherwise(None)


def product(expression: pl.Expr, window: int) -> pl.Expr:
    # Alpha101 only applies Product to ranks, which are strictly positive.
    return ts_sum(expression.log(), window).exp()


@dataclass(frozen=True)
class Panel:
    open: pl.Expr
    high: pl.Expr
    low: pl.Expr
    close: pl.Expr
    vol: pl.Expr
    vwap: pl.Expr
    returns: pl.Expr


def panel() -> Panel:
    close = pl.col("qfq_close")
    return Panel(
        open=pl.col("qfq_open"),
        high=pl.col("qfq_high"),
        low=pl.col("qfq_low"),
        close=close,
        vol=pl.col("volume_share"),
        vwap=pl.col("qfq_vwap"),
        returns=close / delay(close) - 1.0,
    )


def load_calendar_panel(
    catalog: Path,
    start: str | None = None,
    end: str | None = None,
    lookback_sessions: int = 0,
    price_basis: str = "qfq",
) -> pl.LazyFrame:
    """Read one price basis and expand to a calendar-aligned long panel."""
    if price_basis not in {"qfq", "raw"}:
        raise ValueError("price_basis must be qfq or raw")
    source = "daily_qfq q" if price_basis == "qfq" else "daily_aggregated q"
    fields = ("q.qfq_open, q.qfq_high, q.qfq_low, q.qfq_close, q.qfq_vwap" if price_basis == "qfq"
              else "q.open AS qfq_open, q.high AS qfq_high, q.low AS qfq_low, q.close AS qfq_close, q.amount_cny/nullif(q.volume_share,0) AS qfq_vwap")
    price_prefix = "q.qfq_" if price_basis == "qfq" else "q."
    # Preserve the established QFQ calendar semantics: suspended sessions with
    # zero volume remain observations in rolling price windows.  Raw factors
    # use the stricter tradability input contract and exclude zero-volume bars.
    volume_clause = "q.volume_share >= 0" if price_basis == "qfq" else "q.volume_share > 0"
    clauses = [
        f"{price_prefix}open > 0", f"{price_prefix}high > 0", f"{price_prefix}low > 0",
        f"{price_prefix}close > 0", volume_clause,
    ]
    if end:
        clauses.append(f"q.trade_date <= DATE '{end}'")
    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        effective_start = start
        if start and lookback_sessions:
            prior = connection.execute(
                "SELECT min(trade_date) FROM ("
                "SELECT trade_date FROM observed_calendar "
                "WHERE is_observed_market_day AND trade_date < ? "
                "ORDER BY trade_date DESC LIMIT ?"
                ")",
                [start, lookback_sessions],
            ).fetchone()[0]
            if prior is not None:
                effective_start = str(prior)
        if effective_start:
            clauses.append(f"q.trade_date >= DATE '{effective_start}'")
        bars = pl.from_arrow(connection.execute(
            "SELECT q.trade_date, q.ts_code, " + fields + ", q.volume_share "
            "FROM " + source + " WHERE " + " AND ".join(clauses)
        ).arrow())
        calendar_where = ["is_observed_market_day"]
        if effective_start:
            calendar_where.append(f"trade_date >= DATE '{effective_start}'")
        if end:
            calendar_where.append(f"trade_date <= DATE '{end}'")
        calendar = pl.from_arrow(connection.execute(
            "SELECT trade_date FROM observed_calendar WHERE " + " AND ".join(calendar_where)
        ).arrow())
        index_panel = pl.from_arrow(connection.execute(
            "SELECT trade_date, avg(open) AS index_open, avg(close) AS index_close "
            "FROM index_daily WHERE index_code IN ('000300.SH', '000905.SH') "
            "GROUP BY trade_date"
        ).arrow())
    finally:
        connection.close()
    securities = bars.select("ts_code").unique()
    return (
        calendar.lazy().join(securities.lazy(), how="cross")
        .join(bars.lazy(), on=list(KEYS), how="left")
        .join(index_panel.lazy(), on="trade_date", how="left")
        .sort(["ts_code", "trade_date"])
    )


def factor_wq014(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Polars reference implementation for WQ Alpha014."""
    # Materialize each window stage.  Polars correctly rejects neither nested
    # ``over`` calls nor their syntax, but a nested window can otherwise have
    # ambiguous grouping semantics and produce all-null output.
    return (
        frame
        .with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(delta(pl.col("_returns"), 3).alias("_returns_delta_3"))
        .with_columns(rank(pl.col("_returns_delta_3")).alias("_rank_returns_delta_3"))
        .with_columns(correlation(pl.col("qfq_open"), pl.col("volume_share"), 10).alias("_open_volume_corr_10"))
        .select(*KEYS, (-pl.col("_rank_returns_delta_3") * pl.col("_open_volume_corr_10")).alias("factor_value"))
        .filter(pl.col("factor_value").is_finite())
    )


def factor_wq001(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(stddev(pl.col("_returns"), 20).alias("_returns_std"))
        .with_columns(pl.when(pl.col("_returns") < 0).then(pl.col("_returns_std")).otherwise(pl.col("qfq_close")).alias("_base"))
        .with_columns(ts_argmax(pl.col("_base").pow(2), 5).alias("_argmax"))
        .with_columns(rank(pl.col("_argmax")).alias("factor_value")),
        pl.col("factor_value") - 0.5,
    )


def _finish(frame: pl.LazyFrame, expression: pl.Expr) -> pl.LazyFrame:
    return frame.select(*KEYS, expression.cast(pl.Float64).alias("factor_value")).filter(pl.col("factor_value").is_finite())


def factor_wq002(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(pl.when(pl.col("volume_share") > 0).then(pl.col("volume_share").log()).otherwise(None).alias("_log_vol"))
        .with_columns(delta(pl.col("_log_vol"), 2).alias("_delta_log_vol"), ((pl.col("qfq_close") - pl.col("qfq_open")) / pl.col("qfq_open")).alias("_intraday_return"))
        .with_columns(rank(pl.col("_delta_log_vol")).alias("_rank_delta_log_vol"), rank(pl.col("_intraday_return")).alias("_rank_intraday_return"))
        .with_columns(correlation(pl.col("_rank_delta_log_vol"), pl.col("_rank_intraday_return"), 6).alias("_corr")),
        -pl.col("_corr"),
    )


def factor_wq003(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("qfq_open")).alias("_rank_open"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(correlation(pl.col("_rank_open"), pl.col("_rank_vol"), 10).alias("_corr")),
        -pl.col("_corr"),
    )


def factor_wq004(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("qfq_low")).alias("_rank_low"))
        .with_columns(ts_rank(pl.col("_rank_low"), 9).alias("_ts_rank_low")),
        -pl.col("_ts_rank_low").cast(pl.Float64),
    )


def factor_wq005(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("qfq_vwap"), 10).alias("_vwap_mean_10"))
        .with_columns((pl.col("qfq_open") - pl.col("_vwap_mean_10")).alias("_open_vwap_mean"), (pl.col("qfq_close") - pl.col("qfq_vwap")).alias("_close_vwap"))
        .with_columns(rank(pl.col("_open_vwap_mean")).alias("_rank_open_vwap_mean"), rank(pl.col("_close_vwap")).alias("_rank_close_vwap")),
        -pl.col("_rank_open_vwap_mean") * pl.col("_rank_close_vwap"),
    )


def factor_wq006(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame.with_columns(correlation(pl.col("qfq_open"), pl.col("volume_share"), 10).alias("_corr")), -pl.col("_corr"))


def factor_wq007(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("volume_share"), 20).alias("_adv20"), delta(pl.col("qfq_close"), 7).alias("_close_delta_7"))
        .with_columns(ts_rank(pl.col("_close_delta_7").abs(), 60).alias("_ts_rank_abs_delta"))
        .with_columns(pl.when(pl.col("volume_share") > pl.col("_adv20")).then(-pl.col("_ts_rank_abs_delta") * pl.col("_close_delta_7").sign()).otherwise(-1.0).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq008(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(ts_sum(pl.col("qfq_open"), 5).alias("_open_sum_5"), ts_sum(pl.col("_returns"), 5).alias("_returns_sum_5"))
        .with_columns((pl.col("_open_sum_5") * pl.col("_returns_sum_5")).alias("_open_ret_sum_product"))
        .with_columns(delay(pl.col("_open_ret_sum_product"), 10).alias("_delayed_product"))
        .with_columns(rank(pl.col("_open_ret_sum_product") - pl.col("_delayed_product")).alias("_rank_value")),
        -pl.col("_rank_value"),
    )


def factor_wq009(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_close")).alias("_close_delta"))
        .with_columns(ts_min(pl.col("_close_delta"), 5).alias("_delta_min"), ts_max(pl.col("_close_delta"), 5).alias("_delta_max"))
        .with_columns(pl.when((pl.col("_delta_min") > 0) | (pl.col("_delta_max") < 0)).then(-pl.col("_close_delta")).otherwise(pl.col("_close_delta")).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq010(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_close")).alias("_close_delta"))
        .with_columns(ts_min(pl.col("_close_delta"), 4).alias("_delta_min"), ts_max(pl.col("_close_delta"), 4).alias("_delta_max"))
        .with_columns(pl.when((pl.col("_delta_min") > 0) | (pl.col("_delta_max") < 0)).then(-pl.col("_close_delta")).otherwise(pl.col("_close_delta")).alias("_reversal"))
        .with_columns(rank(pl.col("_reversal")).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq011(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns((pl.col("qfq_vwap") - pl.col("qfq_close")).alias("_vwap_close"), delta(pl.col("volume_share"), 3).alias("_vol_delta"))
        .with_columns(ts_max(pl.col("_vwap_close"), 3).alias("_vwap_close_max"), ts_min(pl.col("_vwap_close"), 3).alias("_vwap_close_min"))
        .with_columns(rank(pl.col("_vwap_close_max")).alias("_rank_max"), rank(pl.col("_vwap_close_min")).alias("_rank_min"), rank(pl.col("_vol_delta")).alias("_rank_delta")),
        (pl.col("_rank_max") + pl.col("_rank_min")) * pl.col("_rank_delta"),
    )


def factor_wq012(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("volume_share")).alias("_vol_delta"), delta(pl.col("qfq_close")).alias("_close_delta")),
        pl.col("_vol_delta").sign() * -pl.col("_close_delta"),
    )


def factor_wq013(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("qfq_close")).alias("_rank_close"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(covariance(pl.col("_rank_close"), pl.col("_rank_vol"), 5).alias("_cov")),
        -pl.col("_cov"),
    )


def factor_wq015(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("qfq_high")).alias("_rank_high"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(correlation(pl.col("_rank_high"), pl.col("_rank_vol"), 3).alias("_corr"))
        .with_columns(rank(pl.col("_corr")).alias("_rank_corr"))
        .with_columns(ts_sum(pl.col("_rank_corr"), 3).alias("_sum_rank_corr")),
        -pl.col("_sum_rank_corr"),
    )


def factor_wq016(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("qfq_high")).alias("_rank_high"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(covariance(pl.col("_rank_high"), pl.col("_rank_vol"), 5).alias("_cov")),
        -pl.col("_cov"),
    )


def factor_wq017(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delta(pl.col("qfq_close")).alias("_close_delta"))
        .with_columns(sma(pl.col("volume_share"), 20).alias("_adv20"), ts_rank(pl.col("qfq_close"), 10).alias("_ts_rank_close"), delta(pl.col("_close_delta")).alias("_close_second_delta"))
        .with_columns(ts_rank(pl.col("volume_share") / pl.col("_adv20"), 5).alias("_ts_rank_vol_adv"))
        .with_columns(rank(pl.col("_ts_rank_close")).alias("_rank_close"), rank(pl.col("_close_second_delta")).alias("_rank_second_delta"), rank(pl.col("_ts_rank_vol_adv")).alias("_rank_vol_adv")),
        -pl.col("_rank_close") * pl.col("_rank_second_delta") * pl.col("_rank_vol_adv"),
    )


def factor_wq018(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns((pl.col("qfq_close") - pl.col("qfq_open")).alias("_close_open"))
        .with_columns(stddev(pl.col("_close_open").abs(), 5).alias("_std_abs_close_open"), correlation(pl.col("qfq_close"), pl.col("qfq_open"), 10).alias("_corr"))
        .with_columns(rank(pl.col("_std_abs_close_open") + pl.col("_close_open") + pl.col("_corr")).alias("_rank_value")),
        -pl.col("_rank_value"),
    )


def factor_wq019(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"), (pl.col("qfq_close") - delay(pl.col("qfq_close"), 7) + delta(pl.col("qfq_close"), 7)).alias("_signal"))
        .with_columns(ts_sum(pl.col("_returns"), 250).alias("_returns_sum_250"))
        .with_columns(rank(1 + pl.col("_returns_sum_250")).alias("_rank_returns")),
        -pl.col("_signal").sign() * (1 + pl.col("_rank_returns")),
    )


def factor_wq020(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns((pl.col("qfq_open") - delay(pl.col("qfq_high"))).alias("_open_high"), (pl.col("qfq_open") - delay(pl.col("qfq_close"))).alias("_open_close"), (pl.col("qfq_open") - delay(pl.col("qfq_low"))).alias("_open_low"))
        .with_columns(rank(pl.col("_open_high")).alias("_rank_high"), rank(pl.col("_open_close")).alias("_rank_close"), rank(pl.col("_open_low")).alias("_rank_low")),
        -pl.col("_rank_high") * pl.col("_rank_close") * pl.col("_rank_low"),
    )


def factor_wq021(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("qfq_close"), 8).alias("_close_mean_8"), stddev(pl.col("qfq_close"), 8).alias("_close_std_8"), sma(pl.col("qfq_close"), 2).alias("_close_mean_2"), sma(pl.col("volume_share"), 20).alias("_adv20"))
        .with_columns(pl.when((pl.col("_close_mean_8") + pl.col("_close_std_8") < pl.col("_close_mean_2")) | (pl.col("volume_share") / pl.col("_adv20") >= 1)).then(-1.0).otherwise(1.0).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq022(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(correlation(pl.col("qfq_high"), pl.col("volume_share"), 5).alias("_high_volume_corr_5"), stddev(pl.col("qfq_close"), 20).alias("_close_std_20"))
        .with_columns(delta(pl.col("_high_volume_corr_5"), 5).alias("_corr_delta_5"), rank(pl.col("_close_std_20")).alias("_rank_close_std")),
        -pl.col("_corr_delta_5") * pl.col("_rank_close_std"),
    )


def factor_wq023(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("qfq_high"), 20).alias("_high_mean_20"), delta(pl.col("qfq_high"), 2).alias("_high_delta_2"))
        .with_columns(pl.when(pl.col("_high_mean_20") < pl.col("qfq_high")).then(-pl.col("_high_delta_2")).otherwise(0.0).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq024(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("qfq_close"), 100).alias("_close_mean_100"), delay(pl.col("qfq_close"), 100).alias("_close_delay_100"), delta(pl.col("qfq_close"), 3).alias("_close_delta_3"), ts_min(pl.col("qfq_close"), 100).alias("_close_min_100"))
        .with_columns(delta(pl.col("_close_mean_100"), 100).alias("_close_mean_delta_100"))
        .with_columns(pl.when(pl.col("_close_mean_delta_100") / pl.col("_close_delay_100") <= 0.05).then(-pl.col("_close_delta_3")).otherwise(-(pl.col("qfq_close") - pl.col("_close_min_100"))).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq025(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"), sma(pl.col("volume_share"), 20).alias("_adv20"))
        .with_columns(rank(-pl.col("_returns") * pl.col("_adv20") * pl.col("qfq_vwap") * (pl.col("qfq_high") - pl.col("qfq_close"))).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq026(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(ts_rank(pl.col("volume_share"), 5).alias("_ts_rank_vol"), ts_rank(pl.col("qfq_high"), 5).alias("_ts_rank_high"))
        .with_columns(correlation(pl.col("_ts_rank_vol"), pl.col("_ts_rank_high"), 5).alias("_corr"))
        .with_columns(ts_max(pl.col("_corr"), 3).alias("_corr_max_3")),
        -pl.col("_corr_max_3"),
    )


def factor_wq027(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(rank(pl.col("volume_share")).alias("_rank_vol"), rank(pl.col("qfq_vwap")).alias("_rank_vwap"))
        .with_columns(correlation(pl.col("_rank_vol"), pl.col("_rank_vwap"), 6).alias("_corr"))
        .with_columns((sma(pl.col("_corr"), 2) / 2.0).alias("_corr_mean"))
        .with_columns(pl.when(rank(pl.col("_corr_mean")) > 0.5).then(-1.0).otherwise(1.0).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq028(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 20).alias("_adv20"))
        .with_columns(correlation(pl.col("_adv20"), pl.col("qfq_low"), 5).alias("_corr"))
        .with_columns((pl.col("_corr") + (pl.col("qfq_high") + pl.col("qfq_low")) / 2.0 - pl.col("qfq_close")).alias("_signal")),
        scale(pl.col("_signal")),
    )


def factor_wq036(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_close") - pl.col("qfq_open")).alias("_close_open"), delay(pl.col("volume_share")).alias("_volume_delay"), (pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"), sma(pl.col("volume_share"), 20).alias("_adv20"), sma(pl.col("qfq_close"), 200).alias("_close_mean_200"))
        .with_columns(correlation(pl.col("_close_open"), pl.col("_volume_delay"), 15).alias("_corr_close_volume"), delay(-pl.col("_returns"), 6).alias("_negative_return_delay"), correlation(pl.col("qfq_vwap"), pl.col("_adv20"), 6).alias("_corr_vwap_adv"))
        .with_columns(ts_rank(pl.col("_negative_return_delay"), 5).alias("_negative_return_rank"))
        .with_columns(rank(pl.col("_corr_close_volume")).alias("_rank_corr_close_volume"), rank(pl.col("qfq_open") - pl.col("qfq_close")).alias("_rank_open_close"), rank(pl.col("_negative_return_rank")).alias("_rank_return"), rank(pl.col("_corr_vwap_adv").abs()).alias("_rank_abs_corr"), rank((pl.col("_close_mean_200") / 200.0 - pl.col("qfq_open")) * pl.col("_close_open")).alias("_rank_mean_gap")),
        2.21 * pl.col("_rank_corr_close_volume") + 0.7 * pl.col("_rank_open_close") + 0.73 * pl.col("_rank_return") + pl.col("_rank_abs_corr") + 0.6 * pl.col("_rank_mean_gap"),
    )


def factor_wq037(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns((pl.col("qfq_open") - pl.col("qfq_close")).alias("_open_close"))
        .with_columns(delay(pl.col("_open_close")).alias("_open_close_delay"))
        .with_columns(correlation(pl.col("_open_close_delay"), pl.col("qfq_close"), 200).alias("_corr"))
        .with_columns(rank(pl.col("_corr")).alias("_rank_corr"), rank(pl.col("_open_close")).alias("_rank_open_close")),
        pl.col("_rank_corr") + pl.col("_rank_open_close"),
    )


def factor_wq038(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(ts_rank(pl.col("qfq_open"), 10).alias("_open_rank"), (pl.col("qfq_close") / pl.col("qfq_open")).alias("_close_open"))
        .with_columns(rank(pl.col("_open_rank")).alias("_rank_open"), rank(pl.col("_close_open")).alias("_rank_close_open")),
        -pl.col("_rank_open") * pl.col("_rank_close_open"),
    )


def factor_wq039(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 20).alias("_adv20"), delta(pl.col("qfq_close"), 7).alias("_close_delta"), (pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(decay_linear(pl.col("volume_share") / pl.col("_adv20"), 9).alias("_volume_decay"), ts_sum(pl.col("_returns"), 250).alias("_returns_sum"))
        .with_columns(rank(pl.col("_volume_decay")).alias("_rank_volume_decay"), rank(pl.col("_returns_sum")).alias("_rank_returns"))
        .with_columns(rank(pl.col("_close_delta") * (1.0 - pl.col("_rank_volume_decay"))).alias("_rank_signal")),
        -pl.col("_rank_signal") * (1.0 + pl.col("_rank_returns")),
    )


def factor_wq040(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(stddev(pl.col("qfq_high"), 10).alias("_high_std"), correlation(pl.col("qfq_high"), pl.col("volume_share"), 10).alias("_corr"))
        .with_columns(rank(pl.col("_high_std")).alias("_rank_std")),
        -pl.col("_rank_std") * pl.col("_corr"),
    )


def factor_wq045(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delay(pl.col("qfq_close"), 5).alias("_close_delay5"), correlation(pl.col("qfq_close"), pl.col("volume_share"), 2).alias("_close_volume_corr"), ts_sum(pl.col("qfq_close"), 5).alias("_close_sum5"), ts_sum(pl.col("qfq_close"), 20).alias("_close_sum20"))
        .with_columns(sma(pl.col("_close_delay5"), 20).alias("_mean_delay_close"), correlation(pl.col("_close_sum5"), pl.col("_close_sum20"), 2).alias("_sum_corr"))
        .with_columns(rank(pl.col("_mean_delay_close")).alias("_rank_mean"), rank(pl.col("_sum_corr")).alias("_rank_sum_corr")),
        -pl.col("_rank_mean") * pl.col("_close_volume_corr") * pl.col("_rank_sum_corr"),
    )


def factor_wq047(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 20).alias("_adv20"), sma(pl.col("qfq_high"), 5).alias("_high_mean5"), delay(pl.col("qfq_vwap"), 5).alias("_vwap_delay5"))
        .with_columns(rank(1.0 / pl.col("qfq_close")).alias("_rank_inverse_close"), rank(pl.col("qfq_high") - pl.col("qfq_close")).alias("_rank_high_close"), rank(pl.col("qfq_vwap") - pl.col("_vwap_delay5")).alias("_rank_vwap_delta")),
        (pl.col("_rank_inverse_close") * pl.col("volume_share") / pl.col("_adv20")) * (pl.col("qfq_high") * pl.col("_rank_high_close") / (pl.col("_high_mean5") / 5.0)) - pl.col("_rank_vwap_delta"),
    )


def _wq_trend(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        ((delay(pl.col("qfq_close"), 20) - delay(pl.col("qfq_close"), 10)) / 10.0 - (delay(pl.col("qfq_close"), 10) - pl.col("qfq_close")) / 10.0).alias("_trend"),
        delta(pl.col("qfq_close")).alias("_close_delta"),
    )


def factor_wq046(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        _wq_trend(frame).with_columns(pl.when(pl.col("_trend") < 0).then(1.0).when(pl.col("_trend") > 0.25).then(-1.0).otherwise(-pl.col("_close_delta")).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq049(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        _wq_trend(frame).with_columns(pl.when(pl.col("_trend") < -0.1).then(1.0).otherwise(-pl.col("_close_delta")).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq051(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        _wq_trend(frame).with_columns(pl.when(pl.col("_trend") < -0.05).then(1.0).otherwise(-pl.col("_close_delta")).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_wq071(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 180).alias("_adv180"))
        .with_columns(ts_rank(pl.col("qfq_close"), 3).alias("_close_rank3"), ts_rank(pl.col("_adv180"), 12).alias("_adv_rank12"))
        .with_columns(correlation(pl.col("_close_rank3"), pl.col("_adv_rank12"), 18).alias("_corr"), (rank(pl.col("qfq_low") + pl.col("qfq_open") - 2.0 * pl.col("qfq_vwap")).pow(2)).alias("_rank_square"))
        .with_columns(decay_linear(pl.col("_corr"), 4).alias("_corr_decay"), decay_linear(pl.col("_rank_square"), 16).alias("_rank_decay"))
        .with_columns(ts_rank(pl.col("_corr_decay"), 16).alias("_p1"), ts_rank(pl.col("_rank_decay"), 4).alias("_p2")),
        pairwise_max(pl.col("_p1"), pl.col("_p2")),
    )


def factor_wq073(frame: pl.LazyFrame) -> pl.LazyFrame:
    weighted = pl.col("qfq_open") * 0.147155 + pl.col("qfq_low") * (1.0 - 0.147155)
    return _finish(
        frame
        .with_columns(delta(pl.col("qfq_vwap"), 5).alias("_vwap_delta"), weighted.alias("_weighted"))
        .with_columns(delta(pl.col("_weighted"), 2).alias("_weighted_delta"))
        .with_columns(decay_linear(pl.col("_vwap_delta"), 3).alias("_vwap_decay"), decay_linear(-pl.col("_weighted_delta") / pl.col("_weighted"), 3).alias("_weighted_decay"))
        .with_columns(rank(pl.col("_vwap_decay")).alias("_p1"), ts_rank(pl.col("_weighted_decay"), 17).alias("_p2")),
        -pairwise_max(pl.col("_p1"), pl.col("_p2")),
    )


def factor_wq077(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 40).alias("_adv40"), ((pl.col("qfq_high") + pl.col("qfq_low")) / 2.0 - pl.col("qfq_vwap")).alias("_price"))
        .with_columns(correlation((pl.col("qfq_high") + pl.col("qfq_low")) / 2.0, pl.col("_adv40"), 3).alias("_corr"))
        .with_columns(decay_linear(pl.col("_price"), 20).alias("_price_decay"), decay_linear(pl.col("_corr"), 6).alias("_corr_decay"))
        .with_columns(rank(pl.col("_price_decay")).alias("_p1"), rank(pl.col("_corr_decay")).alias("_p2")),
        pairwise_min(pl.col("_p1"), pl.col("_p2")),
    )


def factor_wq088(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 60).alias("_adv60"), (rank(pl.col("qfq_open")) + rank(pl.col("qfq_low")) - rank(pl.col("qfq_high")) - rank(pl.col("qfq_close"))).alias("_rank_spread"))
        .with_columns(ts_rank(pl.col("qfq_close"), 8).alias("_close_rank"), ts_rank(pl.col("_adv60"), 21).alias("_adv_rank"))
        .with_columns(correlation(pl.col("_close_rank"), pl.col("_adv_rank"), 8).alias("_corr"))
        .with_columns(decay_linear(pl.col("_rank_spread"), 8).alias("_spread_decay"), decay_linear(pl.col("_corr"), 7).alias("_corr_decay"))
        .with_columns(rank(pl.col("_spread_decay")).alias("_p1"), ts_rank(pl.col("_corr_decay"), 3).alias("_p2")),
        pairwise_min(pl.col("_p1"), pl.col("_p2")),
    )


def factor_wq092(frame: pl.LazyFrame) -> pl.LazyFrame:
    condition = (
        ((pl.col("qfq_high") + pl.col("qfq_low")) / 2.0 + pl.col("qfq_close"))
        < (pl.col("qfq_low") + pl.col("qfq_open"))
    ).cast(pl.Float64)
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 30).alias("_adv30"), condition.alias("_condition"))
        .with_columns(correlation(rank(pl.col("qfq_low")), rank(pl.col("_adv30")), 8).alias("_corr"))
        .with_columns(decay_linear(pl.col("_condition"), 15).alias("_condition_decay"), decay_linear(pl.col("_corr"), 7).alias("_corr_decay"))
        .with_columns(ts_rank(pl.col("_condition_decay"), 19).alias("_p1"), ts_rank(pl.col("_corr_decay"), 7).alias("_p2")),
        pairwise_min(pl.col("_p1"), pl.col("_p2")),
    )


def factor_wq096(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("volume_share"), 60).alias("_adv60"), rank(pl.col("qfq_vwap")).alias("_rank_vwap"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(correlation(pl.col("_rank_vwap"), pl.col("_rank_vol"), 4).alias("_corr1"), ts_rank(pl.col("qfq_close"), 7).alias("_close_rank"), ts_rank(pl.col("_adv60"), 4).alias("_adv_rank"))
        .with_columns(correlation(pl.col("_close_rank"), pl.col("_adv_rank"), 4).alias("_corr2"))
        .with_columns(ts_argmax(pl.col("_corr2"), 13).alias("_argmax"))
        .with_columns(decay_linear(pl.col("_corr1"), 4).alias("_decay1"), decay_linear(pl.col("_argmax"), 14).alias("_decay2"))
        .with_columns(ts_rank(pl.col("_decay1"), 8).alias("_p1"), ts_rank(pl.col("_decay2"), 13).alias("_p2")),
        -pairwise_max(pl.col("_p1"), pl.col("_p2")),
    )


def factor_gtja001(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(pl.when(pl.col("volume_share") > 0).then(pl.col("volume_share").log()).otherwise(None).alias("_log_vol"), ((pl.col("qfq_close") - pl.col("qfq_open")) / pl.col("qfq_open")).alias("_intraday"))
        .with_columns(delta(pl.col("_log_vol")).alias("_delta_log_vol"))
        .with_columns(rank(pl.col("_delta_log_vol")).alias("_rank_delta"), rank(pl.col("_intraday")).alias("_rank_intraday"))
        .with_columns(correlation(pl.col("_rank_delta"), pl.col("_rank_intraday"), 6).alias("_corr")),
        -pl.col("_corr"),
    )


def factor_gtja002(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(((pl.col("qfq_close") - pl.col("qfq_low") - (pl.col("qfq_high") - pl.col("qfq_close"))) / (pl.col("qfq_high") - pl.col("qfq_low"))).alias("_range_position"))
        .with_columns(delta(pl.col("_range_position")).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja003(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delay(pl.col("qfq_close")).alias("_previous_close"))
        .with_columns(pl.when(pl.col("qfq_close") > pl.col("_previous_close")).then(pairwise_min(pl.col("qfq_low"), pl.col("_previous_close"))).otherwise(pairwise_max(pl.col("qfq_high"), pl.col("_previous_close"))).alias("_boundary"))
        .with_columns(pl.when(pl.col("qfq_close") == pl.col("_previous_close")).then(0.0).otherwise(pl.col("qfq_close") - pl.col("_boundary")).alias("_move"))
        .with_columns(ts_sum(pl.col("_move"), 6).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja004(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(sma(pl.col("qfq_close"), 8).alias("_mean_8"), stddev(pl.col("qfq_close"), 8).alias("_std_8"), sma(pl.col("qfq_close"), 2).alias("_mean_2"), sma(pl.col("volume_share"), 20).alias("_adv20"))
        .with_columns(pl.when(pl.col("_mean_8") + pl.col("_std_8") < pl.col("_mean_2")).then(-1.0).when((pl.col("_mean_2") < pl.col("_mean_8") - pl.col("_std_8")) | (pl.col("volume_share") / pl.col("_adv20") >= 1.0)).then(1.0).otherwise(-1.0).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja005(frame: pl.LazyFrame) -> pl.LazyFrame:
    return factor_wq026(frame)


def factor_gtja006(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_open") * 0.85 + pl.col("qfq_high") * 0.15, 4).alias("_delta"))
        .with_columns(rank(pl.col("_delta").sign()).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja007(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_vwap") - pl.col("qfq_close")).alias("_vwap_close"), delta(pl.col("volume_share"), 3).alias("_vol_delta"))
        .with_columns(ts_max(pl.col("_vwap_close"), 3).alias("_max"), ts_min(pl.col("_vwap_close"), 3).alias("_min"))
        .with_columns(rank(pl.col("_max")).alias("_rank_max"), rank(pl.col("_min")).alias("_rank_min"), rank(pl.col("_vol_delta")).alias("_rank_vol_delta")),
        pl.col("_rank_max") + pl.col("_rank_min") * pl.col("_rank_vol_delta"),
    )


def factor_gtja008(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(((pl.col("qfq_high") + pl.col("qfq_low")) * 0.1 + pl.col("qfq_vwap") * 0.8).alias("_price"))
        .with_columns(delta(pl.col("_price"), 4).alias("_delta")),
        rank(-pl.col("_delta")),
    )


def factor_gtja009(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delay(pl.col("qfq_high")).alias("_high_delay"), delay(pl.col("qfq_low")).alias("_low_delay"))
        .with_columns((((pl.col("qfq_high") + pl.col("qfq_low")) / 2.0 - (pl.col("_high_delay") + pl.col("_low_delay")) / 2.0) * (pl.col("qfq_high") - pl.col("qfq_low")) / pl.col("volume_share")).alias("_signal"))
        .with_columns(ewm_sma(pl.col("_signal"), 7, 2).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja010(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(stddev(pl.col("_returns"), 20).alias("_std_returns"))
        .with_columns(pl.when(pl.col("_returns") < 0).then(pl.col("_std_returns")).otherwise(pl.col("qfq_close")).alias("_base"))
        .with_columns(ts_max(pl.col("_base").pow(2), 5).alias("_max")),
        rank(pl.col("_max")),
    )


def factor_gtja011(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame.with_columns(((pl.col("qfq_close") - pl.col("qfq_low") - (pl.col("qfq_high") - pl.col("qfq_close"))) / (pl.col("qfq_high") - pl.col("qfq_low")) * pl.col("volume_share")).alias("_signal")), ts_sum(pl.col("_signal"), 6))


def factor_gtja012(frame: pl.LazyFrame) -> pl.LazyFrame:
    return factor_wq005(frame)


def factor_gtja013(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame, (pl.col("qfq_high") * pl.col("qfq_low")).sqrt() - pl.col("qfq_vwap"))


def factor_gtja014(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame, delta(pl.col("qfq_close"), 5))


def factor_gtja015(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame, pl.col("qfq_open") / delay(pl.col("qfq_close")) - 1.0)


def factor_gtja016(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("volume_share")).alias("_rank_vol"), rank(pl.col("qfq_vwap")).alias("_rank_vwap"))
        .with_columns(correlation(pl.col("_rank_vol"), pl.col("_rank_vwap"), 5).alias("_corr"))
        .with_columns(rank(pl.col("_corr")).alias("_rank_corr"))
        .with_columns(ts_max(pl.col("_rank_corr"), 5).alias("_max")),
        -pl.col("_max"),
    )


def factor_gtja017(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(ts_max(pl.col("qfq_vwap"), 15).alias("_vwap_max"), delta(pl.col("qfq_close"), 5).alias("_close_delta"))
        .with_columns(rank(pl.col("qfq_vwap") - pl.col("_vwap_max")).alias("_rank")),
        pl.col("_rank").pow(pl.col("_close_delta")),
    )


def factor_gtja018(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame, pl.col("qfq_close") / delay(pl.col("qfq_close"), 5))


def factor_gtja019(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delay(pl.col("qfq_close"), 5).alias("_previous"))
        .with_columns(pl.when(pl.col("qfq_close") < pl.col("_previous")).then((pl.col("qfq_close") - pl.col("_previous")) / pl.col("_previous")).when(pl.col("qfq_close") == pl.col("_previous")).then(0.0).otherwise((pl.col("qfq_close") - pl.col("_previous")) / pl.col("qfq_close")).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja020(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame, delta(pl.col("qfq_close"), 6) / delay(pl.col("qfq_close"), 6) * 100.0)


def _linear_time_slope(frame: pl.LazyFrame, value: pl.Expr, window: int) -> pl.LazyFrame:
    return (
        frame.with_columns(value.alias("_trend_value"))
        .with_columns(pl.col("trade_date").cum_count().over("ts_code").cast(pl.Float64).alias("_trend_time"))
        .with_columns(covariance(pl.col("_trend_value"), pl.col("_trend_time"), window).alias("_trend_cov"), stddev(pl.col("_trend_time"), window).pow(2).alias("_trend_var"))
    )


def factor_gtja021(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(_linear_time_slope(frame, pl.col("qfq_close"), 6), pl.col("_trend_cov") / pl.col("_trend_var"))


def factor_gtja116(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(_linear_time_slope(frame, pl.col("qfq_close"), 20), pl.col("_trend_cov") / pl.col("_trend_var"))


def factor_gtja147(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(_linear_time_slope(frame, sma(pl.col("qfq_close"), 12), 12), pl.col("_trend_cov") / pl.col("_trend_var"))


def factor_gtja022(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("qfq_close"), 6).alias("_mean_6"))
        .with_columns(((pl.col("qfq_close") - pl.col("_mean_6")) / pl.col("_mean_6")).alias("_deviation"))
        .with_columns(delta(pl.col("_deviation"), 3).alias("_deviation_delta"))
        .with_columns((pl.col("_deviation") - pl.col("_deviation_delta")).alias("_signal"))
        .with_columns(ewm_sma(pl.col("_signal"), 12, 1).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja023(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delay(pl.col("qfq_close")).alias("_previous"), stddev(pl.col("qfq_close"), 20).alias("_std"))
        .with_columns(pl.when(pl.col("qfq_close") > pl.col("_previous")).then(pl.col("_std")).otherwise(0.0).alias("_up"), pl.when(pl.col("qfq_close") <= pl.col("_previous")).then(pl.col("_std")).otherwise(0.0).alias("_down"))
        .with_columns(ewm_sma(pl.col("_up"), 20, 1).alias("_up_sma"), ewm_sma(pl.col("_down"), 20, 1).alias("_down_sma")),
        pl.col("_up_sma") / (pl.col("_up_sma") + pl.col("_down_sma")) * 100.0,
    )


def factor_gtja024(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_close"), 5).alias("_delta"))
        .with_columns(ewm_sma(pl.col("_delta"), 5, 1).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja025(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delta(pl.col("qfq_close"), 7).alias("_close_delta"), sma(pl.col("volume_share"), 20).alias("_adv20"), (pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(decay_linear(pl.col("volume_share") / pl.col("_adv20"), 9).alias("_decay_vol"), ts_sum(pl.col("_returns"), 250).alias("_return_sum"))
        .with_columns(rank(pl.col("_decay_vol")).alias("_rank_decay_vol"), rank(pl.col("_return_sum")).alias("_rank_return_sum"))
        .with_columns(rank(pl.col("_close_delta") * (1.0 - pl.col("_rank_decay_vol"))).alias("_rank_signal")),
        -pl.col("_rank_signal") * (1.0 + pl.col("_rank_return_sum")),
    )


def factor_gtja026(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("qfq_close"), 7).alias("_mean_7"), delay(pl.col("qfq_close"), 5).alias("_close_delay_5"))
        .with_columns(correlation(pl.col("qfq_vwap"), pl.col("_close_delay_5"), 230).alias("_corr")),
        pl.col("_mean_7") - pl.col("qfq_close") + pl.col("_corr"),
    )


def factor_gtja027(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delay(pl.col("qfq_close"), 3).alias("_delay_3"), delay(pl.col("qfq_close"), 6).alias("_delay_6"))
        .with_columns((((pl.col("qfq_close") - pl.col("_delay_3")) / pl.col("_delay_3") + (pl.col("qfq_close") - pl.col("_delay_6")) / pl.col("_delay_6")) * 100.0).alias("_signal"))
        .with_columns(decay_linear(pl.col("_signal"), 12).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja028(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(ts_min(pl.col("qfq_low"), 9).alias("_low_min"), ts_max(pl.col("qfq_high"), 9).alias("_high_max"))
        .with_columns(((pl.col("qfq_close") - pl.col("_low_min")) / (pl.col("_high_max") - pl.col("_low_min")) * 100.0).alias("_signal"))
        .with_columns(ewm_sma(pl.col("_signal"), 3, 1).alias("_sma"))
        .with_columns(ewm_sma(pl.col("_sma"), 3, 1).alias("_sma_2")),
        3.0 * pl.col("_sma") - 2.0 * pl.col("_sma_2"),
    )


def factor_gtja029(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame, delta(pl.col("qfq_close"), 6) / delay(pl.col("qfq_close"), 6) * pl.col("volume_share"))


def factor_gtja031(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("qfq_close"), 12).alias("_mean_12")),
        (pl.col("qfq_close") - pl.col("_mean_12")) / pl.col("_mean_12") * 100.0,
    )


def factor_gtja032(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("qfq_high")).alias("_rank_high"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(correlation(pl.col("_rank_high"), pl.col("_rank_vol"), 3).alias("_corr"))
        .with_columns(rank(pl.col("_corr")).alias("_rank_corr"))
        .with_columns(ts_sum(pl.col("_rank_corr"), 3).alias("_sum")),
        -pl.col("_sum"),
    )


def factor_gtja033(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(ts_min(pl.col("qfq_low"), 5).alias("_low_min"), (pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(delay(pl.col("_low_min"), 5).alias("_low_min_delay"), ts_sum(pl.col("_returns"), 240).alias("_ret_sum_240"), ts_sum(pl.col("_returns"), 20).alias("_ret_sum_20"), ts_rank(pl.col("volume_share"), 5).alias("_ts_rank_vol"))
        .with_columns(rank((pl.col("_ret_sum_240") - pl.col("_ret_sum_20")) / 220.0).alias("_rank_returns")),
        (-pl.col("_low_min") + pl.col("_low_min_delay")) * pl.col("_rank_returns") * pl.col("_ts_rank_vol"),
    )


def factor_gtja034(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(frame.with_columns(sma(pl.col("qfq_close"), 12).alias("_mean_12")), pl.col("_mean_12") / pl.col("qfq_close"))


def factor_gtja035(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(delta(pl.col("qfq_open")).alias("_open_delta"))
        .with_columns(decay_linear(pl.col("_open_delta"), 15).alias("_open_decay"), correlation(pl.col("volume_share"), pl.col("qfq_open"), 17).alias("_corr"))
        .with_columns(decay_linear(pl.col("_corr"), 7).alias("_corr_decay"))
        .with_columns(rank(pl.col("_open_decay")).alias("_rank_open"), rank(pl.col("_corr_decay")).alias("_rank_corr")),
        -pairwise_min(pl.col("_rank_open"), pl.col("_rank_corr")),
    )


def factor_gtja036(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(rank(pl.col("volume_share")).alias("_rank_vol"), rank(pl.col("qfq_vwap")).alias("_rank_vwap"))
        .with_columns(correlation(pl.col("_rank_vol"), pl.col("_rank_vwap"), 6).alias("_corr"))
        .with_columns(ts_sum(pl.col("_corr"), 2).alias("_sum")),
        rank(pl.col("_sum")),
    )


def factor_gtja037(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns((pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0).alias("_returns"))
        .with_columns(ts_sum(pl.col("qfq_open"), 5).alias("_open_sum"), ts_sum(pl.col("_returns"), 5).alias("_return_sum"))
        .with_columns((pl.col("_open_sum") * pl.col("_return_sum")).alias("_product"))
        .with_columns(delay(pl.col("_product"), 10).alias("_product_delay")),
        -rank(pl.col("_product") - pl.col("_product_delay")),
    )


def factor_gtja038(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("qfq_high"), 20).alias("_mean_high"), delta(pl.col("qfq_high"), 2).alias("_delta"))
        .with_columns(pl.when(pl.col("_mean_high") < pl.col("qfq_high")).then(-pl.col("_delta")).otherwise(0.0).alias("factor_value")),
        pl.col("factor_value"),
    )


def factor_gtja039(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA39 with every nested rolling stage materialized explicitly.

    The source formula nests a 180-day average, 37-day sum, 14-day
    correlation and 12-day linear decay.  Leaving that as one expression
    makes Polars repeatedly expand the window graph; named columns turn it
    into a compact, reusable columnar plan.
    """
    return _finish(
        frame
        .with_columns(
            delta(pl.col("qfq_close"), 2).alias("_close_delta_2"),
            sma(pl.col("volume_share"), 180).alias("_adv180"),
        )
        .with_columns(
            decay_linear(pl.col("_close_delta_2"), 8).alias("_close_decay_8"),
            ts_sum(pl.col("_adv180"), 37).alias("_adv180_sum_37"),
        )
        .with_columns(
            correlation(
                pl.col("qfq_vwap") * 0.3 + pl.col("qfq_open") * 0.7,
                pl.col("_adv180_sum_37"),
                14,
            ).alias("_price_adv_corr_14"),
        )
        .with_columns(decay_linear(pl.col("_price_adv_corr_14"), 12).alias("_corr_decay_12"))
        .with_columns(
            rank(pl.col("_close_decay_8")).alias("_rank_close_decay"),
            rank(pl.col("_corr_decay_12")).alias("_rank_corr_decay"),
        ),
        pl.col("_rank_corr_decay") - pl.col("_rank_close_decay"),
    )


def factor_gtja040(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delay(pl.col("qfq_close")).alias("_previous"))
        .with_columns(pl.when(pl.col("qfq_close") > pl.col("_previous")).then(pl.col("volume_share")).otherwise(0.0).alias("_up"), pl.when(pl.col("qfq_close") <= pl.col("_previous")).then(pl.col("volume_share")).otherwise(0.0).alias("_down"))
        .with_columns(ts_sum(pl.col("_up"), 26).alias("_up_sum"), ts_sum(pl.col("_down"), 26).alias("_down_sum")),
        pl.col("_up_sum") / pl.col("_down_sum") * 100.0,
    )


def factor_gtja041(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_vwap"), 3).alias("_delta"))
        .with_columns(ts_max(pl.col("_delta"), 5).alias("_delta_max"))
        .with_columns(rank(pl.col("_delta_max")).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja042(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(stddev(pl.col("qfq_high"), 10).alias("_std"), correlation(pl.col("qfq_high"), pl.col("volume_share"), 10).alias("_corr"))
        .with_columns(rank(pl.col("_std")).alias("_rank_std")),
        -pl.col("_rank_std") * pl.col("_corr"),
    )


def factor_gtja043(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delay(pl.col("qfq_close")).alias("_previous"))
        .with_columns(pl.when(pl.col("qfq_close") > pl.col("_previous")).then(pl.col("volume_share")).when(pl.col("qfq_close") < pl.col("_previous")).then(-pl.col("volume_share")).otherwise(0.0).alias("_signed_vol")),
        ts_sum(pl.col("_signed_vol"), 6),
    )


def factor_gtja044(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA44's two decay/rank branches as a staged native-Polars plan."""
    return _finish(
        frame
        .with_columns(
            sma(pl.col("volume_share"), 10).alias("_adv10"),
            delta(pl.col("qfq_vwap"), 3).alias("_vwap_delta_3"),
        )
        .with_columns(
            correlation(pl.col("qfq_low"), pl.col("_adv10"), 7).alias("_low_adv_corr_7"),
            decay_linear(pl.col("_vwap_delta_3"), 10).alias("_vwap_decay_10"),
        )
        .with_columns(decay_linear(pl.col("_low_adv_corr_7"), 6).alias("_corr_decay_6"))
        .with_columns(
            (ts_rank(pl.col("_corr_decay_6"), 4) / 4.0).alias("_corr_ts_rank_4"),
            (ts_rank(pl.col("_vwap_decay_10"), 15) / 15.0).alias("_vwap_ts_rank_15"),
        ),
        pl.col("_corr_ts_rank_4") + pl.col("_vwap_ts_rank_15"),
    )


def factor_gtja045(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_close") * 0.6 + pl.col("qfq_open") * 0.4).alias("_delta"), sma(pl.col("volume_share"), 150).alias("_adv150"))
        .with_columns(correlation(pl.col("qfq_vwap"), pl.col("_adv150"), 15).alias("_corr"))
        .with_columns(rank(pl.col("_delta")).alias("_rank_delta"), rank(pl.col("_corr")).alias("_rank_corr")),
        pl.col("_rank_delta") * pl.col("_rank_corr"),
    )


def factor_gtja046(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(sma(pl.col("qfq_close"), 3).alias("_mean3"), sma(pl.col("qfq_close"), 6).alias("_mean6"), sma(pl.col("qfq_close"), 12).alias("_mean12"), sma(pl.col("qfq_close"), 24).alias("_mean24")),
        (pl.col("_mean3") + pl.col("_mean6") + pl.col("_mean12") + pl.col("_mean24")) / (4.0 * pl.col("qfq_close")),
    )


def factor_gtja047(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(ts_max(pl.col("qfq_high"), 6).alias("_high_max"), ts_min(pl.col("qfq_low"), 6).alias("_low_min"))
        .with_columns(((pl.col("_high_max") - pl.col("qfq_close")) / (pl.col("_high_max") - pl.col("_low_min")) * 100.0).alias("_signal")),
        ewm_sma(pl.col("_signal"), 9, 1),
    )


def factor_gtja048(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delta(pl.col("qfq_close")).alias("_delta"))
        .with_columns(delay(pl.col("_delta")).alias("_delta_delay1"), delay(pl.col("_delta"), 2).alias("_delta_delay2"))
        .with_columns(rank(pl.col("_delta").sign() + (-pl.col("_delta_delay1")).sign() + (-pl.col("_delta_delay2")).sign()).alias("_rank"))
        .with_columns(ts_sum(pl.col("volume_share"), 5).alias("_vol5"), ts_sum(pl.col("volume_share"), 20).alias("_vol20")),
        -pl.col("_rank") * pl.col("_vol5") / pl.col("_vol20"),
    )


def factor_gtja053(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(delay(pl.col("qfq_close")).alias("_previous"))
        .with_columns(pl.when(pl.col("qfq_close") > pl.col("_previous")).then(1.0).otherwise(0.0).alias("_up")),
        ts_sum(pl.col("_up"), 12) / 12.0 * 100.0,
    )


def factor_gtja054(frame: pl.LazyFrame) -> pl.LazyFrame:
    return factor_wq018(frame)


def factor_gtja057(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame.with_columns(ts_min(pl.col("qfq_low"), 9).alias("_low_min"), ts_max(pl.col("qfq_high"), 9).alias("_high_max"))
        .with_columns(((pl.col("qfq_close") - pl.col("_low_min")) / (pl.col("_high_max") - pl.col("_low_min")) * 100.0).alias("_signal")),
        ewm_sma(pl.col("_signal"), 3, 1),
    )


def _gtja_price_pressure(frame: pl.LazyFrame, variant: int) -> pl.LazyFrame:
    """Shared staged form for GTJA55 and GTJA137's range-pressure signal."""
    high_close = (pl.col("qfq_high") - delay(pl.col("qfq_close"))).abs()
    low_close = (pl.col("qfq_low") - delay(pl.col("qfq_close"))).abs()
    high_low = (pl.col("qfq_high") - delay(pl.col("qfq_low"))).abs()
    previous_gap = (delay(pl.col("qfq_close") - pl.col("qfq_open"))).abs()
    condition_1 = (high_close > low_close) & (high_close > high_low)
    condition_2 = (low_close > high_low) & (low_close > high_close)
    false_branch = high_low + previous_gap / 4.0
    if variant == 55:
        true_branch_2 = low_close + high_low / 2.0 + previous_gap / 4.0
    else:
        true_branch_2 = low_close + high_close / 2.0 + previous_gap / 4.0
    denominator = pl.when(condition_1).then(high_close + low_close / 2.0 + previous_gap / 4.0).when(condition_2).then(true_branch_2).otherwise(false_branch)
    return frame.with_columns(
        (16.0 * (delta(pl.col("qfq_close")) + (pl.col("qfq_close") - pl.col("qfq_open")) / 2.0 + delay(pl.col("qfq_close")) - delay(pl.col("qfq_open")))).alias("_numerator"),
        denominator.alias("_denominator"), pairwise_max(high_close, low_close).alias("_range"),
    )


def factor_gtja055(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        _gtja_price_pressure(frame, 55).with_columns((pl.col("_numerator") / pl.col("_denominator") * pl.col("_range")).alias("_signal")),
        ts_sum(pl.col("_signal"), 20),
    )


def factor_gtja056(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA56 with the rank/correlation cascade materialized by stage."""
    return _finish(
        frame
        .with_columns(
            ts_min(pl.col("qfq_open"), 12).alias("_open_min_12"),
            ts_sum((pl.col("qfq_high") + pl.col("qfq_low")) / 2.0, 19).alias("_mid_sum_19"),
            sma(pl.col("volume_share"), 40).alias("_adv40"),
        )
        .with_columns(
            rank(pl.col("qfq_open") - pl.col("_open_min_12")).alias("_rank_open_min"),
            ts_sum(pl.col("_adv40"), 19).alias("_adv40_sum_19"),
        )
        .with_columns(correlation(pl.col("_mid_sum_19"), pl.col("_adv40_sum_19"), 13).alias("_corr_13"))
        .with_columns(rank(pl.col("_corr_13")).alias("_rank_corr"))
        .with_columns(rank(pl.col("_rank_corr").pow(5)).alias("_rank_corr_pow_5")),
        (pl.col("_rank_open_min") < pl.col("_rank_corr_pow_5")).cast(pl.Float64),
    )


def factor_gtja061(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA61: staged decay branches avoid nested rolling-window expansion."""
    return _finish(
        frame
        .with_columns(
            delta(pl.col("qfq_vwap")).alias("_vwap_delta"),
            sma(pl.col("volume_share"), 80).alias("_adv80"),
        )
        .with_columns(
            decay_linear(pl.col("_vwap_delta"), 12).alias("_vwap_decay_12"),
            correlation(pl.col("qfq_low"), pl.col("_adv80"), 8).alias("_low_adv_corr_8"),
        )
        .with_columns(
            rank(pl.col("_vwap_decay_12")).alias("_rank_vwap_decay"),
            rank(pl.col("_low_adv_corr_8")).alias("_rank_low_adv_corr"),
        )
        .with_columns(decay_linear(pl.col("_rank_low_adv_corr"), 17).alias("_corr_rank_decay_17"))
        .with_columns(rank(pl.col("_corr_rank_decay_17")).alias("_rank_corr_decay")),
        -pairwise_max(pl.col("_rank_vwap_decay"), pl.col("_rank_corr_decay")),
    )


def factor_gtja064(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA64's dual correlation/decay branches as a compact Polars DAG."""
    return _finish(
        frame
        .with_columns(
            rank(pl.col("qfq_vwap")).alias("_rank_vwap"),
            rank(pl.col("volume_share")).alias("_rank_vol"),
            rank(pl.col("qfq_close")).alias("_rank_close"),
            sma(pl.col("volume_share"), 60).alias("_adv60"),
        )
        .with_columns(
            correlation(pl.col("_rank_vwap"), pl.col("_rank_vol"), 4).alias("_vwap_vol_corr_4"),
            rank(pl.col("_adv60")).alias("_rank_adv60"),
        )
        .with_columns(
            decay_linear(pl.col("_vwap_vol_corr_4"), 4).alias("_vwap_vol_decay_4"),
            correlation(pl.col("_rank_close"), pl.col("_rank_adv60"), 4).alias("_close_adv_corr_4"),
        )
        .with_columns(ts_max(pl.col("_close_adv_corr_4"), 13).alias("_close_adv_corr_max_13"))
        .with_columns(decay_linear(pl.col("_close_adv_corr_max_13"), 14).alias("_close_adv_decay_14"))
        .with_columns(
            rank(pl.col("_vwap_vol_decay_4")).alias("_rank_vwap_vol_decay"),
            rank(pl.col("_close_adv_decay_14")).alias("_rank_close_adv_decay"),
        ),
        -pairwise_max(pl.col("_rank_vwap_vol_decay"), pl.col("_rank_close_adv_decay")),
    )


def factor_gtja073(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA73 materializes its two nested decay chains before ranking."""
    return _finish(
        frame
        .with_columns(
            correlation(pl.col("qfq_close"), pl.col("volume_share"), 10).alias("_close_vol_corr_10"),
            sma(pl.col("volume_share"), 30).alias("_adv30"),
        )
        .with_columns(decay_linear(pl.col("_close_vol_corr_10"), 16).alias("_corr_decay_16"))
        .with_columns(decay_linear(pl.col("_corr_decay_16"), 4).alias("_corr_decay_4"), correlation(pl.col("qfq_vwap"), pl.col("_adv30"), 4).alias("_vwap_adv_corr_4"))
        .with_columns(decay_linear(pl.col("_vwap_adv_corr_4"), 3).alias("_vwap_adv_decay_3"))
        .with_columns(
            (ts_rank(pl.col("_corr_decay_4"), 5) / 5.0).alias("_corr_ts_rank_5"),
            rank(pl.col("_vwap_adv_decay_3")).alias("_rank_vwap_adv_decay"),
        ),
        pl.col("_rank_vwap_adv_decay") - pl.col("_corr_ts_rank_5"),
    )


def factor_gtja074(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA74's two correlation branches, staged before cross-sectional rank."""
    return _finish(
        frame
        .with_columns(
            ts_sum(pl.col("qfq_low") * 0.35 + pl.col("qfq_vwap") * 0.65, 20).alias("_price_sum_20"),
            sma(pl.col("volume_share"), 40).alias("_adv40"),
            rank(pl.col("qfq_vwap")).alias("_rank_vwap"),
            rank(pl.col("volume_share")).alias("_rank_vol"),
        )
        .with_columns(ts_sum(pl.col("_adv40"), 20).alias("_adv40_sum_20"))
        .with_columns(
            correlation(pl.col("_price_sum_20"), pl.col("_adv40_sum_20"), 7).alias("_price_adv_corr_7"),
            correlation(pl.col("_rank_vwap"), pl.col("_rank_vol"), 6).alias("_vwap_vol_corr_6"),
        )
        .with_columns(
            rank(pl.col("_price_adv_corr_7")).alias("_rank_price_adv_corr"),
            rank(pl.col("_vwap_vol_corr_6")).alias("_rank_vwap_vol_corr"),
        ),
        pl.col("_rank_price_adv_corr") + pl.col("_rank_vwap_vol_corr"),
    )


def factor_gtja077(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA77: native decay and correlation stages for both rank branches."""
    midpoint = (pl.col("qfq_high") + pl.col("qfq_low")) / 2.0
    return _finish(
        frame
        .with_columns(
            (midpoint - pl.col("qfq_vwap")).alias("_midpoint_minus_vwap"),
            sma(pl.col("volume_share"), 40).alias("_adv40"),
        )
        .with_columns(
            decay_linear(pl.col("_midpoint_minus_vwap"), 20).alias("_price_decay_20"),
            correlation(midpoint, pl.col("_adv40"), 3).alias("_midpoint_adv_corr_3"),
        )
        .with_columns(decay_linear(pl.col("_midpoint_adv_corr_3"), 6).alias("_corr_decay_6"))
        .with_columns(
            rank(pl.col("_price_decay_20")).alias("_rank_price_decay"),
            rank(pl.col("_corr_decay_6")).alias("_rank_corr_decay"),
        ),
        pairwise_min(pl.col("_rank_price_decay"), pl.col("_rank_corr_decay")),
    )


def factor_gtja083(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(rank(pl.col("qfq_high")).alias("_rank_high"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(covariance(pl.col("_rank_high"), pl.col("_rank_vol"), 5).alias("_covariance_5"))
        .with_columns(rank(pl.col("_covariance_5")).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja087(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA87's weighted price and normalized-price decay branches."""
    midpoint = (pl.col("qfq_high") + pl.col("qfq_low")) / 2.0
    return _finish(
        frame
        .with_columns(
            delta(pl.col("qfq_vwap"), 4).alias("_vwap_delta_4"),
            ((pl.col("qfq_low") - pl.col("qfq_vwap")) / (pl.col("qfq_open") - midpoint)).alias("_normalized_price"),
        )
        .with_columns(
            decay_linear(pl.col("_vwap_delta_4"), 7).alias("_vwap_decay_7"),
            decay_linear(pl.col("_normalized_price"), 11).alias("_price_decay_11"),
        )
        .with_columns(
            rank(pl.col("_vwap_decay_7")).alias("_rank_vwap_decay"),
            (ts_rank(pl.col("_price_decay_11"), 7) / 7.0).alias("_price_ts_rank_7"),
        ),
        -(pl.col("_rank_vwap_decay") + pl.col("_price_ts_rank_7")),
    )


def factor_gtja090(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(rank(pl.col("qfq_vwap")).alias("_rank_vwap"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(correlation(pl.col("_rank_vwap"), pl.col("_rank_vol"), 5).alias("_corr_5"))
        .with_columns(rank(pl.col("_corr_5")).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja091(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA91, preserving the reference's scalar max(close, 5) term."""
    return _finish(
        frame
        .with_columns(
            rank(pl.col("qfq_close") - pairwise_max(pl.col("qfq_close"), pl.lit(5.0))).alias("_rank_close_delta"),
            sma(pl.col("volume_share"), 40).alias("_adv40"),
        )
        .with_columns(correlation(pl.col("_adv40"), pl.col("qfq_low"), 5).alias("_adv_low_corr_5"))
        .with_columns(rank(pl.col("_adv_low_corr_5")).alias("_rank_adv_low_corr")),
        -pl.col("_rank_close_delta") * pl.col("_rank_adv_low_corr"),
    )


def factor_gtja092(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA92, materializing both the 3-day and 180-day decay branches."""
    composite = pl.col("qfq_close") * 0.35 + pl.col("qfq_vwap") * 0.65
    return _finish(
        frame
        .with_columns(
            delta(composite, 2).alias("_composite_delta_2"),
            sma(pl.col("volume_share"), 180).alias("_adv180"),
        )
        .with_columns(
            decay_linear(pl.col("_composite_delta_2"), 3).alias("_composite_decay_3"),
            correlation(pl.col("_adv180"), pl.col("qfq_close"), 13).alias("_adv_close_corr_13"),
        )
        .with_columns(decay_linear(pl.col("_adv_close_corr_13").abs(), 5).alias("_abs_corr_decay_5"))
        .with_columns(
            rank(pl.col("_composite_decay_3")).alias("_rank_composite_decay"),
            (ts_rank(pl.col("_abs_corr_decay_5"), 15) / 15.0).alias("_corr_ts_rank_15"),
        ),
        -pairwise_max(pl.col("_rank_composite_decay"), pl.col("_corr_ts_rank_15")),
    )


def factor_gtja098(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA98's 100-day average branch, without re-expanding it per condition."""
    return _finish(
        frame
        .with_columns(
            sma(pl.col("qfq_close"), 100).alias("_mean_close_100"),
            ts_min(pl.col("qfq_close"), 100).alias("_min_close_100"),
            delay(pl.col("qfq_close"), 3).alias("_close_delay_3"),
        )
        .with_columns(
            delay(pl.col("_mean_close_100"), 100).alias("_mean_close_100_delay_100"),
            delay(pl.col("qfq_close"), 100).alias("_close_delay_100"),
        )
        .with_columns(
            ((pl.col("_mean_close_100") - pl.col("_mean_close_100_delay_100")) / pl.col("_close_delay_100")).alias("_mean_change"),
        ),
        pl.when(pl.col("_mean_change") <= 0.05)
        .then(-(pl.col("qfq_close") - pl.col("_min_close_100")))
        .otherwise(-(pl.col("qfq_close") - pl.col("_close_delay_3"))),
    )


def factor_gtja099(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(rank(pl.col("qfq_close")).alias("_rank_close"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(covariance(pl.col("_rank_close"), pl.col("_rank_vol"), 5).alias("_covariance_5"))
        .with_columns(rank(pl.col("_covariance_5")).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja101(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA101's two correlation/rank branches, evaluated once each."""
    return _finish(
        frame
        .with_columns(
            sma(pl.col("volume_share"), 30).alias("_adv30"),
            rank(pl.col("qfq_high") * 0.1 + pl.col("qfq_vwap") * 0.9).alias("_rank_high_vwap"),
            rank(pl.col("volume_share")).alias("_rank_vol"),
        )
        .with_columns(ts_sum(pl.col("_adv30"), 37).alias("_adv30_sum_37"))
        .with_columns(
            correlation(pl.col("qfq_close"), pl.col("_adv30_sum_37"), 15).alias("_close_adv_corr_15"),
            correlation(pl.col("_rank_high_vwap"), pl.col("_rank_vol"), 11).alias("_ranked_corr_11"),
        )
        .with_columns(
            rank(pl.col("_close_adv_corr_15")).alias("_rank_close_adv_corr"),
            rank(pl.col("_ranked_corr_11")).alias("_rank_ranked_corr"),
        ),
        -(pl.col("_rank_close_adv_corr") < pl.col("_rank_ranked_corr")).cast(pl.Float64),
    )


def factor_gtja104(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(correlation(pl.col("qfq_high"), pl.col("volume_share"), 5).alias("_high_vol_corr_5"), stddev(pl.col("qfq_close"), 20).alias("_close_std_20"))
        .with_columns(delay(pl.col("_high_vol_corr_5"), 4).alias("_high_vol_corr_delay_4"), rank(pl.col("_close_std_20")).alias("_rank_close_std")),
        -(pl.col("_high_vol_corr_5") - pl.col("_high_vol_corr_delay_4")) * pl.col("_rank_close_std"),
    )


def factor_gtja105(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(rank(pl.col("qfq_open")).alias("_rank_open"), rank(pl.col("volume_share")).alias("_rank_vol"))
        .with_columns(correlation(pl.col("_rank_open"), pl.col("_rank_vol"), 10).alias("factor_value")),
        -pl.col("factor_value"),
    )


def factor_gtja107(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        frame
        .with_columns(
            delay(pl.col("qfq_high")).alias("_high_delay"),
            delay(pl.col("qfq_close")).alias("_close_delay"),
            delay(pl.col("qfq_low")).alias("_low_delay"),
        )
        .with_columns(
            rank(pl.col("qfq_open") - pl.col("_high_delay")).alias("_rank_open_high"),
            rank(pl.col("qfq_open") - pl.col("_close_delay")).alias("_rank_open_close"),
            rank(pl.col("qfq_open") - pl.col("_low_delay")).alias("_rank_open_low"),
        ),
        -pl.col("_rank_open_high") * pl.col("_rank_open_close") * pl.col("_rank_open_low"),
    )


def factor_gtja149(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA149 beta on down-index sessions; null samples are intentionally skipped."""
    return_series = pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0
    index_return = pl.col("index_close") / delay(pl.col("index_close")) - 1.0
    condition = pl.col("index_close") < delay(pl.col("index_close"))
    stock_down = pl.when(condition).then(return_series).otherwise(None)
    index_down = pl.when(condition).then(index_return).otherwise(None)
    covariance_down = _over_security(pl.rolling_cov(stock_down, index_down, window_size=252, min_samples=2))
    index_std_down = _over_security(index_down.rolling_std(window_size=252, min_samples=2))
    return _finish(frame, covariance_down / index_std_down.pow(2))


def factor_gtja137(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _finish(
        _gtja_price_pressure(frame, 137).with_columns((pl.col("_numerator") / pl.col("_denominator") * pl.col("_range")).alias("factor_value")),
        pl.col("factor_value"),
    )


@numba.njit(cache=True)
def _gtja143_kernel(close: np.ndarray) -> np.ndarray:
    result = np.full(close.size, np.nan)
    previous = np.nan
    state = 1.0
    for index in range(close.size):
        current = close[index]
        if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
            previous = current
            continue
        ratio = current / previous
        state = (ratio - 1.0) * state if ratio > 1.0 else state
        result[index] = state
        previous = current
    return result


def factor_gtja143(frame: pl.LazyFrame) -> pl.LazyFrame:
    """GTJA143's stateful accumulate recurrence, with a Numba-only kernel."""
    grouped = frame.group_by("ts_code", maintain_order=True).agg(
        pl.col("trade_date"),
        pl.map_groups(
            [pl.col("qfq_close")],
            lambda series: pl.Series(_gtja143_kernel(series[0].to_numpy())),
            return_dtype=pl.Float64,
            returns_scalar=False,
        ).alias("factor_value"),
    ).explode(["trade_date", "factor_value"])
    return grouped.select(*KEYS, "factor_value").filter(pl.col("factor_value").is_finite())


@lru_cache(maxsize=191)
def _gtja_body(number: int) -> str:
    """Return one trusted DolphinDB function body from the bundled source."""
    source = GTJA_REFERENCE.read_text(encoding="utf-8")
    match = re.search(rf"def gtjaAlpha{number}\([^)]*\)\s*\{{", source)
    if match is None:
        raise ValueError(f"GTJA source does not define alpha {number}")
    depth, cursor = 1, match.end()
    while cursor < len(source) and depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    if depth:
        raise ValueError(f"GTJA alpha {number} has unbalanced braces")
    return source[match.end():cursor - 1]


def _gtja_weights(stop: int, multiplier: float = 1.0) -> tuple[float, ...]:
    return tuple((index + 1) * multiplier for index in range(stop))


def _gtja_mavg(expression: pl.Expr, window_or_weights: int | tuple[float, ...], window: int | None = None) -> pl.Expr:
    if isinstance(window_or_weights, tuple):
        if window is not None and len(window_or_weights) != window:
            raise ValueError("GTJA weighted-average window does not match its weight vector")
        return weighted_average(expression, window_or_weights)
    return sma(expression, int(window_or_weights))


def _gtja_mrank(expression: pl.Expr, percent: bool, window: int) -> pl.Expr:
    value = ts_rank(expression, window)
    return value / window if percent else value


def _gtja_mcount(expression: pl.Expr, window: int) -> pl.Expr:
    return _over_security(expression.is_not_null().cast(pl.Float64).rolling_sum(window_size=window, min_samples=window))


def _gtja_mbeta(left: pl.Expr, right: pl.Expr, window: int) -> pl.Expr:
    return covariance(left, right, window) / stddev(right, window).pow(2)


def _gtja_iif(condition: pl.Expr, when_true: pl.Expr | float | int, when_false: pl.Expr | float | int) -> pl.Expr:
    return pl.when(condition.cast(pl.Boolean)).then(when_true).otherwise(when_false)


def _gtja_expression(source: str) -> str:
    """Translate only the documented DolphinDB vector-expression subset."""
    output = source.strip().rstrip(";").replace("\\", "/")
    output = re.sub(r"\btrue\b", "True", output, flags=re.IGNORECASE)
    output = re.sub(r"\bfalse\b", "False", output, flags=re.IGNORECASE)
    output = re.sub(r"\bNULL\b", "None", output)
    output = output.replace("&&", "&").replace("||", "|")
    output = re.sub(r"1\.\.(\d+)\s*\*\s*([0-9.]+)", r"_gtja_weights(\1, \2)", output)
    output = re.sub(r"1\.\.(\d+)", r"_gtja_weights(\1)", output)
    output = re.sub(r"\bmove\(", "delay(", output)
    output = re.sub(r"\bmfirst\(", "_gtja_mfirst(", output)
    output = re.sub(r"\bmavg\(", "_gtja_mavg(", output)
    output = re.sub(r"\bmsum\(", "ts_sum(", output)
    output = re.sub(r"\bmstd\(", "stddev(", output)
    output = re.sub(r"\bmmax\(", "ts_max(", output)
    output = re.sub(r"\bmmin\(", "ts_min(", output)
    output = re.sub(r"\bmcorr\(", "correlation(", output)
    output = re.sub(r"\bmcovar\(", "covariance(", output)
    output = re.sub(r"\bmbeta\(", "_gtja_mbeta(", output)
    output = re.sub(r"\bmrank\(", "_gtja_mrank(", output)
    output = re.sub(r"\bmcount\(", "_gtja_mcount(", output)
    output = re.sub(r"\bmimax\(", "ts_argmax(", output)
    output = re.sub(r"\bmimin\(", "ts_argmin(", output)
    output = re.sub(r"\browRank\(", "rank(", output)
    output = re.sub(r"\biif\(", "_gtja_iif(", output)
    output = re.sub(r"\bratios\(", "_gtja_ratios(", output)
    output = re.sub(r"\bewmMean\(", "_gtja_ewm(", output)
    output = re.sub(r"\bpow\(", "_gtja_pow(", output)
    output = re.sub(r"\bsign\(", "_gtja_sign(", output)
    output = re.sub(r"\blog\(", "_gtja_log(", output)
    return output


@dataclass
class _GTJAProgram:
    frame: pl.LazyFrame
    stage_number: int = 0

    def stage(self, value: object) -> object:
        if not isinstance(value, pl.Expr):
            return value
        name = f"_gtja_stage_{self.stage_number}"
        self.stage_number += 1
        self.frame = self.frame.with_columns(value.alias(name))
        return pl.col(name)

    def bind(self, name: str, value: object) -> object:
        if not isinstance(value, pl.Expr):
            value = pl.lit(value)
        self.frame = self.frame.with_columns(value.alias(name))
        return pl.col(name)


def _gtja_expr(node: ast.AST, environment: dict[str, object], program: _GTJAProgram) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _gtja_expr(node.operand, environment, program)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _gtja_expr(node.left, environment, program), _gtja_expr(node.right, environment, program)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow): return (left if isinstance(left, pl.Expr) else pl.lit(left)).pow(right)
        if isinstance(node.op, ast.BitAnd): return left & right
        if isinstance(node.op, ast.BitOr): return left | right
    if isinstance(node, ast.Compare):
        left, right = _gtja_expr(node.left, environment, program), _gtja_expr(node.comparators[0], environment, program)
        operations = {ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right, ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right, ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right}
        return operations[type(node.ops[0])]()
    if isinstance(node, ast.BoolOp):
        values = [_gtja_expr(value, environment, program) for value in node.values]
        return values[0] | values[1] if isinstance(node.op, ast.Or) else values[0] & values[1]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        arguments = [_gtja_expr(argument, environment, program) for argument in node.args]
        keywords = {
            keyword.arg: _gtja_expr(keyword.value, environment, program)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        functions: dict[str, Callable[..., object]] = {
            "rank": lambda value, percent=True: rank(value) if percent else value.rank(method="min").over("trade_date"),
            "delay": delay, "delta": delta, "ts_sum": ts_sum, "sma": sma, "stddev": stddev,
            "correlation": correlation, "covariance": covariance, "ts_min": ts_min, "ts_max": ts_max,
            "ts_rank": ts_rank, "ts_argmax": ts_argmax, "ts_argmin": ts_argmin,
            "min": pairwise_min, "max": pairwise_max, "abs": lambda value: value.abs(),
            "_gtja_iif": _gtja_iif, "_gtja_mfirst": lambda value, count: delay(value, count - 1),
            "_gtja_mavg": _gtja_mavg, "_gtja_mrank": _gtja_mrank, "_gtja_mcount": _gtja_mcount,
            "_gtja_mbeta": _gtja_mbeta, "_gtja_weights": _gtja_weights,
            "_gtja_ratios": lambda value: value / delay(value),
            "_gtja_ewm": lambda value, alpha: _over_security(value.ewm_mean(alpha=alpha, adjust=False, min_samples=1)),
            "_gtja_pow": lambda left, right: (left if isinstance(left, pl.Expr) else pl.lit(left)).pow(right),
            "_gtja_sign": lambda value: value.sign(), "_gtja_log": lambda value: value.log(),
            "rowMax": lambda value: value.max().over("trade_date"), "rowMin": lambda value: value.min().over("trade_date"),
        }
        if name not in functions:
            raise ValueError(f"unsupported GTJA function {name}")
        return program.stage(functions[name](*arguments, **keywords))
    raise ValueError(f"unsupported GTJA syntax {type(node).__name__}")


def _generic_gtja(number: int, frame: pl.LazyFrame) -> pl.LazyFrame:
    """Compile a straight-line GTJA formula into a staged native-Polars DAG."""
    environment: dict[str, object] = {
        "rank": lambda value, percent=True: rank(value) if percent else value.rank(method="min").over("trade_date"),
        "delay": delay, "delta": delta, "ts_sum": ts_sum,
        "sma": sma, "stddev": stddev, "correlation": correlation,
        "covariance": covariance, "ts_min": ts_min, "ts_max": ts_max,
        "ts_rank": ts_rank, "ts_argmax": ts_argmax, "ts_argmin": ts_argmin,
        "min": pairwise_min, "max": pairwise_max, "abs": lambda value: value.abs(),
        "_gtja_iif": _gtja_iif, "_gtja_mfirst": lambda value, count: delay(value, count - 1),
        "_gtja_mavg": _gtja_mavg, "_gtja_mrank": _gtja_mrank,
        "_gtja_mcount": _gtja_mcount, "_gtja_mbeta": _gtja_mbeta,
        "_gtja_weights": _gtja_weights, "_gtja_ratios": lambda value: value / delay(value),
        "_gtja_ewm": lambda value, alpha: _over_security(value.ewm_mean(alpha=alpha, adjust=False, min_samples=1)),
        "_gtja_pow": lambda left, right: (left if isinstance(left, pl.Expr) else pl.lit(left)).pow(right),
        "_gtja_sign": lambda value: value.sign(), "_gtja_log": lambda value: value.log(),
        "rowMax": lambda value: value.max().over("trade_date"),
        "rowMin": lambda value: value.min().over("trade_date"),
        "open": pl.col("qfq_open"), "high": pl.col("qfq_high"), "low": pl.col("qfq_low"),
        "close": pl.col("qfq_close"), "vol": pl.col("volume_share"), "vwap": pl.col("qfq_vwap"),
        "index_open": pl.col("index_open"), "index_close": pl.col("index_close"),
    }
    program = _GTJAProgram(frame)
    lines = [line.split("//", 1)[0].strip().rstrip(";") for line in _gtja_body(number).splitlines()]
    for line in filter(None, lines):
        if line.startswith("return "):
            expression = _gtja_expr(ast.parse(_gtja_expression(line[7:]), mode="eval").body, environment, program)
            return _finish(program.frame, expression if isinstance(expression, pl.Expr) else pl.lit(expression))
        assignment = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", line)
        if assignment is None:
            raise ValueError(f"GTJA alpha {number} unsupported statement: {line}")
        name = assignment.group(1)
        value = _gtja_expr(ast.parse(_gtja_expression(assignment.group(2)), mode="eval").body, environment, program)
        environment[name] = program.bind(name, value)
    raise ValueError(f"GTJA alpha {number} has no return statement")


@lru_cache(maxsize=101)
def _wq_function(number: int) -> ast.FunctionDef:
    module = ast.parse(WQ_REFERENCE.read_text(encoding="utf-8"))
    target = f"alpha{number:03d}"
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == target:
            return node
    raise ValueError(f"Alpha101 source does not define alpha {number}")


@dataclass
class _WQProgram:
    frame: pl.LazyFrame
    stage_number: int = 0

    def stage(self, value: object) -> object:
        if not isinstance(value, pl.Expr):
            return value
        name = f"_wq_stage_{self.stage_number}"
        self.stage_number += 1
        self.frame = self.frame.with_columns(value.alias(name))
        return pl.col(name)

    def bind(self, name: str, value: object) -> object:
        if not isinstance(value, pl.Expr):
            value = pl.lit(value)
        self.frame = self.frame.with_columns(value.alias(name))
        return pl.col(name)


def _wq_expr(node: ast.AST, environment: dict[str, object], program: _WQProgram) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return environment[node.attr]
        value = _wq_expr(node.value, environment, program)
        # Pandas code occasionally calls ``.to_frame().CLOSE``; a Polars
        # expression is already a single series, so the accessor is identity.
        if node.attr in {"CLOSE", "close"}:
            return value
        raise ValueError(f"unsupported Alpha101 attribute {node.attr}")
    if isinstance(node, ast.UnaryOp):
        value = _wq_expr(node.operand, environment, program)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _wq_expr(node.left, environment, program), _wq_expr(node.right, environment, program)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow): return (left if isinstance(left, pl.Expr) else pl.lit(left)).pow(right)
        raise ValueError(f"unsupported Alpha101 binary operator {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left, right = _wq_expr(node.left, environment, program), _wq_expr(node.comparators[0], environment, program)
        operator = node.ops[0]
        if isinstance(operator, ast.Lt): return left < right
        if isinstance(operator, ast.LtE): return left <= right
        if isinstance(operator, ast.Gt): return left > right
        if isinstance(operator, ast.GtE): return left >= right
        if isinstance(operator, ast.Eq): return left == right
        if isinstance(operator, ast.NotEq): return left != right
    if isinstance(node, ast.BoolOp):
        values = [_wq_expr(value, environment, program) for value in node.values]
        return values[0] | values[1] if isinstance(node.op, ast.Or) else values[0] & values[1]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            receiver = _wq_expr(node.func.value, environment, program)
            name = node.func.attr
            if name == "replace": return program.stage(receiver.fill_nan(None))
            if name in {"fillna", "fill_null"}:
                value = _wq_expr(node.args[0], environment, program) if node.args else _wq_expr(next(keyword.value for keyword in node.keywords if keyword.arg == "value"), environment, program)
                return program.stage(receiver.fill_nan(value).fill_null(value))
            if name == "to_frame": return program.stage(receiver)
            if name == "abs": return program.stage(receiver.abs())
            if name == "pow": return program.stage(receiver.pow(_wq_expr(node.args[0], environment, program)))
            if name == "copy": return receiver
            raise ValueError(f"unsupported Alpha101 method {name}")
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsupported Alpha101 call target")
        name = node.func.id
        arguments = [_wq_expr(argument, environment, program) for argument in node.args]
        functions: dict[str, Callable[..., object]] = {
            "rank": rank, "scale": scale, "delay": delay, "delta": delta,
            "ts_sum": ts_sum, "sma": sma, "stddev": stddev, "correlation": correlation,
            "covariance": covariance, "ts_min": ts_min, "ts_max": ts_max,
            "ts_rank": ts_rank, "ts_argmax": ts_argmax, "ts_argmin": ts_argmin,
            "decay_linear": decay_linear, "product": product, "abs": lambda value: value.abs(),
            "sign": lambda value: value.sign(), "log": lambda value: value.log(),
            "min": pairwise_min, "max": pairwise_max, "pow": lambda left, right: left.pow(right),
        }
        if name not in functions:
            raise ValueError(f"unsupported Alpha101 function {name}")
        return program.stage(functions[name](*arguments))
    raise ValueError(f"unsupported Alpha101 syntax {type(node).__name__}")


def _generic_wq(number: int, frame: pl.LazyFrame) -> pl.LazyFrame:
    environment: dict[str, object] = {
        "open": pl.col("qfq_open"), "high": pl.col("qfq_high"), "low": pl.col("qfq_low"),
        "close": pl.col("qfq_close"), "volume": pl.col("volume_share"), "vwap": pl.col("qfq_vwap"),
        "returns": pl.col("qfq_close") / delay(pl.col("qfq_close")) - 1.0,
    }
    program = _WQProgram(frame)
    for statement in _wq_function(number).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            value = _wq_expr(statement.value, environment, program)
            name = statement.targets[0].id
            environment[name] = program.bind(name, value)
            continue
        if isinstance(statement, ast.Return):
            value = _wq_expr(statement.value, environment, program)
            return _finish(program.frame, value if isinstance(value, pl.Expr) else pl.lit(value))
        raise ValueError(f"Alpha101 {number} uses unsupported statement {type(statement).__name__}")
    raise ValueError(f"Alpha101 {number} has no return statement")


BUILDERS: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    "wq_alpha001_qfq_v1": factor_wq001,
    "wq_alpha002_qfq_v1": factor_wq002,
    "wq_alpha003_qfq_v1": factor_wq003,
    "wq_alpha004_qfq_v1": factor_wq004,
    "wq_alpha005_qfq_v1": factor_wq005,
    "wq_alpha006_qfq_v1": factor_wq006,
    "wq_alpha007_qfq_v1": factor_wq007,
    "wq_alpha008_qfq_v1": factor_wq008,
    "wq_alpha009_qfq_v1": factor_wq009,
    "wq_alpha010_qfq_v1": factor_wq010,
    "wq_alpha011_qfq_v1": factor_wq011,
    "wq_alpha012_qfq_v1": factor_wq012,
    "wq_alpha013_qfq_v1": factor_wq013,
    "wq_alpha014_qfq_v1": factor_wq014,
    "wq_alpha015_qfq_v1": factor_wq015,
    "wq_alpha016_qfq_v1": factor_wq016,
    "wq_alpha017_qfq_v1": factor_wq017,
    "wq_alpha018_qfq_v1": factor_wq018,
    "wq_alpha019_qfq_v1": factor_wq019,
    "wq_alpha020_qfq_v1": factor_wq020,
    "wq_alpha021_qfq_v1": factor_wq021,
    "wq_alpha022_qfq_v1": factor_wq022,
    "wq_alpha023_qfq_v1": factor_wq023,
    "wq_alpha024_qfq_v1": factor_wq024,
    "wq_alpha025_qfq_v1": factor_wq025,
    "wq_alpha026_qfq_v1": factor_wq026,
    "wq_alpha027_qfq_v1": factor_wq027,
    "wq_alpha028_qfq_v1": factor_wq028,
    "wq_alpha036_qfq_v1": factor_wq036,
    "wq_alpha037_qfq_v1": factor_wq037,
    "wq_alpha038_qfq_v1": factor_wq038,
    "wq_alpha039_qfq_v1": factor_wq039,
    "wq_alpha040_qfq_v1": factor_wq040,
    "wq_alpha045_qfq_v1": factor_wq045,
    "wq_alpha046_qfq_v1": factor_wq046,
    "wq_alpha047_qfq_v1": factor_wq047,
    "wq_alpha049_qfq_v1": factor_wq049,
    "wq_alpha051_qfq_v1": factor_wq051,
    "wq_alpha071_qfq_v1": factor_wq071,
    "wq_alpha073_qfq_v1": factor_wq073,
    "wq_alpha077_qfq_v1": factor_wq077,
    "wq_alpha088_qfq_v1": factor_wq088,
    "wq_alpha092_qfq_v1": factor_wq092,
    "wq_alpha096_qfq_v1": factor_wq096,
    **{f"gtja_alpha{number:03d}_qfq_v1": builder for number, builder in {
        1: factor_gtja001, 2: factor_gtja002, 3: factor_gtja003, 4: factor_gtja004,
        5: factor_gtja005, 6: factor_gtja006, 7: factor_gtja007, 8: factor_gtja008,
        9: factor_gtja009, 10: factor_gtja010, 11: factor_gtja011, 12: factor_gtja012,
        13: factor_gtja013, 14: factor_gtja014, 15: factor_gtja015, 16: factor_gtja016,
        17: factor_gtja017, 18: factor_gtja018, 19: factor_gtja019, 20: factor_gtja020,
        21: factor_gtja021, 22: factor_gtja022, 23: factor_gtja023, 24: factor_gtja024,
        25: factor_gtja025, 26: factor_gtja026, 27: factor_gtja027, 28: factor_gtja028,
        29: factor_gtja029,
        31: factor_gtja031, 32: factor_gtja032, 33: factor_gtja033, 34: factor_gtja034,
        35: factor_gtja035, 36: factor_gtja036, 37: factor_gtja037, 38: factor_gtja038,
        39: factor_gtja039,
        40: factor_gtja040, 41: factor_gtja041, 42: factor_gtja042, 43: factor_gtja043,
        44: factor_gtja044, 45: factor_gtja045, 46: factor_gtja046, 47: factor_gtja047, 48: factor_gtja048,
        53: factor_gtja053, 54: factor_gtja054, 55: factor_gtja055, 56: factor_gtja056, 57: factor_gtja057,
        61: factor_gtja061, 64: factor_gtja064, 73: factor_gtja073, 74: factor_gtja074,
        77: factor_gtja077,
        83: factor_gtja083, 87: factor_gtja087, 90: factor_gtja090, 91: factor_gtja091, 92: factor_gtja092,
        98: factor_gtja098, 99: factor_gtja099,
        101: factor_gtja101, 104: factor_gtja104, 105: factor_gtja105, 107: factor_gtja107,
        149: factor_gtja149,
        116: factor_gtja116, 137: factor_gtja137, 143: factor_gtja143, 147: factor_gtja147,
    }.items()},
}

# Most GTJA source functions are straight-line compositions of the whitelisted
# operators above.  Dedicated builders win for formulas whose nested windows or
# semantics need explicit materialization; the remaining functions use the
# same Polars long-panel execution path through this compiler.
for _number in range(1, 192):
    if _number != 30:
        BUILDERS.setdefault(
            f"gtja_alpha{_number:03d}_qfq_v1",
            lambda frame, number=_number: _generic_gtja(number, frame),
        )

for _number in range(1, 102):
    BUILDERS.setdefault(
        f"wq_alpha{_number:03d}_qfq_v1",
        lambda frame, number=_number: _generic_wq(number, frame),
    )


def build_factor(
    factor_id: str,
    catalog: Path,
    start: str | None = None,
    end: str | None = None,
    lookback_sessions: int = 0,
) -> pl.DataFrame:
    """Execute one registered Polars factor and return canonical long rows."""
    output = evaluate_factor(factor_id, load_calendar_panel(catalog, start, end, lookback_sessions))
    if start:
        output = output.filter(pl.col("trade_date") >= pl.lit(start).str.to_date())
    output = output.collect()
    if output.is_empty():
        raise RuntimeError(f"{factor_id} produced no finite rows in the requested range")
    return output.sort(KEYS)


def evaluate_factor(factor_id: str, frame: pl.LazyFrame) -> pl.LazyFrame:
    """Apply one factor builder to an already-loaded calendar panel."""
    try:
        builder = BUILDERS[factor_id]
    except KeyError as exc:
        raise ValueError(f"{factor_id} has not yet been ported to the Polars engine") from exc
    return builder(frame)
