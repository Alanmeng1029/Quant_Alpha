"""Daily factor registry and builders backed by the canonical DuckDB catalog.

The first production tranche ports WQ Alpha001-020 and GTJA Alpha001-020.
The evaluator deliberately uses a calendar-aligned wide panel: an absent stock
observation remains a missing value rather than shortening a rolling window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata


DEFAULT_DATABASE_DIR = "A_stock_database"
KEYS = ("trade_date", "ts_code")
FACTOR_STORAGE_UNIVERSE = "CSI300 union CSI500 daily constituents"
WQ_REFERENCE = "/Users/alanmxy/大学/大学/alpha101_adjusted.py"
GTJA_REFERENCE = "/Users/alanmxy/大学/大学/gtja191Alpha.dos"
WQ_AVAILABLE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 60,
                61, 62, 64, 65, 66, 68, 71, 72, 73, 74, 75, 77, 78, 81, 83, 84, 85, 86,
                88, 92, 94, 95, 96, 98, 99, 101)
POLARS_WQ_IMPLEMENTED = frozenset(WQ_AVAILABLE)
# All entries below compile and execute against the Polars long-panel fixture.
# Alpha030 additionally requires external Fama-French inputs.
POLARS_GTJA_IMPLEMENTED = frozenset(range(1, 192)) - {30}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _write_atomic(frame: pl.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, output)


def _storage_universe(data_root: Path) -> pl.DataFrame:
    """Daily de-duplicated CSI300 union CSI500 membership for factor storage."""
    cache = data_root / "lake" / "derived" / "factor_research" / "csi300_csi500_daily_members.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    reference = data_root / "lake" / "canonical" / "reference"
    tradable = pl.scan_parquet(str(reference / "trading_universe/year=*/universe.parquet")).select("trade_date", "ts_code")
    constituents = pl.scan_parquet(str(reference / "index_constituents/year=*/constituents.parquet")).filter(
        pl.col("index_code").is_in(["000300.SH", "000905.SH"])
    ).select("ts_code", "as_of_date").unique()
    members = tradable.sort(["ts_code", "trade_date"]).join_asof(
        constituents.sort(["ts_code", "as_of_date"]), left_on="trade_date", right_on="as_of_date", by="ts_code", strategy="backward"
    ).filter(pl.col("as_of_date").is_not_null()).select("trade_date", "ts_code").unique().sort(KEYS).collect()
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(members, cache)
    return members


def _restrict_storage_universe(frame: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Trim only the artifact, never the source panel used to calculate it."""
    return frame.join(members, on=list(KEYS), how="inner").sort(KEYS)


def compact_storage_universe(data_root: Path, family: str | None = None, dry_run: bool = False) -> list[dict[str, object]]:
    """Rewrite existing artifacts to the approved storage universe without recomputing values."""
    members = _storage_universe(data_root)
    results: list[dict[str, object]] = []
    for definition in sorted(REGISTRY.values(), key=lambda item: item.factor_id):
        if family and definition.family != family:
            continue
        output_dir = factor_directory(data_root, definition); output = output_dir / "factor.parquet"
        if not output.exists():
            continue
        source = pl.read_parquet(output)
        trimmed = _restrict_storage_universe(source, members)
        result = {"factor_id": definition.factor_id, "before_rows": source.height, "after_rows": trimmed.height, "removed_rows": source.height - trimmed.height, "output": str(output)}
        if not dry_run:
            if trimmed.is_empty():
                raise RuntimeError(f"{definition.factor_id} has no rows in {FACTOR_STORAGE_UNIVERSE}; refusing replacement")
            _write_atomic(trimmed, output)
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"factor_id": definition.factor_id, "version": "v1"}
            manifest.update({
                "storage_universe": FACTOR_STORAGE_UNIVERSE,
                "calculation_universe": manifest.get("calculation_universe", "all valid daily_qfq observations"),
                "rows": trimmed.height,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "storage_compacted_at": _utc_now(),
                "storage_compaction": "filtered existing factor values by daily CSI300 union CSI500 membership; formula was not recomputed",
            })
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, manifest_path)
        results.append(result)
    return results


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="min", pct=True)


def scale(frame: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """Cross-sectional L1 normalization for each trading day."""
    denominator = frame.abs().sum(axis=1).replace(0, np.nan)
    return frame.mul(k).div(denominator, axis=0)


def delay(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.shift(periods)


def delta(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.diff(periods)


def ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).sum()


def sma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).mean()


def stddev(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).std()


def correlation(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return left.rolling(window, min_periods=window).corr(right)


def covariance(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return left.rolling(window, min_periods=window).cov(right)


def ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).min()


def ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).max()


def ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).rank(method="min")


def ts_argmax(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).apply(lambda values: int(np.argmax(values)) + 1, raw=True)


def ts_argmin(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).apply(lambda values: int(np.argmin(values)) + 1, raw=True)


def pairwise_max(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.maximum(left.to_numpy(), right.to_numpy()), index=left.index, columns=left.columns)


def pairwise_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.minimum(left.to_numpy(), right.to_numpy()), index=left.index, columns=left.columns)


def product(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).apply(np.prod, raw=True)


def decay_linear(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Causal linear-decay moving average with no missing-value imputation."""
    weights = np.arange(1, window + 1, dtype=float)
    divisor = weights.sum()
    return frame.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights) / divisor), raw=True
    )


def ewm_sma(frame: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """Chinese technical-analysis SMA(X,N,M), equivalent to recursive EMA M/N."""
    return frame.ewm(alpha=m / n, adjust=False, min_periods=1).mean()


@dataclass(frozen=True)
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    vol: pd.DataFrame
    vwap: pd.DataFrame
    returns: pd.DataFrame


@dataclass(frozen=True)
class FactorDefinition:
    family: str
    number: int
    source_file: str
    source_formula: str
    max_window: int
    evaluator: Callable[[Panel], pd.DataFrame] | None
    status: str = "implemented"
    reason: str | None = None
    polars_status: str = "planned"

    @property
    def factor_id(self) -> str:
        return f"{self.family}_alpha{self.number:03d}_qfq_v1"

    @property
    def directory_name(self) -> str:
        return f"{self.family}_alpha{self.number:03d}_qfq"


def _wq_9(panel: Panel) -> pd.DataFrame:
    d = delta(panel.close)
    return (-d).where(~((ts_min(d, 5) > 0) | (ts_max(d, 5) < 0)), d)


def _wq_10(panel: Panel) -> pd.DataFrame:
    d = delta(panel.close)
    return (-d).where(~((ts_min(d, 4) > 0) | (ts_max(d, 4) < 0)), d)


def _wq_definitions() -> dict[tuple[str, int], FactorDefinition]:
    formulas = {
        1: "rank(Ts_ArgMax(SignedPower((returns<0?stddev(returns,20):close),2),5))-0.5",
        2: "-correlation(rank(delta(log(volume),2)),rank((close-open)/open),6)",
        3: "-correlation(rank(open),rank(volume),10)", 4: "-Ts_Rank(rank(low),9)",
        5: "rank(open-sum(vwap,10)/10)*-abs(rank(close-vwap))", 6: "-correlation(open,volume,10)",
        7: "adv20<volume ? -Ts_Rank(abs(delta(close,7)),60)*sign(delta(close,7)) : -1",
        8: "-rank(sum(open,5)*sum(returns,5)-delay(sum(open,5)*sum(returns,5),10))",
        9: "conditional delta(close,1) reversal", 10: "rank(conditional delta(close,1) reversal)",
        11: "(rank(max(vwap-close,3))+rank(min(vwap-close,3)))*rank(delta(volume,3))",
        12: "sign(delta(volume,1))*-delta(close,1)", 13: "-rank(covariance(rank(close),rank(volume),5))",
        14: "-rank(delta(returns,3))*correlation(open,volume,10)",
        15: "-sum(rank(correlation(rank(high),rank(volume),3)),3)",
        16: "-rank(covariance(rank(high),rank(volume),5))",
        17: "-rank(Ts_Rank(close,10))*rank(delta(delta(close,1),1))*rank(Ts_Rank(volume/adv20,5))",
        18: "-rank(stddev(abs(close-open),5)+(close-open)+correlation(close,open,10))",
        19: "-sign((close-delay(close,7))+delta(close,7))*(1+rank(1+sum(returns,250)))",
        20: "-rank(open-delay(high,1))*rank(open-delay(close,1))*rank(open-delay(low,1))",
        21: "close/volume conditional regime", 22: "-delta(correlation(high,volume,5),5)*rank(stddev(close,20))",
        23: "high above 20-day average ? -delta(high,2) : 0", 24: "100-day close regime reversal",
        25: "rank(-returns*adv20*vwap*(high-close))", 26: "-ts_max(correlation(ts_rank(volume,5),ts_rank(high,5),5),3)",
        27: "rank(mean(correlation(rank(volume),rank(vwap),6),2)) threshold", 28: "scale(correlation(adv20,low,5)+(high+low)/2-close)",
        29: "source adjusted rolling rank expression",
        30: "(1-rank(sum of three close-change signs))*sum(volume,5)/sum(volume,20)",
        31: "three-term decayed close/volume reversal", 32: "scale(mean(close,7)-close)+20*scale(corr(vwap,delay(close,5),230))",
        33: "rank(open/close-1)", 34: "rank(2-rank(std(returns,2)/std(returns,5))-rank(delta(close,1)))",
        35: "ts_rank(volume,32)*(1-ts_rank(close+high-low,16))*(1-ts_rank(returns,32))",
        36: "weighted multi-term correlation signal", 37: "rank(corr(delay(open-close,1),close,200))+rank(open-close)",
        38: "-rank(ts_rank(open,10))*rank(close/open)", 39: "ranked close/volume reversal", 40: "-rank(std(high,10))*corr(high,volume,10)",
        41: "sqrt(high*low)-vwap", 42: "rank(vwap-close)/rank(vwap+close)",
        43: "ts_rank(volume/adv20,20)*ts_rank(-delta(close,7),8)", 44: "-corr(high,rank(volume),5)",
        45: "-rank(mean(delay(close,5),20))*corr(close,volume,2)*rank(corr(sum(close,5),sum(close,20),2))",
        46: "close trend conditional reversal", 47: "close/vwap/volume composite",
        49: "close trend conditional reversal -10%", 50: "-ts_max(rank(corr(rank(volume),rank(vwap),5)),5)",
        51: "close trend conditional reversal -5%", 52: "low/return/volume rank composite",
        53: "-delta(range position,9)", 54: "open/close range power ratio", 55: "-corr(rank(range position),rank(volume),6)",
        57: "-(close-vwap)/decay_linear(rank(ts_argmax(close,30)),2)",
        60: "-(2*scale(rank(range-volume))-scale(rank(ts_argmax(close,10))))",
        61: "rank(vwap-ts_min(vwap,16)) < rank(corr(vwap,adv180,18))",
        62: "rank(corr(vwap,sum(adv20,22),10)) < rank(rank(open)*2 < rank(midpoint)+rank(high))",
        64: "weighted price/volume correlation comparison", 65: "weighted open/vwap correlation comparison",
        66: "negative VWAP/range decayed composite", 68: "high/volume correlation comparison",
        71: "max of two decayed rank signals", 72: "ratio of two decayed correlation ranks",
        73: "negative max of VWAP and open/low decay signals", 74: "close/volume correlation comparison",
        75: "VWAP correlation comparison", 77: "minimum of two decayed rank signals",
        78: "power of two correlation ranks",
        81: "negative product/correlation rank comparison", 83: "range/VWAP volume ratio",
        84: "signed-power VWAP time rank", 85: "power of two correlation ranks", 86: "negative correlation rank comparison",
        88: "minimum of two decayed rank signals", 92: "minimum of two decayed time-rank signals",
        94: "negative VWAP range/correlation power", 95: "open range versus correlation time rank",
        96: "negative maximum of two decayed time ranks", 98: "difference of two decayed ranks",
        99: "negative correlation rank comparison", 101: "normalized intraday close-open range",
    }
    def clean(value: pd.DataFrame) -> pd.DataFrame: return value.replace([np.inf, -np.inf], np.nan)
    evaluators: dict[int, Callable[[Panel], pd.DataFrame]] = {
        1: lambda p: rank(ts_argmax(p.close.where(p.returns >= 0, stddev(p.returns, 20)).pow(2), 5)) - 0.5,
        2: lambda p: clean(-correlation(rank(delta(np.log(p.vol.where(p.vol > 0)), 2)), rank((p.close-p.open)/p.open), 6)),
        3: lambda p: clean(-correlation(rank(p.open), rank(p.vol), 10)),
        4: lambda p: -ts_rank(rank(p.low), 9),
        5: lambda p: rank(p.open-ts_sum(p.vwap, 10)/10) * -rank(p.close-p.vwap).abs(),
        6: lambda p: clean(-correlation(p.open, p.vol, 10)),
        7: lambda p: (-ts_rank(delta(p.close, 7).abs(), 60) * np.sign(delta(p.close, 7))).where(sma(p.vol, 20) < p.vol, -1),
        8: lambda p: -rank(ts_sum(p.open, 5)*ts_sum(p.returns, 5) - delay(ts_sum(p.open, 5)*ts_sum(p.returns, 5), 10)),
        9: _wq_9, 10: lambda p: rank(_wq_10(p)),
        11: lambda p: (rank(ts_max(p.vwap-p.close, 3))+rank(ts_min(p.vwap-p.close, 3))) * rank(delta(p.vol, 3)),
        12: lambda p: np.sign(delta(p.vol, 1)) * -delta(p.close, 1),
        13: lambda p: -rank(covariance(rank(p.close), rank(p.vol), 5)),
        14: lambda p: -rank(delta(p.returns, 3)) * clean(correlation(p.open, p.vol, 10)),
        15: lambda p: -ts_sum(rank(clean(correlation(rank(p.high), rank(p.vol), 3))), 3),
        16: lambda p: -rank(covariance(rank(p.high), rank(p.vol), 5)),
        17: lambda p: -rank(ts_rank(p.close, 10)) * rank(delta(delta(p.close, 1), 1)) * rank(ts_rank(p.vol/sma(p.vol, 20), 5)),
        18: lambda p: -rank(stddev((p.close-p.open).abs(), 5)+(p.close-p.open)+clean(correlation(p.close, p.open, 10))),
        19: lambda p: -np.sign((p.close-delay(p.close, 7))+delta(p.close, 7))*(1+rank(1+ts_sum(p.returns, 250))),
        20: lambda p: -rank(p.open-delay(p.high))*rank(p.open-delay(p.close))*rank(p.open-delay(p.low)),
        21: lambda p: pd.DataFrame(np.where(
            (sma(p.close, 8) + stddev(p.close, 8) < sma(p.close, 2))
            | (p.vol / sma(p.vol, 20) >= 1), -1.0, 1.0
        ), index=p.close.index, columns=p.close.columns),
        22: lambda p: -delta(clean(correlation(p.high, p.vol, 5)), 5) * rank(stddev(p.close, 20)),
        23: lambda p: (-delta(p.high, 2)).where(sma(p.high, 20) < p.high, 0.0),
        24: lambda p: (-delta(p.close, 3)).where(
            ~(delta(sma(p.close, 100), 100) / delay(p.close, 100) <= 0.05),
            -(p.close - ts_min(p.close, 100)),
        ),
        25: lambda p: rank(-p.returns * sma(p.vol, 20) * p.vwap * (p.high - p.close)),
        26: lambda p: -ts_max(clean(correlation(ts_rank(p.vol, 5), ts_rank(p.high, 5), 5)), 3),
        27: lambda p: pd.DataFrame(np.where(
            rank(sma(clean(correlation(rank(p.vol), rank(p.vwap), 6)), 2) / 2.0) > 0.5,
            -1.0, 1.0,
        ), index=p.close.index, columns=p.close.columns),
        28: lambda p: (clean(correlation(sma(p.vol, 20), p.low, 5)) + (p.high + p.low) / 2 - p.close).div(
            (clean(correlation(sma(p.vol, 20), p.low, 5)) + (p.high + p.low) / 2 - p.close).abs().sum(axis=1), axis=0
        ),
        29: lambda p: ts_min(rank(rank(np.log(ts_sum(rank(rank(-rank(delta(p.close - 1, 5)))), 2)))), 5)
        + ts_rank(delay(-p.returns, 6), 5),
        30: lambda p: (1 - rank(np.sign(delta(p.close)) + np.sign(delay(delta(p.close))) + np.sign(delay(delta(p.close), 2)))) * ts_sum(p.vol, 5) / ts_sum(p.vol, 20),
        31: lambda p: rank(rank(rank(decay_linear(-rank(rank(delta(p.close, 10))), 10))))
        + rank(-delta(p.close, 3)) + np.sign(scale(clean(correlation(sma(p.vol, 20), p.low, 12)))),
        32: lambda p: scale(sma(p.close, 7) - p.close) + 20 * scale(clean(correlation(p.vwap, delay(p.close, 5), 230))),
        33: lambda p: rank(-1 + p.open / p.close),
        34: lambda p: rank(2 - rank((stddev(p.returns, 2) / stddev(p.returns, 5)).replace([np.inf, -np.inf], np.nan)) - rank(delta(p.close))),
        35: lambda p: ts_rank(p.vol, 32) * (1 - ts_rank(p.close + p.high - p.low, 16)) * (1 - ts_rank(p.returns, 32)),
        36: lambda p: 2.21 * rank(clean(correlation(p.close - p.open, delay(p.vol), 15)))
        + 0.7 * rank(p.open - p.close) + 0.73 * rank(ts_rank(delay(-p.returns, 6), 5))
        + rank(clean(correlation(p.vwap, sma(p.vol, 20), 6)).abs())
        + 0.6 * rank((sma(p.close, 200) - p.open) * (p.close - p.open)),
        37: lambda p: rank(clean(correlation(delay(p.open - p.close), p.close, 200))) + rank(p.open - p.close),
        38: lambda p: -rank(ts_rank(p.open, 10)) * rank((p.close / p.open).replace([np.inf, -np.inf], np.nan)),
        39: lambda p: -rank(delta(p.close, 7) * (1 - rank(decay_linear(p.vol / sma(p.vol, 20), 9)))) * (1 + rank(ts_sum(p.returns, 250))),
        40: lambda p: -rank(stddev(p.high, 10)) * clean(correlation(p.high, p.vol, 10)),
        41: lambda p: np.sqrt(p.high * p.low) - p.vwap,
        42: lambda p: rank(p.vwap - p.close) / rank(p.vwap + p.close),
        43: lambda p: ts_rank(p.vol / sma(p.vol, 20), 20) * ts_rank(-delta(p.close, 7), 8),
        44: lambda p: -clean(correlation(p.high, rank(p.vol), 5)),
        45: lambda p: -rank(sma(delay(p.close, 5), 20)) * clean(correlation(p.close, p.vol, 2)) * rank(clean(correlation(ts_sum(p.close, 5), ts_sum(p.close, 20), 2))),
        46: lambda p: pd.DataFrame(np.where(
            (((delay(p.close, 20) - delay(p.close, 10)) / 10) - ((delay(p.close, 10) - p.close) / 10)) > 0.25,
            -1.0,
            np.where((((delay(p.close, 20) - delay(p.close, 10)) / 10) - ((delay(p.close, 10) - p.close) / 10)) < 0, 1.0, -delta(p.close)),
        ), index=p.close.index, columns=p.close.columns),
        47: lambda p: (rank(1 / p.close) * p.vol / sma(p.vol, 20)) * (p.high * rank(p.high - p.close) / sma(p.high, 5)) - rank(p.vwap - delay(p.vwap, 5)),
        49: lambda p: pd.DataFrame(np.where(
            (((delay(p.close, 20) - delay(p.close, 10)) / 10) - ((delay(p.close, 10) - p.close) / 10)) < -0.1,
            1.0, -delta(p.close),
        ), index=p.close.index, columns=p.close.columns),
        50: lambda p: -ts_max(rank(clean(correlation(rank(p.vol), rank(p.vwap), 5))), 5),
        51: lambda p: pd.DataFrame(np.where(
            (((delay(p.close, 20) - delay(p.close, 10)) / 10) - ((delay(p.close, 10) - p.close) / 10)) < -0.05,
            1.0, -delta(p.close),
        ), index=p.close.index, columns=p.close.columns),
        52: lambda p: -delta(ts_min(p.low, 5), 5) * rank((ts_sum(p.returns, 240) - ts_sum(p.returns, 20)) / 220) * ts_rank(p.vol, 5),
        53: lambda p: -delta(((p.close - p.low - (p.high - p.close)) / (p.close - p.low).replace(0, 0.0001)), 9),
        54: lambda p: -(p.low - p.close) * p.open.pow(5) / ((p.low - p.high).replace(0, -0.0001) * p.close.pow(5)),
        55: lambda p: -clean(correlation(rank((p.close - ts_min(p.low, 12)) / (ts_max(p.high, 12) - ts_min(p.low, 12)).replace(0, np.nan)), rank(p.vol), 6)),
        57: lambda p: -(p.close - p.vwap) / decay_linear(rank(ts_argmax(p.close, 30)), 2),
        60: lambda p: -(2 * scale(rank((p.close - p.low - (p.high - p.close)) * p.vol / (p.high - p.low).replace(0, np.nan))) - scale(rank(ts_argmax(p.close, 10)))),
        61: lambda p: (rank(p.vwap - ts_min(p.vwap, 16)) < rank(clean(correlation(p.vwap, sma(p.vol, 180), 18)))).astype(float),
        62: lambda p: -(rank(clean(correlation(p.vwap, ts_sum(sma(p.vol, 20), 22), 10))) < rank((rank(p.open) + rank(p.open)) < (rank((p.high + p.low) / 2) + rank(p.high)))).astype(float),
        64: lambda p: -(rank(clean(correlation(ts_sum(p.open * 0.178404 + p.low * (1 - 0.178404), 13), ts_sum(sma(p.vol, 120), 13), 17))) < rank(delta(((p.high + p.low) / 2) * 0.178404 + p.vwap * (1 - 0.178404), 4))).astype(float),
        65: lambda p: -(rank(clean(correlation(p.open * 0.00817205 + p.vwap * (1 - 0.00817205), ts_sum(sma(p.vol, 60), 9), 6))) < rank(p.open - ts_min(p.open, 14))).astype(float),
        66: lambda p: -(rank(decay_linear(delta(p.vwap, 4), 7)) + ts_rank(decay_linear((((p.low - p.vwap) / (p.open - (p.high + p.low) / 2)).replace([np.inf, -np.inf], np.nan)), 11), 7)),
        68: lambda p: -(ts_rank(clean(correlation(rank(p.high), rank(sma(p.vol, 15)), 9)), 14) < rank(delta(p.close * 0.518371 + p.low * (1 - 0.518371), 2))).astype(float),
        71: lambda p: pairwise_max(
            ts_rank(decay_linear(clean(correlation(ts_rank(p.close, 3), ts_rank(sma(p.vol, 180), 12), 18)), 4), 16),
            ts_rank(decay_linear(rank(p.low + p.open - 2 * p.vwap).pow(2), 16), 4),
        ),
        72: lambda p: rank(decay_linear(clean(correlation((p.high + p.low) / 2, sma(p.vol, 40), 9)), 10)) / rank(decay_linear(clean(correlation(ts_rank(p.vwap, 4), ts_rank(p.vol, 19), 7)), 3)),
        73: lambda p: -pairwise_max(
            rank(decay_linear(delta(p.vwap, 5), 3)),
            ts_rank(decay_linear(-delta(p.open * 0.147155 + p.low * (1 - 0.147155), 2) / (p.open * 0.147155 + p.low * (1 - 0.147155)), 3), 17),
        ),
        74: lambda p: -(rank(clean(correlation(p.close, ts_sum(sma(p.vol, 30), 37), 15))) < rank(clean(correlation(rank(p.high * 0.0261661 + p.vwap * (1 - 0.0261661)), rank(p.vol), 11)))).astype(float),
        75: lambda p: (rank(clean(correlation(p.vwap, p.vol, 4))) < rank(clean(correlation(rank(p.low), rank(sma(p.vol, 50)), 12)))).astype(float),
        77: lambda p: pairwise_min(
            rank(decay_linear((p.high + p.low) / 2 - p.vwap, 20)),
            rank(decay_linear(clean(correlation((p.high + p.low) / 2, sma(p.vol, 40), 3)), 6)),
        ),
        78: lambda p: rank(clean(correlation(ts_sum(p.low * 0.352233 + p.vwap * (1 - 0.352233), 20), ts_sum(sma(p.vol, 40), 20), 7))).pow(rank(clean(correlation(rank(p.vwap), rank(p.vol), 6)))),
        81: lambda p: -(rank(np.log(product(rank(rank(clean(correlation(p.vwap, ts_sum(sma(p.vol, 10), 50), 8))).pow(4)), 15))) < rank(clean(correlation(rank(p.vwap), rank(p.vol), 5)))).astype(float),
        83: lambda p: (rank(delay((p.high - p.low) / sma(p.close, 5), 2)) * rank(rank(p.vol))) / (((p.high - p.low) / sma(p.close, 5)) / (p.vwap - p.close)),
        84: lambda p: ts_rank(p.vwap - ts_max(p.vwap, 15), 21).pow(delta(p.close, 5)),
        85: lambda p: rank(clean(correlation(p.high * 0.876703 + p.close * (1 - 0.876703), sma(p.vol, 30), 10))).pow(rank(clean(correlation(ts_rank((p.high + p.low) / 2, 4), ts_rank(p.vol, 10), 7)))),
        86: lambda p: -(ts_rank(clean(correlation(p.close, ts_sum(sma(p.vol, 20), 15), 6)), 20) < rank(p.close - p.vwap)).astype(float),
        88: lambda p: pairwise_min(
            rank(decay_linear(rank(p.open) + rank(p.low) - rank(p.high) - rank(p.close), 8)),
            ts_rank(decay_linear(clean(correlation(ts_rank(p.close, 8), ts_rank(sma(p.vol, 60), 21), 8)), 7), 3),
        ),
        92: lambda p: pairwise_min(
            ts_rank(decay_linear((((p.high + p.low) / 2 + p.close) < (p.low + p.open)).astype(float), 15), 19),
            ts_rank(decay_linear(clean(correlation(rank(p.low), rank(sma(p.vol, 30)), 8)), 7), 7),
        ),
        94: lambda p: -rank(p.vwap - ts_min(p.vwap, 12)).pow(ts_rank(clean(correlation(ts_rank(p.vwap, 20), ts_rank(sma(p.vol, 60), 4), 18)), 3)),
        95: lambda p: (rank(p.open - ts_min(p.open, 12)) < ts_rank(rank(clean(correlation(ts_sum((p.high + p.low) / 2, 19), ts_sum(sma(p.vol, 40), 19), 13))).pow(5), 12)).astype(float),
        96: lambda p: -pairwise_max(
            ts_rank(decay_linear(clean(correlation(rank(p.vwap), rank(p.vol), 4)), 4), 8),
            ts_rank(decay_linear(ts_argmax(clean(correlation(ts_rank(p.close, 7), ts_rank(sma(p.vol, 60), 4), 4)), 13), 14), 13),
        ),
        98: lambda p: rank(decay_linear(clean(correlation(p.vwap, ts_sum(sma(p.vol, 5), 26), 5)), 7))
        - rank(decay_linear(ts_rank(ts_argmin(clean(correlation(rank(p.open), rank(sma(p.vol, 15)), 21)), 9), 7), 8)),
        99: lambda p: -(rank(clean(correlation(ts_sum((p.high + p.low) / 2, 20), ts_sum(sma(p.vol, 60), 20), 9))) < rank(clean(correlation(p.low, p.vol, 6)))).astype(float),
        101: lambda p: (p.close - p.open) / (p.high - p.low + 0.001),
    }
    result: dict[tuple[str, int], FactorDefinition] = {}
    for number in WQ_AVAILABLE:
        result[("wq", number)] = FactorDefinition(
            "wq", number, WQ_REFERENCE, formulas.get(number, "Reference formula pending port"),
            252, evaluators.get(number),
            "implemented" if number in evaluators else "planned",
            None if number in evaluators else "Queued after first validated tranche",
            "implemented" if number in POLARS_WQ_IMPLEMENTED else "planned",
        )
    return result


def _gtja_3(panel: Panel) -> pd.DataFrame:
    prior = delay(panel.close)
    boundary = pd.DataFrame(np.where(panel.close.to_numpy() > prior.to_numpy(), np.minimum(panel.low.to_numpy(), prior.to_numpy()), np.maximum(panel.high.to_numpy(), prior.to_numpy())), index=panel.close.index, columns=panel.close.columns)
    return (panel.close-boundary).where(panel.close.ne(prior), 0.0)


def _gtja_4(panel: Panel) -> pd.DataFrame:
    cond1 = sma(panel.close, 8)+stddev(panel.close, 8) < sma(panel.close, 2)
    cond2 = sma(panel.close, 2) < sma(panel.close, 8)-stddev(panel.close, 8)
    volume_condition = panel.vol/sma(panel.vol, 20) >= 1
    return pd.DataFrame(np.where(cond1, -1.0, np.where(cond2 | volume_condition, 1.0, -1.0)), index=panel.close.index, columns=panel.close.columns)


def _gtja_19(panel: Panel) -> pd.DataFrame:
    prior = delay(panel.close, 5)
    down = (panel.close-prior)/prior
    up = (panel.close-prior)/panel.close
    return down.where(panel.close < prior, up.where(panel.close != prior, 0.0))


def _gtja_iif(condition: pd.DataFrame, when_true: pd.DataFrame | float, when_false: pd.DataFrame | float) -> pd.DataFrame:
    """DolphinDB-style elementwise conditional retaining panel labels."""
    if not isinstance(when_true, pd.DataFrame):
        when_true = pd.DataFrame(when_true, index=condition.index, columns=condition.columns)
    if not isinstance(when_false, pd.DataFrame):
        when_false = pd.DataFrame(when_false, index=condition.index, columns=condition.columns)
    return when_true.where(condition, when_false)


def _gtja_mfirst(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """DolphinDB mfirst uses an inclusive window; 2 means one-session lag."""
    return delay(frame, count - 1)


def _gtja_mrank(frame: pd.DataFrame, percent: bool, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).rank(method="min", pct=percent)


def _gtja_mcount(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).count()


def _gtja_mbeta(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return covariance(left, right, window) / right.rolling(window, min_periods=window).var()


def _gtja_weighted_average(frame: pd.DataFrame, weights: np.ndarray | int, window: int | None = None) -> pd.DataFrame:
    if window is None:
        return sma(frame, int(weights))
    values = np.asarray(weights, dtype=float)
    if len(values) != window:
        raise ValueError(f"GTJA weighted average expected {window} weights, got {len(values)}")
    values = values / values.sum()
    return frame.rolling(window, min_periods=window).apply(lambda row: float(np.dot(row, values)), raw=True)


def _gtja_weights(stop: int, multiplier: float = 1.0) -> np.ndarray:
    return np.arange(1, stop + 1, dtype=float) * multiplier


def _gtja_body(number: int) -> str:
    source = Path(GTJA_REFERENCE).read_text(encoding="utf-8")
    match = re.search(rf"def gtjaAlpha{number}\([^)]*\)\s*\{{", source)
    if not match:
        raise ValueError(f"GTJA source does not define alpha {number}")
    depth = 1
    cursor = match.end()
    while cursor < len(source) and depth:
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        raise ValueError(f"GTJA source alpha {number} has unbalanced braces")
    return source[match.end() : cursor - 1]


def _gtja_expression(expression: str) -> str:
    """Translate the simple, vectorized DolphinDB expression subset used by GTJA191."""
    converted = expression.strip().replace("\\", "/")
    converted = re.sub(r"\btrue\b", "True", converted, flags=re.IGNORECASE)
    converted = re.sub(r"\bfalse\b", "False", converted, flags=re.IGNORECASE)
    converted = re.sub(r"\bNULL\b", "np.nan", converted)
    converted = converted.replace("||", "|").replace("&&", "&")
    converted = re.sub(r"1\.\.(\d+)\s*\*\s*([0-9.]+)", r"_gtja_weights(\1, \2)", converted)
    converted = re.sub(r"1\.\.(\d+)", r"_gtja_weights(\1)", converted)
    converted = re.sub(r"\bmove\(", "delay(", converted)
    converted = re.sub(r"\bmfirst\(", "_gtja_mfirst(", converted)
    converted = re.sub(r"\bmsum\(", "ts_sum(", converted)
    converted = re.sub(r"\bmstd\(", "stddev(", converted)
    converted = re.sub(r"\bmmax\(", "ts_max(", converted)
    converted = re.sub(r"\bmmin\(", "ts_min(", converted)
    converted = re.sub(r"\bmcorr\(", "correlation(", converted)
    converted = re.sub(r"\bmcovar\(", "covariance(", converted)
    converted = re.sub(r"\bmbeta\(", "_gtja_mbeta(", converted)
    converted = re.sub(r"\bmrank\(", "_gtja_mrank(", converted)
    converted = re.sub(r"\bmcount\(", "_gtja_mcount(", converted)
    converted = re.sub(r"\bmavg\(", "_gtja_weighted_average(", converted)
    converted = re.sub(r"\browRank\(", "rank(", converted)
    converted = re.sub(r"\biif\(", "_gtja_iif(", converted)
    converted = re.sub(r"\bratios\(", "_gtja_ratios(", converted)
    converted = re.sub(r"\bewmMean\(", "_gtja_ewm(", converted)
    converted = converted.replace("pow(", "np.power(").replace("sign(", "np.sign(")
    return converted


def _gtja_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    return frame / delay(frame)


def _gtja_ewm(frame: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    return frame.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _evaluate_gtja_source(number: int, panel: Panel) -> pd.DataFrame:
    """Evaluate the straight-line GTJA source subset and reject unsupported control flow."""
    environment: dict[str, object] = {
        "np": np, "abs": abs, "log": np.log, "min": pairwise_min, "max": pairwise_max,
        "rank": rank, "delay": delay, "delta": delta, "ts_sum": ts_sum, "sma": sma,
        "stddev": stddev, "correlation": correlation, "covariance": covariance,
        "ts_min": ts_min, "ts_max": ts_max, "ts_rank": ts_rank, "ts_argmax": ts_argmax,
        "_gtja_iif": _gtja_iif, "_gtja_mfirst": _gtja_mfirst, "_gtja_mrank": _gtja_mrank,
        "_gtja_mcount": _gtja_mcount, "_gtja_mbeta": _gtja_mbeta,
        "_gtja_weighted_average": _gtja_weighted_average, "_gtja_weights": _gtja_weights,
        "_gtja_ratios": _gtja_ratios, "_gtja_ewm": _gtja_ewm,
        "open": panel.open, "high": panel.high, "low": panel.low, "close": panel.close,
        "vol": panel.vol, "vwap": panel.vwap,
    }
    statements = [line.strip().rstrip(";") for line in _gtja_body(number).splitlines() if line.strip()]
    for statement in statements:
        if statement.startswith("return "):
            value = eval(_gtja_expression(statement[7:]), {"__builtins__": {}}, environment)
            if not isinstance(value, pd.DataFrame):
                value = pd.DataFrame(value, index=panel.close.index, columns=panel.close.columns)
            return value.replace([np.inf, -np.inf], np.nan)
        assignment = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", statement)
        if assignment:
            environment[assignment.group(1)] = eval(_gtja_expression(assignment.group(2)), {"__builtins__": {}}, environment)
            continue
        raise ValueError(f"GTJA alpha {number} unsupported statement: {statement}")
    raise ValueError(f"GTJA alpha {number} has no return statement")


def _gtja_definitions() -> dict[tuple[str, int], FactorDefinition]:
    formulas = {1: "-CORR(RANK(DELTA(LOG(VOLUME),1)),RANK((CLOSE-OPEN)/OPEN),6)", 2: "-DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),1)", 3: "SUM(close versus prior close range expression,6)", 4: "close/volume conditional regime", 5: "-TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)", 6: "-RANK(SIGN(DELTA(OPEN*.85+HIGH*.15,4)))", 7: "source-code precedence: RANK(MAX)+RANK(MIN)*RANK(DELTA(VOLUME,3))", 8: "RANK(-DELTA((HIGH+LOW)/2*.2+VWAP*.8,4))", 9: "SMA(range displacement/range/volume,7,2)", 10: "RANK(MAX((RET<0?STD(RET,20):CLOSE)^2,5))", 11: "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)", 12: "RANK(OPEN-MEAN(VWAP,10))*-RANK(ABS(CLOSE-VWAP))", 13: "SQRT(HIGH*LOW)-VWAP", 14: "CLOSE-DELAY(CLOSE,5)", 15: "OPEN/DELAY(CLOSE,1)-1", 16: "-TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5)", 17: "RANK(VWAP-MAX(VWAP,15))^DELTA(CLOSE,5)", 18: "CLOSE/DELAY(CLOSE,5)", 19: "conditional five-session close return", 20: "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100"}
    def safe(value: pd.DataFrame) -> pd.DataFrame: return value.replace([np.inf, -np.inf], np.nan)
    evaluators: dict[int, Callable[[Panel], pd.DataFrame]] = {
        1: lambda p: safe(-correlation(rank(delta(np.log(p.vol.where(p.vol > 0)))), rank((p.close-p.open)/p.open), 6)),
        2: lambda p: -delta((p.close-p.low-(p.high-p.close))/(p.high-p.low)), 3: lambda p: ts_sum(_gtja_3(p), 6), 4: _gtja_4,
        5: lambda p: -ts_max(correlation(ts_rank(p.vol, 5), ts_rank(p.high, 5), 5), 3),
        6: lambda p: -rank(np.sign(delta(p.open*.85+p.high*.15, 4))),
        7: lambda p: rank(ts_max(p.vwap-p.close, 3)) + rank(ts_min(p.vwap-p.close, 3))*rank(delta(p.vol, 3)),
        8: lambda p: rank(-delta((p.high+p.low)/2*.2+p.vwap*.8, 4)),
        9: lambda p: ewm_sma(((p.high+p.low)/2-(delay(p.high)+delay(p.low))/2)*(p.high-p.low)/p.vol, 7, 2),
        10: lambda p: rank(ts_max((p.close.where(p.returns >= 0, stddev(p.returns, 20))).pow(2), 5)),
        11: lambda p: ts_sum((p.close-p.low-(p.high-p.close))/(p.high-p.low)*p.vol, 6),
        12: lambda p: rank(p.open-ts_sum(p.vwap, 10)/10)*-rank((p.close-p.vwap).abs()), 13: lambda p: np.sqrt(p.high*p.low)-p.vwap,
        14: lambda p: p.close-delay(p.close, 5), 15: lambda p: p.open/delay(p.close)-1,
        16: lambda p: -ts_max(rank(correlation(rank(p.vol), rank(p.vwap), 5)), 5),
        17: lambda p: safe(rank(p.vwap-ts_max(p.vwap, 15)).pow(delta(p.close, 5))), 18: lambda p: p.close/delay(p.close, 5), 19: _gtja_19, 20: lambda p: (p.close-delay(p.close, 6))/delay(p.close, 6)*100}
    result: dict[tuple[str, int], FactorDefinition] = {}
    for number in range(1, 192):
        if number == 30: result[("gtja", number)] = FactorDefinition("gtja", number, GTJA_REFERENCE, "WMA(REGRESI(... MKT, SMB, HML ...)^2,20)", 60, None, "missing_input", "MKT/SMB/HML daily factor series is not in the data lake", "missing_input")
        else: result[("gtja", number)] = FactorDefinition("gtja", number, GTJA_REFERENCE, formulas.get(number, "Reference formula pending port"), 252, evaluators.get(number), "implemented" if number in evaluators else "planned", None if number in evaluators else "Queued after first validated tranche", "implemented" if number in POLARS_GTJA_IMPLEMENTED else "planned")
    return result


REGISTRY = _wq_definitions() | _gtja_definitions()


def build_gtja_alpha014(data_root: Path, start: str | None, end: str | None) -> dict[str, object]:
    """Compatibility builder and exact calendar-lag oracle for GTJA Alpha14."""
    definition = _definition("gtja", 14)
    catalog = data_root / "lake" / "catalog" / "a_share.duckdb"
    where = ["q.qfq_close IS NOT NULL", "q.qfq_close > 0", "p.qfq_close IS NOT NULL", "p.qfq_close > 0"]
    if start: where.append(f"q.trade_date >= DATE '{start}'")
    if end: where.append(f"q.trade_date <= DATE '{end}'")
    query = f"""
      WITH calendar AS (
        SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS day_index
        FROM observed_calendar WHERE is_observed_market_day
      )
      SELECT q.trade_date, q.ts_code, q.qfq_close - p.qfq_close AS factor_value
      FROM daily_qfq q JOIN calendar c ON c.trade_date=q.trade_date
      JOIN calendar previous_day ON previous_day.day_index=c.day_index-5
      LEFT JOIN daily_qfq p ON p.ts_code=q.ts_code AND p.trade_date=previous_day.trade_date
      WHERE {' AND '.join(where)} ORDER BY q.trade_date, q.ts_code
    """
    connection = duckdb.connect(str(catalog), read_only=True)
    try: frame = pl.from_arrow(connection.execute(query).arrow())
    finally: connection.close()
    frame = _restrict_storage_universe(frame, _storage_universe(data_root))
    if frame.is_empty(): raise RuntimeError(f"{definition.factor_id} has no rows in {FACTOR_STORAGE_UNIVERSE}")
    output_dir = factor_directory(data_root, definition); output = output_dir / "factor.parquet"; _write_atomic(frame, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"factor_id": definition.factor_id, "version": "v1", "family": "gtja", "number": 14, "formula": definition.source_formula, "source_file": GTJA_REFERENCE, "adjustment": "qfq close calendar-lag implementation", "lag_semantics": "Five prior observed market sessions; missing stock observations do not compress the lag.", "storage_universe": FACTOR_STORAGE_UNIVERSE, "calculation_universe": "all valid daily_qfq observations", "start": str(frame["trade_date"].min()), "end": str(frame["trade_date"].max()), "rows": frame.height, "sha256": digest, "generated_at": _utc_now(), "validation_status": "reference_oracle"}
    temporary = output_dir.joinpath("manifest.json").with_suffix(".json.tmp"); temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, output_dir / "manifest.json")
    return {"output": str(output), **manifest}


def _definition(family: str, number: int) -> FactorDefinition:
    try: return REGISTRY[(family, number)]
    except KeyError as exc: raise ValueError(f"Unknown factor {family}_alpha{number:03d}") from exc


def _load_panel(data_root: Path, start: str | None, end: str | None) -> Panel:
    catalog = data_root / "lake" / "catalog" / "a_share.duckdb"
    if not catalog.exists(): raise FileNotFoundError(f"Catalog does not exist: {catalog}; run a-share-data build-catalog first")
    clauses = ["q.qfq_open > 0", "q.qfq_high > 0", "q.qfq_low > 0", "q.qfq_close > 0", "q.volume_share >= 0"]
    if start: clauses.append(f"q.trade_date >= DATE '{start}'")
    if end: clauses.append(f"q.trade_date <= DATE '{end}'")
    query = f"SELECT c.trade_date, q.ts_code, q.qfq_open, q.qfq_high, q.qfq_low, q.qfq_close, q.qfq_vwap, q.volume_share FROM observed_calendar c LEFT JOIN daily_qfq q USING (trade_date) WHERE c.is_observed_market_day AND q.ts_code IS NOT NULL AND {' AND '.join(clauses)} ORDER BY c.trade_date, q.ts_code"
    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        long = pl.from_arrow(connection.execute(query).arrow()); calendar = pl.from_arrow(connection.execute("SELECT trade_date FROM observed_calendar WHERE is_observed_market_day ORDER BY trade_date").arrow())["trade_date"].to_list()
    finally: connection.close()
    if long.is_empty(): raise RuntimeError("No valid daily rows available for factor construction")
    data = long.to_pandas(); index = pd.Index(calendar, name="trade_date")
    def field(name: str) -> pd.DataFrame: return data.pivot(index="trade_date", columns="ts_code", values=name).reindex(index=index).sort_index(axis=1)
    close = field("qfq_close")
    return Panel(field("qfq_open"), field("qfq_high"), field("qfq_low"), close, field("volume_share"), field("qfq_vwap"), close.pct_change(fill_method=None))


def _to_long(values: pd.DataFrame, start: str | None, end: str | None) -> pl.DataFrame:
    values = values.replace([np.inf, -np.inf], np.nan); values.index.name = "trade_date"
    data = values.stack(future_stack=True).rename("factor_value").reset_index().dropna(subset=["factor_value"]); data.columns = ["trade_date", "ts_code", "factor_value"]
    frame = pl.from_pandas(data).with_columns(pl.col("trade_date").cast(pl.Date), pl.col("factor_value").cast(pl.Float64))
    if start: frame = frame.filter(pl.col("trade_date") >= pl.lit(start).str.to_date())
    if end: frame = frame.filter(pl.col("trade_date") <= pl.lit(end).str.to_date())
    return frame.sort("trade_date", "ts_code")


def factor_directory(data_root: Path, definition: FactorDefinition) -> Path:
    return data_root / "lake" / "derived" / "factors" / definition.directory_name / "v1"


def build_factor(data_root: Path, definition: FactorDefinition, start: str | None, end: str | None, panel: Panel | None = None, engine: str = "pandas", polars_panel: pl.LazyFrame | None = None) -> dict[str, object]:
    if engine == "polars":
        if definition.polars_status != "implemented":
            raise RuntimeError(f"{definition.factor_id} is {definition.polars_status} for the Polars engine")
        from a_share_data.polars_factor_engine import build_factor as build_polars_factor, evaluate_factor
        # Compute enough prior market sessions for rolling windows, then trim
        # output back to the requested interval.  This preserves requested-start
        # values for long-horizon factors such as WQ Alpha019.
        if polars_panel is None:
            frame = build_polars_factor(
                definition.factor_id,
                data_root / "lake" / "catalog" / "a_share.duckdb",
                start,
                end,
                lookback_sessions=definition.max_window + 2,
            )
        else:
            output = evaluate_factor(definition.factor_id, polars_panel)
            if start:
                output = output.filter(pl.col("trade_date") >= pl.lit(start).str.to_date())
            frame = output.collect().sort(KEYS)
    elif engine == "pandas":
        if definition.status != "implemented" or definition.evaluator is None: raise RuntimeError(f"{definition.factor_id} is {definition.status}: {definition.reason or 'not yet ported'}")
        frame = _to_long(definition.evaluator(panel or _load_panel(data_root, start, end)), start, end)
    else:
        raise ValueError(f"Unsupported factor engine {engine!r}; use polars or pandas")
    if frame.is_empty(): raise RuntimeError(f"{definition.factor_id} produced no finite rows")
    frame = _restrict_storage_universe(frame, _storage_universe(data_root))
    if frame.is_empty(): raise RuntimeError(f"{definition.factor_id} has no rows in {FACTOR_STORAGE_UNIVERSE}")
    if frame.select(pl.struct(["trade_date", "ts_code"]).n_unique()).item() != frame.height: raise RuntimeError(f"{definition.factor_id} primary key is not unique")
    output_dir = factor_directory(data_root, definition); output = output_dir / "factor.parquet"; _write_atomic(frame, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"factor_id": definition.factor_id, "version": "v1", "family": definition.family, "number": definition.number, "formula": definition.source_formula, "source_file": definition.source_file, "calculation_engine": engine, "adjustment": "Prices use qfq OHLC/VWAP; volume uses raw volume_share. Raw factor values are not winsorized or standardized.", "input_fields": ["daily_qfq.qfq_open", "daily_qfq.qfq_high", "daily_qfq.qfq_low", "daily_qfq.qfq_close", "daily_qfq.qfq_vwap", "daily_qfq.volume_share", "observed_calendar.trade_date"], "lag_semantics": "Calendar-aligned market sessions; a missing stock observation never compresses a rolling window.", "storage_universe": FACTOR_STORAGE_UNIVERSE, "calculation_universe": "all valid daily_qfq observations", "max_window": definition.max_window, "start": str(frame["trade_date"].min()), "end": str(frame["trade_date"].max()), "rows": frame.height, "sha256": digest, "generated_at": _utc_now(), "validation_status": "pending_reference_oracle"}
    manifest_path = output_dir / "manifest.json"; temporary = manifest_path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, manifest_path)
    return {"output": str(output), **manifest}


def _numbers(value: str) -> list[int]:
    output: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            left, right = part.split("-", 1); output.update(range(int(left), int(right)+1))
        elif part.strip(): output.add(int(part))
    return sorted(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily A-share factor Parquet files"); commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List factor registry status"); listing.add_argument("--family", choices=["wq", "gtja"])
    status = commands.add_parser("status", help="Show built factor artifacts"); status.add_argument("--data-root", default=DEFAULT_DATABASE_DIR)
    build = commands.add_parser("build", help="Build one supported factor"); build.add_argument("--family", required=True, choices=["wq", "gtja"]); build.add_argument("--number", required=True, type=int); build.add_argument("--data-root", default=DEFAULT_DATABASE_DIR); build.add_argument("--start", type=_parse_date); build.add_argument("--end", type=_parse_date); build.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    batch = commands.add_parser("build-batch", help="Build multiple factors from one calendar-aligned panel"); batch.add_argument("--family", required=True, choices=["wq", "gtja"]); batch.add_argument("--numbers", required=True, help="Comma-separated IDs or ranges, for example 1-20,25"); batch.add_argument("--data-root", default=DEFAULT_DATABASE_DIR); batch.add_argument("--start", type=_parse_date); batch.add_argument("--end", type=_parse_date); batch.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    compact = commands.add_parser("compact-universe", help="Filter existing factor artifacts to daily CSI300 union CSI500 members"); compact.add_argument("--data-root", default=DEFAULT_DATABASE_DIR); compact.add_argument("--family", choices=["wq", "gtja"]); compact.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        rows = [{"family": d.family, "number": d.number, "factor_id": d.factor_id, "status": d.status, "polars_status": d.polars_status, "reason": d.reason} for d in REGISTRY.values() if not args.family or d.family == args.family]; print(json.dumps(rows, ensure_ascii=False, indent=2)); return
    if args.command == "status":
        root = Path(args.data_root); rows = [{"factor_id": d.factor_id, "registry_status": d.status, "polars_status": d.polars_status, "artifact_exists": (factor_directory(root, d) / "factor.parquet").exists(), "artifact": str(factor_directory(root, d) / "factor.parquet")} for d in REGISTRY.values()]; print(json.dumps(rows, ensure_ascii=False, indent=2)); return
    root = Path(args.data_root)
    if args.command == "compact-universe":
        print(json.dumps(compact_storage_universe(root, args.family, args.dry_run), ensure_ascii=False, indent=2)); return
    if args.command == "build":
        result = build_gtja_alpha014(root, args.start, args.end) if args.engine == "pandas" and args.family == "gtja" and args.number == 14 else build_factor(root, _definition(args.family, args.number), args.start, args.end, engine=args.engine)
        print(json.dumps(result, ensure_ascii=False)); return
    definitions = [_definition(args.family, number) for number in _numbers(args.numbers)]; panel = _load_panel(root, args.start, args.end) if args.engine == "pandas" else None; polars_panel = None; results = []
    if args.engine == "polars":
        from a_share_data.polars_factor_engine import load_calendar_panel
        max_lookback = max(definition.max_window for definition in definitions) + 2
        # Build the calendar-aligned market grid once.  Factor evaluation below
        # only reuses this in-memory Polars frame and writes one artifact at a time.
        polars_panel = load_calendar_panel(root / "lake" / "catalog" / "a_share.duckdb", args.start, args.end, max_lookback).collect().lazy()
    for definition in definitions:
        try: results.append({"status": "ok", **(build_gtja_alpha014(root, args.start, args.end) if args.engine == "pandas" and definition.family == "gtja" and definition.number == 14 else build_factor(root, definition, args.start, args.end, panel, args.engine, polars_panel))})
        except Exception as exc: results.append({"status": "failed", "factor_id": definition.factor_id, "error": str(exc)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
