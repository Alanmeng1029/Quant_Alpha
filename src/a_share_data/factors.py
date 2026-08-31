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
WQ_REFERENCE = "/Users/alanmxy/大学/大学/alpha101_adjusted.py"
GTJA_REFERENCE = "/Users/alanmxy/大学/大学/gtja191Alpha.dos"
WQ_AVAILABLE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 60,
                61, 62, 64, 65, 66, 68, 71, 72, 73, 74, 75, 77, 78, 81, 83, 84, 85, 86,
                88, 92, 94, 95, 96, 98, 99, 101)


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


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="min", pct=True)


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
    }
    result: dict[tuple[str, int], FactorDefinition] = {}
    for number in WQ_AVAILABLE:
        result[("wq", number)] = FactorDefinition("wq", number, WQ_REFERENCE, formulas.get(number, "Reference formula pending port"), 250 if number == 19 else 60, evaluators.get(number), "implemented" if number in evaluators else "planned", None if number in evaluators else "Queued after first validated tranche")
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
        if number == 30: result[("gtja", number)] = FactorDefinition("gtja", number, GTJA_REFERENCE, "WMA(REGRESI(... MKT, SMB, HML ...)^2,20)", 60, None, "missing_input", "MKT/SMB/HML daily factor series is not in the data lake")
        else: result[("gtja", number)] = FactorDefinition("gtja", number, GTJA_REFERENCE, formulas.get(number, "Reference formula pending port"), 60, evaluators.get(number), "implemented" if number in evaluators else "planned", None if number in evaluators else "Queued after first validated tranche")
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
    output_dir = factor_directory(data_root, definition); output = output_dir / "factor.parquet"; _write_atomic(frame, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"factor_id": definition.factor_id, "version": "v1", "family": "gtja", "number": 14, "formula": definition.source_formula, "source_file": GTJA_REFERENCE, "adjustment": "qfq close calendar-lag implementation", "lag_semantics": "Five prior observed market sessions; missing stock observations do not compress the lag.", "start": str(frame["trade_date"].min()), "end": str(frame["trade_date"].max()), "rows": frame.height, "sha256": digest, "generated_at": _utc_now(), "validation_status": "reference_oracle"}
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


def build_factor(data_root: Path, definition: FactorDefinition, start: str | None, end: str | None, panel: Panel | None = None) -> dict[str, object]:
    if definition.status != "implemented" or definition.evaluator is None: raise RuntimeError(f"{definition.factor_id} is {definition.status}: {definition.reason or 'not yet ported'}")
    frame = _to_long(definition.evaluator(panel or _load_panel(data_root, start, end)), start, end)
    if frame.is_empty(): raise RuntimeError(f"{definition.factor_id} produced no finite rows")
    if frame.select(pl.struct(["trade_date", "ts_code"]).n_unique()).item() != frame.height: raise RuntimeError(f"{definition.factor_id} primary key is not unique")
    output_dir = factor_directory(data_root, definition); output = output_dir / "factor.parquet"; _write_atomic(frame, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"factor_id": definition.factor_id, "version": "v1", "family": definition.family, "number": definition.number, "formula": definition.source_formula, "source_file": definition.source_file, "adjustment": "Prices use qfq OHLC/VWAP; volume uses raw volume_share. Raw factor values are not winsorized or standardized.", "input_fields": ["daily_qfq.qfq_open", "daily_qfq.qfq_high", "daily_qfq.qfq_low", "daily_qfq.qfq_close", "daily_qfq.qfq_vwap", "daily_qfq.volume_share", "observed_calendar.trade_date"], "lag_semantics": "Calendar-aligned market sessions; a missing stock observation never compresses a rolling window.", "max_window": definition.max_window, "start": str(frame["trade_date"].min()), "end": str(frame["trade_date"].max()), "rows": frame.height, "sha256": digest, "generated_at": _utc_now(), "validation_status": "pending_reference_oracle"}
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
    build = commands.add_parser("build", help="Build one supported factor"); build.add_argument("--family", required=True, choices=["wq", "gtja"]); build.add_argument("--number", required=True, type=int); build.add_argument("--data-root", default=DEFAULT_DATABASE_DIR); build.add_argument("--start", type=_parse_date); build.add_argument("--end", type=_parse_date)
    batch = commands.add_parser("build-batch", help="Build multiple factors from one calendar-aligned panel"); batch.add_argument("--family", required=True, choices=["wq", "gtja"]); batch.add_argument("--numbers", required=True, help="Comma-separated IDs or ranges, for example 1-20,25"); batch.add_argument("--data-root", default=DEFAULT_DATABASE_DIR); batch.add_argument("--start", type=_parse_date); batch.add_argument("--end", type=_parse_date)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        rows = [{"family": d.family, "number": d.number, "factor_id": d.factor_id, "status": d.status, "reason": d.reason} for d in REGISTRY.values() if not args.family or d.family == args.family]; print(json.dumps(rows, ensure_ascii=False, indent=2)); return
    if args.command == "status":
        root = Path(args.data_root); rows = [{"factor_id": d.factor_id, "registry_status": d.status, "artifact_exists": (factor_directory(root, d) / "factor.parquet").exists(), "artifact": str(factor_directory(root, d) / "factor.parquet")} for d in REGISTRY.values()]; print(json.dumps(rows, ensure_ascii=False, indent=2)); return
    root = Path(args.data_root)
    if args.command == "build":
        result = build_gtja_alpha014(root, args.start, args.end) if args.family == "gtja" and args.number == 14 else build_factor(root, _definition(args.family, args.number), args.start, args.end)
        print(json.dumps(result, ensure_ascii=False)); return
    definitions = [_definition(args.family, number) for number in _numbers(args.numbers)]; panel = _load_panel(root, args.start, args.end); results = []
    for definition in definitions:
        try: results.append({"status": "ok", **(build_gtja_alpha014(root, args.start, args.end) if definition.family == "gtja" and definition.number == 14 else build_factor(root, definition, args.start, args.end, panel))})
        except Exception as exc: results.append({"status": "failed", "factor_id": definition.factor_id, "error": str(exc)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
