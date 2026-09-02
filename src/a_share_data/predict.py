"""Fast, reproducible rolling LightGBM return predictions.

The module deliberately owns only the factor-source adapter and model protocol.
Portfolio construction consumes its stable wide prediction parquet separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

CORE40 = tuple([f"gtja_alpha{i:03d}_qfq_v1" for i in range(1, 21)] + [f"wq_alpha{i:03d}_qfq_v1" for i in range(1, 21)])
TRAIN_DAYS = 756
LABEL_LAG = 6
INDEX_CODES = ("000300.SH", "000905.SH")


@dataclass(frozen=True)
class LgbmSettings:
    num_boost_round: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 31
    seed: int = 20260831

    def params(self) -> dict[str, object]:
        return {"objective": "regression", "metric": "l2", "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves, "seed": self.seed, "feature_fraction_seed": self.seed,
                "bagging_seed": self.seed, "data_random_seed": self.seed, "deterministic": True,
                "force_row_wise": True, "num_threads": 0, "verbosity": -1}


def _factor_path(root: Path, factor_id: str) -> Path:
    return root / factor_id.removesuffix("_v1") / "v1" / "factor.parquet"


def _sql(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _fingerprint(paths: list[Path]) -> str:
    payload = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _dates(conn: duckdb.DuckDBPyConnection, start: str | None, end: str | None) -> list[str]:
    where = ["is_observed_market_day"]
    if start: where.append(f"trade_date >= DATE '{start}'")
    if end: where.append(f"trade_date <= DATE '{end}'")
    return [str(x[0]) for x in conn.execute("SELECT trade_date FROM observed_calendar WHERE " + " AND ".join(where) + " ORDER BY trade_date").fetchall()]


def read_factor_ids(path: Path | None) -> tuple[str, ...]:
    """Read one factor id per line; comments and blank lines are ignored."""
    if path is None:
        return CORE40
    factor_ids = tuple(
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    )
    if not factor_ids or len(factor_ids) != len(set(factor_ids)):
        raise ValueError("factor id file must contain one or more unique factor ids")
    return factor_ids


def build_features(catalog: Path, factor_root: Path, feature_root: Path, start: str | None = None, end: str | None = None, replace: bool = False, factor_ids: tuple[str, ...] = CORE40) -> dict:
    """Pivot selected long factor parquet files once, retaining the dynamic universe."""
    paths = [_factor_path(factor_root, f) for f in factor_ids]
    missing = [str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError("Missing Core40 factor files: " + ", ".join(missing[:3]))
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest_path = feature_root / "manifest.json"; digest = _fingerprint(paths)
    manifest = {"version": 1, "factor_ids": factor_ids, "factor_hash": digest, "start": start, "end": end}
    expected_years = {d[:4] for d in _dates(duckdb.connect(str(catalog), read_only=True), start, end)}
    if not replace and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old == manifest and all((feature_root / f"year={y}" / "features.parquet").exists() for y in expected_years):
            return {"cache_hit": True, "years": sorted(expected_years), "factor_hash": digest}
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        date_filter = "".join((f" AND u.trade_date >= DATE '{start}'" if start else "", f" AND u.trade_date <= DATE '{end}'" if end else ""))
        unions = " UNION ALL ".join(f"SELECT trade_date, ts_code, factor_value, '{fid}' factor_id FROM read_parquet('{_sql(path)}')" for fid, path in zip(CORE40, paths))
        # The cache contract is Float32.  Extreme/invalid raw values are treated as
        # missing rather than allowing one malformed observation to abort a run.
        columns = ", ".join(f"try_cast(max(factor_value) FILTER (WHERE factor_id='{fid}') AS FLOAT) AS {fid}" for fid in factor_ids)
        query = f"""WITH universe AS (SELECT DISTINCT trade_date, ts_code FROM index_trading_universe u
                         WHERE index_code IN ('000300.SH','000905.SH'){date_filter}), factors AS ({unions})
                     SELECT u.trade_date, u.ts_code, {columns} FROM universe u LEFT JOIN factors f USING(trade_date, ts_code)
                     GROUP BY u.trade_date, u.ts_code ORDER BY u.trade_date, u.ts_code"""
        frame = pl.from_arrow(conn.execute(query).arrow())
    finally: conn.close()
    frame = frame.with_columns(pl.col("trade_date").dt.year().alias("__year"))
    for year, part in frame.partition_by("__year", as_dict=True).items():
        y = year[0] if isinstance(year, tuple) else year
        dest = feature_root / f"year={y}"; dest.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="features.", suffix=".parquet", dir=dest); os.close(fd)
        part.drop("__year").write_parquet(tmp, compression="zstd"); os.replace(tmp, dest / "features.parquet")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"cache_hit": False, "rows": frame.height, "years": sorted(expected_years), "factor_hash": digest}


def build_labels(catalog: Path, start: str | None = None, end: str | None = None) -> pl.DataFrame:
    """Executable VWAP labels: entry T+1, exits T+2/T+6, then daily EW excess."""
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        dates = _dates(conn, start, end)
        if not dates: return pl.DataFrame({"trade_date": [], "ts_code": []})
        # Need future rows beyond requested end, so construct from full calendar and filter signal dates last.
        query = """WITH calendar AS (SELECT trade_date, row_number() over(order by trade_date) n FROM observed_calendar WHERE is_observed_market_day),
        uni AS (SELECT DISTINCT u.trade_date,u.ts_code FROM index_trading_universe u WHERE index_code IN ('000300.SH','000905.SH')),
        raw AS (SELECT u.trade_date,u.ts_code,
          d1.qfq_vwap e, d2.qfq_vwap x1, d6.qfq_vwap x5,
          d1.amount_cny a1, d2.amount_cny a2, d6.amount_cny a6,
          d1.observation_status s1,d2.observation_status s2,d6.observation_status s6
          FROM uni u JOIN calendar c ON c.trade_date=u.trade_date
          LEFT JOIN calendar ce ON ce.n=c.n+1 LEFT JOIN calendar c2 ON c2.n=c.n+2 LEFT JOIN calendar c6 ON c6.n=c.n+6
          LEFT JOIN daily_qfq d1 ON d1.ts_code=u.ts_code AND d1.trade_date=ce.trade_date
          LEFT JOIN daily_qfq d2 ON d2.ts_code=u.ts_code AND d2.trade_date=c2.trade_date
          LEFT JOIN daily_qfq d6 ON d6.ts_code=u.ts_code AND d6.trade_date=c6.trade_date)
        SELECT trade_date,ts_code,
          CASE WHEN e>0 AND x1>0 AND a1>0 AND a2>0 AND s1='complete_trading' AND s2='complete_trading' THEN x1/e-1 END r_h1,
          CASE WHEN e>0 AND x5>0 AND a1>0 AND a6>0 AND s1='complete_trading' AND s6='complete_trading' THEN x5/e-1 END r_h5 FROM raw"""
        raw = pl.from_arrow(conn.execute(query).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    raw = raw.filter((pl.col("trade_date") >= pl.lit(dates[0]).str.to_date()) & (pl.col("trade_date") <= pl.lit(dates[-1]).str.to_date()))
    return raw.with_columns(
        (pl.col("r_h1") - pl.col("r_h1").mean().over("trade_date")).alias("excess_h1"),
        (pl.col("r_h5") - pl.col("r_h5").mean().over("trade_date")).alias("excess_h5"),
    ).with_columns(
        ((pl.col("excess_h1") + pl.col("excess_h5")) / 2.0).alias("excess_h1_h5_mean")
    )


def winsorize_labels(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    low, high = f"__{column}_lo", f"__{column}_hi"
    return frame.with_columns(pl.col(column).quantile(.01).over("trade_date").alias(low), pl.col(column).quantile(.99).over("trade_date").alias(high)).with_columns(pl.col(column).clip(pl.col(low), pl.col(high)).alias(column)).drop(low, high)


def standardize_features(frame: pl.DataFrame, factor_ids: tuple[str, ...] = CORE40) -> pl.DataFrame:
    """Date-by-date winsorization and z-score in the dynamic universe.

    This uses only values observable at the signal close.  Keeping it after the
    raw cache means a future DB source produces identical model inputs.
    """
    result = frame
    bounds = []
    for feature in factor_ids:
        bounds.extend((
            pl.col(feature).quantile(.01).over("trade_date").alias(f"__{feature}_p01"),
            pl.col(feature).quantile(.99).over("trade_date").alias(f"__{feature}_p99"),
        ))
    result = result.with_columns(bounds)
    clipped = [pl.col(feature).clip(pl.col(f"__{feature}_p01"), pl.col(f"__{feature}_p99")).alias(f"__{feature}_clip") for feature in factor_ids]
    result = result.with_columns(clipped)
    zscores = []
    for feature in factor_ids:
        value = pl.col(f"__{feature}_clip")
        deviation = value.std().over("trade_date")
        zscores.append(pl.when(deviation > 1e-12).then((value - value.mean().over("trade_date")) / deviation).otherwise(None).cast(pl.Float32).alias(feature))
    return result.with_columns(zscores).drop([f"__{feature}_{suffix}" for feature in factor_ids for suffix in ("p01", "p99", "clip")])


def rolling_windows(dates: list[str]) -> list[tuple[str, list[str]]]:
    """(month first signal date, exact 756 training dates), with h5 label availability."""
    out=[]
    for i, date in enumerate(dates):
        if (i == 0 or date[:7] != dates[i-1][:7]) and i >= TRAIN_DAYS + LABEL_LAG:
            out.append((date, dates[i-LABEL_LAG-TRAIN_DAYS:i-LABEL_LAG]))
    return out


def _metrics(frame: pl.DataFrame, prediction: str, target: str) -> dict:
    valid = frame.select("trade_date", pl.col(prediction), pl.col(target)).drop_nulls()
    if valid.height == 0: return {"observations": 0}
    daily = valid.group_by("trade_date").agg(pl.corr(prediction, target).alias("pearson_ic"), pl.corr(pl.col(prediction).rank(), pl.col(target).rank()).alias("rank_ic"))
    err = valid.with_columns((pl.col(prediction)-pl.col(target)).alias("e"))
    mean = daily.select(pl.mean("rank_ic")).item(); std = daily.select(pl.std("rank_ic")).item()
    return {"observations": valid.height, "days": daily.height, "mean_rank_ic": mean, "mean_pearson_ic": daily.select(pl.mean("pearson_ic")).item(), "icir": mean / std if std and np.isfinite(std) else None, "positive_ic_ratio": daily.select((pl.col("rank_ic")>0).mean()).item(), "mae": err.select(pl.col("e").abs().mean()).item(), "rmse": err.select((pl.col("e")**2).mean().sqrt()).item()}


def run_oos(catalog: Path, feature_root: Path, output: Path, start: str | None = None, end: str | None = None, settings: LgbmSettings = LgbmSettings()) -> dict:
    output.mkdir(parents=True, exist_ok=True); models = output / "models"; models.mkdir(exist_ok=True)
    files = sorted(feature_root.glob("year=*/features.parquet"));
    if not files: raise FileNotFoundError("Feature cache is empty; run build-features first")
    feature_manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    factor_ids = tuple(feature_manifest["factor_ids"])
    features = pl.concat([pl.read_parquet(p) for p in files]).with_columns(pl.col("trade_date").cast(pl.Date))
    labels = build_labels(catalog, start, end)
    panel = standardize_features(features, factor_ids).join(labels, on=["trade_date", "ts_code"], how="left")
    dates = sorted(str(x) for x in panel.select("trade_date").unique().to_series().to_list())
    windows = rolling_windows(dates); records=[]; timings=[]; importance=[]
    for signal, train_dates in windows:
        month = signal[:7]; month_dates=[d for d in dates if d[:7]==month and (not end or d<=end)]
        train = panel.filter(pl.col("trade_date").cast(pl.String).is_in(train_dates)); test = panel.filter(pl.col("trade_date").cast(pl.String).is_in(month_dates))
        if test.height == 0: continue
        month_start=time.perf_counter(); target="excess_h1_h5_mean"
        fit=winsorize_labels(train.filter(pl.col(target).is_not_null()), target)
        x=fit.select(factor_ids).to_numpy(); y=fit[target].to_numpy()
        model=lgb.train(settings.params(), lgb.Dataset(x, label=y, feature_name=list(factor_ids)), num_boost_round=settings.num_boost_round)
        model_dir=models / f"month={month}"; model_dir.mkdir(parents=True, exist_ok=True); model.save_model(str(model_dir / "mean_h1_h5.txt"))
        prediction=model.predict(test.select(factor_ids).to_numpy())
        importance.extend({"model_month":month,"horizon":"mean_h1_h5","feature":f,"importance":float(v)} for f,v in zip(factor_ids, model.feature_importance()))
        out=test.select("trade_date","ts_code").with_columns(pl.Series("pred_h1_h5_mean", prediction)).with_columns(pl.col("pred_h1_h5_mean").alias("alpha_daily"),pl.lit(month).alias("model_month"),pl.col("trade_date").shift(-1).over("ts_code").alias("execution_date"))
        # execution date is market-calendar based, not per-stock; repair via date mapping.
        next_map={dates[i]:dates[i+1] for i in range(len(dates)-1)}; out=out.with_columns(pl.col("trade_date").cast(pl.String).replace_strict(next_map, default=None).str.to_date().alias("execution_date"))
        records.append(out); timings.append({"model_month":month,"seconds":time.perf_counter()-month_start,"training_start":train_dates[0],"training_end":train_dates[-1],"training_days":len(train_dates)})
        (models / f"month={month}" / "manifest.json").write_text(json.dumps({"training_start":train_dates[0],"training_end":train_dates[-1],"factor_ids":factor_ids,"feature_hash":hashlib.sha256("|".join(factor_ids).encode()).hexdigest(),"target":target,"settings":asdict(settings)},indent=2))
    pred=pl.concat(records) if records else pl.DataFrame(); pred.write_parquet(output / "predictions.parquet", compression="zstd")
    evaluation=pred.join(labels,on=["trade_date","ts_code"],how="left")
    summary={"settings":asdict(settings),"factor_ids":factor_ids,"target":"equal_mean_of_excess_h1_and_excess_h5","oos_start":str(pred["trade_date"].min()) if pred.height else None,"oos_end":str(pred["trade_date"].max()) if pred.height else None,"mean_target":_metrics(evaluation,"alpha_daily","excess_h1_h5_mean"),"alpha_h1":_metrics(evaluation,"alpha_daily","excess_h1"),"alpha_h5":_metrics(evaluation,"alpha_daily","excess_h5")}
    pl.DataFrame(timings, schema={"model_month":pl.String,"seconds":pl.Float64,"training_start":pl.String,"training_end":pl.String,"training_days":pl.Int64}).write_parquet(output / "model_timings.parquet")
    pl.DataFrame(importance, schema={"model_month":pl.String,"horizon":pl.String,"feature":pl.String,"importance":pl.Float64}).write_parquet(output / "feature_importance.parquet")
    (output / "summary.json").write_text(json.dumps(summary,indent=2,default=str)); return summary


def backtest_targets(catalog: Path, target_weights: Path, output: Path, cost_bps: float = 10.0) -> dict:
    """T+1 VWAP target-weight backtest, with equal-weight active benchmark."""
    output.mkdir(parents=True, exist_ok=True)
    targets = pl.read_parquet(target_weights).with_columns(pl.col("execution_date").cast(pl.Date))
    execution_dates = sorted(targets["execution_date"].drop_nulls().unique().to_list())
    if len(execution_dates) < 2: raise ValueError("Need at least two execution dates")
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        prices = pl.from_arrow(conn.execute(f"SELECT trade_date, ts_code, qfq_vwap FROM daily_qfq WHERE trade_date BETWEEN DATE '{execution_dates[0]}' AND DATE '{execution_dates[-1]}' AND qfq_vwap > 0").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    by_date = {(day[0] if isinstance(day, tuple) else day): frame for day, frame in targets.partition_by("execution_date", as_dict=True).items()}
    price = {(row[0], row[1]): row[2] for row in prices.select("trade_date", "ts_code", "qfq_vwap").iter_rows()}
    rows=[]; previous={}; nav=1.0; bench_nav=1.0
    for i, day in enumerate(execution_dates[:-1]):
        next_day=execution_dates[i+1]; frame=by_date[day]; current={code: float(weight) for code,weight in frame.select("ts_code","target_weight").iter_rows() if weight > 0}
        returns=[]; universe_returns=[]; gross=0.0
        for code in frame["ts_code"].to_list():
            p0,p1=price.get((day,code)),price.get((next_day,code))
            if p0 and p1: universe_returns.append(p1/p0-1)
        for code, weight in current.items():
            p0,p1=price.get((day,code)),price.get((next_day,code))
            if p0 and p1:
                value=p1/p0-1; gross += weight*value; returns.append(value)
        bench=float(np.mean(universe_returns)) if universe_returns else 0.0
        l1=sum(abs(current.get(code,0)-previous.get(code,0)) for code in set(current)|set(previous))
        cost=0.0 if i==0 else l1*cost_bps/10_000
        net=gross-cost; nav*=1+net; bench_nav*=1+bench
        rows.append({"execution_date":day,"next_execution_date":next_day,"gross_return":gross,"transaction_cost":cost,"net_return":net,"equal_weight_return":bench,"active_return":net-bench,"turnover_one_way":l1/2,"nav":nav,"equal_weight_nav":bench_nav,"holding_count":len(current)})
        previous=current
    daily=pl.DataFrame(rows); daily.write_parquet(output/"portfolio_daily.parquet",compression="zstd")
    net=daily["net_return"].to_numpy(); active=daily["active_return"].to_numpy(); years=len(net)/252
    ann_return=nav**(1/years)-1 if years else 0.0; ann_vol=float(np.std(net,ddof=1)*np.sqrt(252)); active_vol=float(np.std(active,ddof=1)*np.sqrt(252)); running=np.maximum.accumulate(daily["nav"].to_numpy()); max_dd=float(np.min(daily["nav"].to_numpy()/running-1))
    summary={"days":len(rows),"gross_total_return":float(np.prod(1+daily["gross_return"].to_numpy())-1),"net_total_return":nav-1,"equal_weight_total_return":bench_nav-1,"annualized_return":ann_return,"annualized_volatility":ann_vol,"sharpe":ann_return/ann_vol if ann_vol else None,"information_ratio":float(np.mean(active)/np.std(active,ddof=1)*np.sqrt(252)) if np.std(active,ddof=1) else None,"max_drawdown":max_dd,"average_turnover_one_way":float(daily["turnover_one_way"].mean()),"total_transaction_cost":float(daily["transaction_cost"].sum())}
    (output/"portfolio_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def main(argv: list[str] | None = None) -> None:
    p=argparse.ArgumentParser(prog="quant-predict"); sub=p.add_subparsers(dest="command",required=True)
    def common(x): x.add_argument("--catalog",type=Path,required=True); x.add_argument("--start"); x.add_argument("--end")
    b=sub.add_parser("build-features"); common(b); b.add_argument("--factor-root",type=Path,required=True); b.add_argument("--feature-root",type=Path,required=True); b.add_argument("--factor-ids-file",type=Path); b.add_argument("--replace",action="store_true")
    r=sub.add_parser("run-oos"); common(r); r.add_argument("--feature-root",type=Path,required=True); r.add_argument("--output",type=Path,required=True)
    bt=sub.add_parser("backtest-portfolio"); bt.add_argument("--catalog",type=Path,required=True); bt.add_argument("--target-weights",type=Path,required=True); bt.add_argument("--output",type=Path,required=True); bt.add_argument("--cost-bps",type=float,default=10.0)
    args=p.parse_args(argv)
    if args.command=="build-features": result=build_features(args.catalog,args.factor_root,args.feature_root,args.start,args.end,args.replace,read_factor_ids(args.factor_ids_file))
    elif args.command=="run-oos": result=run_oos(args.catalog,args.feature_root,args.output,args.start,args.end)
    else: result=backtest_targets(args.catalog,args.target_weights,args.output,args.cost_bps)
    print(json.dumps(result,ensure_ascii=False,default=str))

if __name__ == "__main__": main()
