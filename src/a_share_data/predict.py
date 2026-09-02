"""Fast, reproducible rolling LightGBM return predictions.

The module deliberately owns only the factor-source adapter and model protocol.
Portfolio construction consumes its stable wide prediction parquet separately.
"""
from __future__ import annotations

import argparse
import html
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
INFEASIBLE_EXECUTION_CODES = frozenset({"000937.SZ"})


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
        unions = " UNION ALL ".join(f"SELECT trade_date, ts_code, factor_value, '{fid}' factor_id FROM read_parquet('{_sql(path)}')" for fid, path in zip(factor_ids, paths))
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
    """Executable open-to-open labels: entry T+1, exits T+2/T+6."""
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        dates = _dates(conn, start, end)
        if not dates: return pl.DataFrame({"trade_date": [], "ts_code": []})
        # Need future rows beyond requested end, so construct from full calendar and filter signal dates last.
        query = """WITH calendar AS (SELECT trade_date, row_number() over(order by trade_date) n FROM observed_calendar WHERE is_observed_market_day),
        uni AS (SELECT DISTINCT u.trade_date,u.ts_code FROM index_trading_universe u WHERE index_code IN ('000300.SH','000905.SH')),
        raw AS (SELECT u.trade_date,u.ts_code,
          d1.qfq_open e, d2.qfq_open x1, d6.qfq_open x5,
          i1.open b1, i2.open b2, i6.open b5,
          d1.amount_cny a1, d2.amount_cny a2, d6.amount_cny a6,
          d1.observation_status s1,d2.observation_status s2,d6.observation_status s6
          FROM uni u JOIN calendar c ON c.trade_date=u.trade_date
          LEFT JOIN calendar ce ON ce.n=c.n+1 LEFT JOIN calendar c2 ON c2.n=c.n+2 LEFT JOIN calendar c6 ON c6.n=c.n+6
          LEFT JOIN daily_qfq d1 ON d1.ts_code=u.ts_code AND d1.trade_date=ce.trade_date
          LEFT JOIN daily_qfq d2 ON d2.ts_code=u.ts_code AND d2.trade_date=c2.trade_date
          LEFT JOIN daily_qfq d6 ON d6.ts_code=u.ts_code AND d6.trade_date=c6.trade_date
          LEFT JOIN index_daily i1 ON i1.index_code='000905.SH' AND i1.trade_date=ce.trade_date
          LEFT JOIN index_daily i2 ON i2.index_code='000905.SH' AND i2.trade_date=c2.trade_date
          LEFT JOIN index_daily i6 ON i6.index_code='000905.SH' AND i6.trade_date=c6.trade_date)
        SELECT trade_date,ts_code,
          CASE WHEN e>0 AND x1>0 AND b1>0 AND b2>0 AND a1>0 AND a2>0 AND s1='complete_trading' AND s2='complete_trading' THEN x1/e-b2/b1 END excess_h1,
          CASE WHEN e>0 AND x5>0 AND b1>0 AND b5>0 AND a1>0 AND a6>0 AND s1='complete_trading' AND s6='complete_trading' THEN x5/e-b5/b1 END excess_h5 FROM raw"""
        raw = pl.from_arrow(conn.execute(query).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    raw = raw.filter((pl.col("trade_date") >= pl.lit(dates[0]).str.to_date()) & (pl.col("trade_date") <= pl.lit(dates[-1]).str.to_date()))
    return raw.filter(~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES)).with_columns(
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
    features = pl.concat([pl.read_parquet(p) for p in files]).with_columns(pl.col("trade_date").cast(pl.Date)).filter(~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    labels = build_labels(catalog, start, end)
    panel = standardize_features(features, factor_ids).join(labels, on=["trade_date", "ts_code"], how="left")
    dates = sorted(str(x) for x in panel.select("trade_date").unique().to_series().to_list())
    windows = rolling_windows(dates); records=[]; timings=[]; importance=[]
    for signal, train_dates in windows:
        month = signal[:7]; month_dates=[d for d in dates if d[:7]==month and (not end or d<=end)]
        train = panel.filter(pl.col("trade_date").cast(pl.String).is_in(train_dates)); test = panel.filter(pl.col("trade_date").cast(pl.String).is_in(month_dates))
        if test.height == 0: continue
        model_dir=models / f"month={month}"; model_dir.mkdir(parents=True, exist_ok=True)
        month_start=time.perf_counter()
        out=test.select("trade_date","ts_code")
        for horizon, target in (("h1", "excess_h1"), ("h5", "excess_h5")):
            fit=winsorize_labels(train.filter(pl.col(target).is_not_null()), target)
            model=lgb.train(settings.params(), lgb.Dataset(fit.select(factor_ids).to_numpy(), label=fit[target].to_numpy(), feature_name=list(factor_ids)), num_boost_round=settings.num_boost_round)
            model.save_model(str(model_dir / f"{horizon}.txt"))
            out=out.with_columns(pl.Series(f"pred_{horizon}", model.predict(test.select(factor_ids).to_numpy())))
            importance.extend({"model_month":month,"horizon":horizon,"feature":f,"importance":float(v)} for f,v in zip(factor_ids, model.feature_importance()))
        out=out.with_columns(((pl.col("pred_h1") + pl.col("pred_h5")) / 2.0).alias("alpha_daily"),pl.lit(month).alias("model_month"),pl.col("trade_date").shift(-1).over("ts_code").alias("execution_date"))
        # execution date is market-calendar based, not per-stock; repair via date mapping.
        next_map={dates[i]:dates[i+1] for i in range(len(dates)-1)}; out=out.with_columns(pl.col("trade_date").cast(pl.String).replace_strict(next_map, default=None).str.to_date().alias("execution_date"))
        records.append(out); timings.append({"model_month":month,"seconds":time.perf_counter()-month_start,"training_start":train_dates[0],"training_end":train_dates[-1],"training_days":len(train_dates)})
        (models / f"month={month}" / "manifest.json").write_text(json.dumps({"training_start":train_dates[0],"training_end":train_dates[-1],"factor_ids":factor_ids,"feature_hash":hashlib.sha256("|".join(factor_ids).encode()).hexdigest(),"targets":["excess_h1","excess_h5"],"settings":asdict(settings)},indent=2))
    pred=pl.concat(records) if records else pl.DataFrame(); pred.write_parquet(output / "predictions.parquet", compression="zstd")
    evaluation=pred.join(labels,on=["trade_date","ts_code"],how="left")
    summary={"settings":asdict(settings),"factor_ids":factor_ids,"benchmark":"CSI500 open-to-open","oos_start":str(pred["trade_date"].min()) if pred.height else None,"oos_end":str(pred["trade_date"].max()) if pred.height else None,"mean_alpha":_metrics(evaluation,"alpha_daily","excess_h1_h5_mean"),"h1":_metrics(evaluation,"pred_h1","excess_h1"),"h5":_metrics(evaluation,"pred_h5","excess_h5")}
    pl.DataFrame(timings, schema={"model_month":pl.String,"seconds":pl.Float64,"training_start":pl.String,"training_end":pl.String,"training_days":pl.Int64}).write_parquet(output / "model_timings.parquet")
    pl.DataFrame(importance, schema={"model_month":pl.String,"horizon":pl.String,"feature":pl.String,"importance":pl.Float64}).write_parquet(output / "feature_importance.parquet")
    (output / "summary.json").write_text(json.dumps(summary,indent=2,default=str)); return summary


def optimize_dual_alpha_targets(predictions: Path, output: Path, h1_weight: float = .5, turnover_cap: float = .30, temperature: float = 2.0, max_weight: float = .10) -> dict:
    """Blend cross-sectional h1/h5 alpha, then project targets onto a turnover budget."""
    if not (0 <= h1_weight <= 1 and 0 < turnover_cap <= 1 and temperature > 0 and 0 < max_weight <= 1):
        raise ValueError("invalid dual-alpha optimizer parameters")
    source=pl.read_parquet(predictions).with_columns(pl.col("trade_date").cast(pl.Date),pl.col("execution_date").cast(pl.Date)).filter(pl.col("execution_date").is_not_null() & pl.col("pred_h1").is_finite() & pl.col("pred_h5").is_finite() & ~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    previous: dict[str,float]={}; rows=[]
    for key,frame in source.partition_by("trade_date",as_dict=True).items():
        signal_date=key[0] if isinstance(key,tuple) else key; execution_date=frame["execution_date"][0]
        frame=frame.sort("ts_code"); codes=frame["ts_code"].to_list(); a1=frame["pred_h1"].to_numpy(); a5=frame["pred_h5"].to_numpy()
        def z(value: np.ndarray) -> np.ndarray:
            std=value.std(); return (value-value.mean())/std if std > 1e-12 else np.zeros_like(value)
        score=h1_weight*z(a1)+(1-h1_weight)*z(a5); exp=np.exp(np.clip(score/temperature,-30,30)); desired=exp/exp.sum()
        # Project onto the capped simplex without re-inflating capped names.
        free=np.ones(len(desired),dtype=bool); remaining=1.0
        while True:
            proposal=desired[free] / desired[free].sum() * remaining
            overflow=proposal > max_weight + 1e-15
            if not overflow.any():
                desired[free]=proposal; break
            indices=np.flatnonzero(free)[overflow]; desired[indices]=max_weight; free[indices]=False; remaining=1.0-desired[~free].sum()
        old=np.array([previous.get(code,0.0) for code in codes]); l1=float(np.abs(desired-old).sum()); scale=1.0 if not previous else min(1.0,2*turnover_cap/l1)
        weights=old+scale*(desired-old); previous=dict(zip(codes,weights))
        for code,w,s1,s5 in zip(codes,weights,a1,a5): rows.append({"trade_date":signal_date,"execution_date":execution_date,"ts_code":code,"target_weight":float(w),"pred_h1":float(s1),"pred_h5":float(s5),"alpha_daily":float(h1_weight*s1+(1-h1_weight)*s5),"optimizer":"dual_alpha_turnover_projection"})
    result=pl.DataFrame(rows); output.parent.mkdir(parents=True,exist_ok=True); result.write_parquet(output,compression="zstd")
    return {"output":str(output),"days":result.select("trade_date").n_unique(),"rows":result.height,"h1_weight":h1_weight,"h5_weight":1-h1_weight,"turnover_cap_one_way":turnover_cap,"temperature":temperature,"max_weight":max_weight}


def backtest_targets(catalog: Path, target_weights: Path, output: Path, buy_bps: float = 2.1, sell_bps: float = 7.1) -> dict:
    """Open-to-open target backtest with net position trades and CSI500 baseline."""
    output.mkdir(parents=True, exist_ok=True)
    targets = pl.read_parquet(target_weights).with_columns(pl.col("execution_date").cast(pl.Date)).filter(~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    execution_dates = sorted(targets["execution_date"].drop_nulls().unique().to_list())
    if len(execution_dates) < 2: raise ValueError("Need at least two execution dates")
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        prices = pl.from_arrow(conn.execute(f"SELECT trade_date, ts_code, qfq_open FROM daily_qfq WHERE trade_date BETWEEN DATE '{execution_dates[0]}' AND DATE '{execution_dates[-1]}' AND qfq_open > 0").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
        benchmark = pl.from_arrow(conn.execute(f"SELECT trade_date, open FROM index_daily WHERE index_code='000905.SH' AND trade_date BETWEEN DATE '{execution_dates[0]}' AND DATE '{execution_dates[-1]}' AND open > 0").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    by_date = {(day[0] if isinstance(day, tuple) else day): frame for day, frame in targets.partition_by("execution_date", as_dict=True).items()}
    price = {(row[0], row[1]): row[2] for row in prices.select("trade_date", "ts_code", "qfq_open").iter_rows()}
    benchmark_price = dict(benchmark.select("trade_date", "open").iter_rows())
    rows=[]; previous={}; nav=1.0; bench_nav=1.0
    for i, day in enumerate(execution_dates[:-1]):
        next_day=execution_dates[i+1]; frame=by_date[day]; current={code: float(weight) for code,weight in frame.select("ts_code","target_weight").iter_rows() if weight > 0}
        returns=[]; gross=0.0
        for code, weight in current.items():
            p0,p1=price.get((day,code)),price.get((next_day,code))
            if p0 and p1:
                value=p1/p0-1; gross += weight*value; returns.append(value)
        p0,p1=benchmark_price.get(day),benchmark_price.get(next_day)
        bench=p1/p0-1 if p0 and p1 else 0.0
        delta={code: current.get(code,0)-previous.get(code,0) for code in set(current)|set(previous)}
        buy_turnover=sum(max(change,0.0) for change in delta.values())
        sell_turnover=sum(max(-change,0.0) for change in delta.values())
        cost=0.0 if i==0 else buy_turnover*buy_bps/10_000 + sell_turnover*sell_bps/10_000
        net=gross-cost; nav*=1+net; bench_nav*=1+bench
        rows.append({"execution_date":day,"next_execution_date":next_day,"gross_return":gross,"transaction_cost":cost,"net_return":net,"csi500_return":bench,"active_return":net-bench,"buy_turnover":buy_turnover,"sell_turnover":sell_turnover,"nav":nav,"csi500_nav":bench_nav,"holding_count":len(current)})
        previous=current
    daily=pl.DataFrame(rows); daily.write_parquet(output/"portfolio_daily.parquet",compression="zstd")
    net=daily["net_return"].to_numpy(); active=daily["active_return"].to_numpy(); years=len(net)/252
    ann_return=nav**(1/years)-1 if years else 0.0; ann_vol=float(np.std(net,ddof=1)*np.sqrt(252)); active_vol=float(np.std(active,ddof=1)*np.sqrt(252)); running=np.maximum.accumulate(daily["nav"].to_numpy()); max_dd=float(np.min(daily["nav"].to_numpy()/running-1))
    summary={"days":len(rows),"gross_total_return":float(np.prod(1+daily["gross_return"].to_numpy())-1),"net_total_return":nav-1,"csi500_total_return":bench_nav-1,"annualized_return":ann_return,"annualized_volatility":ann_vol,"sharpe":ann_return/ann_vol if ann_vol else None,"information_ratio":float(np.mean(active)/np.std(active,ddof=1)*np.sqrt(252)) if np.std(active,ddof=1) else None,"max_drawdown":max_dd,"average_buy_turnover":float(daily["buy_turnover"].mean()),"average_sell_turnover":float(daily["sell_turnover"].mean()),"buy_cost_bps":buy_bps,"sell_cost_bps":sell_bps,"total_transaction_cost":float(daily["transaction_cost"].sum())}
    (output/"portfolio_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def summarize_holdings(target_weights: Path) -> dict:
    """Summarize realized target-weight concentration on each execution date."""
    frame = pl.read_parquet(target_weights).filter(pl.col("target_weight") > 0)
    if frame.is_empty():
        raise ValueError("target weights are empty")
    daily = frame.group_by("execution_date").agg(
        pl.len().alias("holding_count"),
        pl.col("target_weight").max().alias("max_weight"),
        (pl.col("target_weight") ** 2).sum().alias("hhi"),
        pl.col("target_weight").sort(descending=True).head(10).sum().alias("top10_weight"),
        pl.col("target_weight").sort(descending=True).head(50).sum().alias("top50_weight"),
    )
    def values(column: str) -> dict:
        data = daily[column]
        return {"min": float(data.min()), "median": float(data.median()), "mean": float(data.mean()), "max": float(data.max())}
    effective = 1 / daily["hhi"]
    return {
        "days": daily.height,
        "holding_count": values("holding_count"),
        "max_single_name_weight": values("max_weight"),
        "effective_number_of_holdings": {
            "min": float(effective.min()), "median": float(effective.median()),
            "mean": float(effective.mean()), "max": float(effective.max()),
        },
        "top10_weight": values("top10_weight"),
        "top50_weight": values("top50_weight"),
    }


def render_backtest_report(portfolio_daily: Path, output: Path, title: str = "Portfolio backtest", target_weights: Path | None = None) -> dict:
    """Render a self-contained HTML tear sheet from portfolio_daily.parquet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    frame=pl.read_parquet(portfolio_daily).with_columns(pl.col("execution_date").cast(pl.Date)).sort("execution_date")
    required={"gross_return","net_return","transaction_cost","buy_turnover","sell_turnover","csi500_return"}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"portfolio daily file missing: {sorted(missing)}")
    gross=frame["gross_return"].to_numpy(); net=frame["net_return"].to_numpy(); csi=frame["csi500_return"].to_numpy(); cost=frame["transaction_cost"].to_numpy()
    gross_nav=np.cumprod(1+gross); net_nav=np.cumprod(1+net); csi_nav=np.cumprod(1+csi); fee=[]; nav=1.0; paid=0.0
    for ret, fee_rate in zip(gross,cost):
        paid += nav*fee_rate; nav *= 1+ret-fee_rate; fee.append(paid)
    drawdown=net_nav/np.maximum.accumulate(net_nav)-1
    years=len(net)/252
    def ann(value: np.ndarray) -> float: return float(np.prod(1+value)**(1/years)-1) if years else 0.0
    def vol(value: np.ndarray) -> float: return float(np.std(value,ddof=1)*np.sqrt(252)) if len(value)>1 else 0.0
    net_excess_wealth = float(net_nav[-1] / csi_nav[-1] - 1)
    metrics={"days":len(frame),"start":str(frame["execution_date"][0]),"end":str(frame["execution_date"][-1]),"gross_total_return":float(gross_nav[-1]-1),"net_total_return":float(net_nav[-1]-1),"csi500_total_return":float(csi_nav[-1]-1),"net_excess_wealth_vs_csi500":net_excess_wealth,"gross_annualized_return":ann(gross),"net_annualized_return":ann(net),"csi500_annualized_return":ann(csi),"net_annualized_excess_vs_csi500":float((net_nav[-1] / csi_nav[-1]) ** (1 / years) - 1) if years else 0.0,"average_daily_active_return_bps":float(np.mean(net-csi)*10_000),"annualized_tracking_error":vol(net-csi),"net_sharpe":ann(net)/vol(net) if vol(net) else None,"information_ratio":float(np.mean(net-csi)/np.std(net-csi,ddof=1)*np.sqrt(252)) if np.std(net-csi,ddof=1) else None,"max_drawdown":float(drawdown.min()),"average_buy_turnover":float(frame["buy_turnover"].mean()),"average_sell_turnover":float(frame["sell_turnover"].mean()),"average_fee_bps":float(np.mean(cost)*10_000),"cumulative_fee_paid_on_initial_nav":float(fee[-1]),"average_holding_count":float(frame["holding_count"].mean())}
    holdings = summarize_holdings(target_weights) if target_weights else None
    dates=frame["execution_date"].to_list()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(dates,gross_nav,label="Gross NAV",lw=1.8); ax.plot(dates,net_nav,label="Net NAV",lw=1.8); ax.plot(dates,csi_nav,label="CSI500 NAV",lw=1.5); ax.plot(dates,fee,label="Cumulative fee paid",lw=1.3,ls="--"); ax.set_title(title+" — NAV and fee"); ax.set_ylabel("Initial NAV = 1"); ax.legend(ncol=4,fontsize=9); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output/"nav_and_fee.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True); axes[0].plot(dates,drawdown,color="#c44e52",lw=1.2); axes[0].fill_between(dates,drawdown,0,color="#c44e52",alpha=.2); axes[0].set_ylabel("Net drawdown"); axes[0].set_title(title+" — drawdown and turnover"); axes[1].plot(dates,frame["buy_turnover"].to_numpy(),label="Buy turnover",lw=1); axes[1].plot(dates,frame["sell_turnover"].to_numpy(),label="Sell turnover",lw=1); axes[1].bar(dates,cost,label="Fee",alpha=.35,width=1); axes[1].set_ylabel("Fraction of NAV"); axes[1].legend(ncol=3,fontsize=9); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output/"drawdown_turnover_fee.png",dpi=160); plt.close(fig)
    def row(key: str, value: object) -> str:
        if isinstance(value, float):
            if any(marker in key for marker in ("return", "drawdown", "turnover", "weight", "excess_wealth")):
                value = f"{value:.4%}"
            else:
                value = f"{value:.4f}"
        return f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
    table="".join(row(key,value) for key,value in metrics.items())
    holdings_table=""
    if holdings:
        holdings_table="<h2>Holdings</h2><table>"+"".join(row(key,json.dumps(value,ensure_ascii=False)) for key,value in holdings.items())+"</table>"
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#18212f}}table{{border-collapse:collapse}}th,td{{padding:7px 12px;border:1px solid #d9e1ea;text-align:left}}img{{display:block;max-width:1100px;width:100%;margin:20px 0}}</style></head><body><h1>{html.escape(title)}</h1><p>Return basis: T+1 open to T+2 open; benchmark: CSI500; costs use the portfolio daily ledger.</p><h2>Performance and CSI500 excess</h2><table>{table}</table>{holdings_table}<img src='nav_and_fee.png'><img src='drawdown_turnover_fee.png'></body></html>"""
    report = {**metrics, "holdings": holdings} if holdings else metrics
    (output/"report.html").write_text(page,encoding="utf-8"); (output/"report_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return {"output":str(output),"report":str(output/"report.html"),**report}


def main(argv: list[str] | None = None) -> None:
    p=argparse.ArgumentParser(prog="quant-predict"); sub=p.add_subparsers(dest="command",required=True)
    def common(x): x.add_argument("--catalog",type=Path,required=True); x.add_argument("--start"); x.add_argument("--end")
    b=sub.add_parser("build-features"); common(b); b.add_argument("--factor-root",type=Path,required=True); b.add_argument("--feature-root",type=Path,required=True); b.add_argument("--factor-ids-file",type=Path); b.add_argument("--replace",action="store_true")
    r=sub.add_parser("run-oos"); common(r); r.add_argument("--feature-root",type=Path,required=True); r.add_argument("--output",type=Path,required=True)
    d=sub.add_parser("optimize-dual-alpha"); d.add_argument("--predictions",type=Path,required=True); d.add_argument("--output",type=Path,required=True); d.add_argument("--h1-weight",type=float,default=.5); d.add_argument("--turnover-cap",type=float,default=.30); d.add_argument("--temperature",type=float,default=2.0); d.add_argument("--max-weight",type=float,default=.10)
    bt=sub.add_parser("backtest-portfolio"); bt.add_argument("--catalog",type=Path,required=True); bt.add_argument("--target-weights",type=Path,required=True); bt.add_argument("--output",type=Path,required=True); bt.add_argument("--buy-bps",type=float,default=2.1); bt.add_argument("--sell-bps",type=float,default=7.1)
    rp=sub.add_parser("render-backtest-report"); rp.add_argument("--portfolio-daily",type=Path,required=True); rp.add_argument("--output",type=Path,required=True); rp.add_argument("--title",default="Portfolio backtest"); rp.add_argument("--target-weights",type=Path)
    args=p.parse_args(argv)
    if args.command=="build-features": result=build_features(args.catalog,args.factor_root,args.feature_root,args.start,args.end,args.replace,read_factor_ids(args.factor_ids_file))
    elif args.command=="run-oos": result=run_oos(args.catalog,args.feature_root,args.output,args.start,args.end)
    elif args.command=="optimize-dual-alpha": result=optimize_dual_alpha_targets(args.predictions,args.output,args.h1_weight,args.turnover_cap,args.temperature,args.max_weight)
    elif args.command=="backtest-portfolio": result=backtest_targets(args.catalog,args.target_weights,args.output,args.buy_bps,args.sell_bps)
    else: result=render_backtest_report(args.portfolio_daily,args.output,args.title,args.target_weights)
    print(json.dumps(result,ensure_ascii=False,default=str))

if __name__ == "__main__": main()
