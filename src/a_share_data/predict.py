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

from a_share_data.ensemble import SelectionSettings, fit_predict_neural, select_features

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


@dataclass(frozen=True)
class EnsembleSettings:
    models: tuple[str, ...] = ("lgbm",)
    refit_months: int = 3
    neural_epochs: int = 20
    neural_max_samples: int = 60_000


def _factor_path(root: Path, factor_id: str) -> Path:
    return root / factor_id.removesuffix("_v1") / "v1" / "factor.parquet"


def _sql(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _fingerprint(paths: list[Path]) -> str:
    payload = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _minute_year_glob(root: Path, year: str) -> Path:
    """The minute-factor writer stores one wide Parquet file per trade date."""
    return root / f"year={year}" / "*.parquet"


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


def all_factor_ids(factor_root: Path) -> tuple[str, ...]:
    """Discover the current factor artifacts without using their future returns."""
    found = tuple(sorted(f"{path.parent.parent.name}_v1" for path in factor_root.glob("*_qfq/v1/factor.parquet")))
    if not found:
        raise FileNotFoundError(f"No factor artifacts found under {factor_root}")
    return found


def build_features(catalog: Path, factor_root: Path, feature_root: Path, start: str | None = None, end: str | None = None, replace: bool = False, factor_ids: tuple[str, ...] = CORE40, minute_factor_sources: tuple[tuple[Path, tuple[str, ...]], ...] = ()) -> dict:
    """Build a daily feature cache from long daily and optional wide minute factors.

    Minute candidates remain in their single daily-wide dataset.  Reading them
    directly prevents multiplying the same data into one long Parquet file per
    minute factor merely for model input.
    """
    minute_factor_ids = tuple(factor for _, factors in minute_factor_sources for factor in factors)
    if len(set(factor_ids)) != len(factor_ids) or len(set(minute_factor_ids)) != len(minute_factor_ids):
        raise ValueError("daily and minute factor ids must each be unique")
    if set(factor_ids).intersection(minute_factor_ids):
        raise ValueError("daily and minute factor ids must not overlap")
    all_factor_ids = factor_ids + minute_factor_ids
    paths = [_factor_path(factor_root, f) for f in factor_ids]
    missing = [str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError("Missing daily factor files: " + ", ".join(missing[:3]))
    for minute_factor_dataset, source_ids in minute_factor_sources:
        if not source_ids:
            raise ValueError(f"No factor ids supplied for minute dataset {minute_factor_dataset}")
        minute_files = sorted(minute_factor_dataset.glob("year=*/*.parquet"))
        if not minute_files:
            raise FileNotFoundError(f"No minute factor Parquet files under {minute_factor_dataset}")
        paths.extend(minute_files)
        manifest_file = minute_factor_dataset / "manifest.json"
        if manifest_file.exists():
            paths.append(manifest_file)
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest_path = feature_root / "manifest.json"; digest = _fingerprint(paths)
    manifest = {
        "version": 2,
        "factor_ids": all_factor_ids,
        "daily_factor_ids": factor_ids,
        "minute_factor_ids": minute_factor_ids,
        "minute_factor_sources": [
            {"dataset": str(dataset.resolve()), "factor_ids": ids}
            for dataset, ids in minute_factor_sources
        ],
        "factor_hash": digest,
        "start": start,
        "end": end,
    }
    expected_years = {d[:4] for d in _dates(duckdb.connect(str(catalog), read_only=True), start, end)}
    if not replace and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old == manifest and all((feature_root / f"year={y}" / "features.parquet").exists() for y in expected_years):
            return {"cache_hit": True, "years": sorted(expected_years), "factor_hash": digest}
    conn = duckdb.connect(str(catalog), read_only=True)
    rows = 0
    try:
        # Pivot one calendar year at a time.  This keeps the all-factor cache
        # bounded in memory and pushes the date predicate into every parquet scan.
        daily_columns = [f"try_cast(max(f.factor_value) FILTER (WHERE f.factor_id='{fid}') AS FLOAT) AS \"{fid}\"" for fid in factor_ids]
        minute_columns = [
            f"try_cast(max(m{source_index}.\"{fid}\") AS FLOAT) AS \"{fid}\""
            for source_index, (_, source_ids) in enumerate(minute_factor_sources)
            for fid in source_ids
        ]
        columns = ", ".join(daily_columns + minute_columns)
        for year in sorted(expected_years):
            lower = max(start or f"{year}-01-01", f"{year}-01-01")
            upper = min(end or f"{year}-12-31", f"{year}-12-31")
            unions = " UNION ALL ".join(f"SELECT trade_date, ts_code, factor_value, '{fid}' factor_id FROM read_parquet('{_sql(path)}') WHERE trade_date BETWEEN DATE '{lower}' AND DATE '{upper}'" for fid, path in zip(factor_ids, paths[:len(factor_ids)]))
            factors_cte = unions if unions else "SELECT CAST(NULL AS DATE) trade_date, CAST(NULL AS VARCHAR) ts_code, CAST(NULL AS DOUBLE) factor_value, CAST(NULL AS VARCHAR) factor_id WHERE FALSE"
            minute_ctes = []
            minute_joins = []
            for source_index, (dataset, source_ids) in enumerate(minute_factor_sources):
                minute_glob = _minute_year_glob(dataset, year)
                selected = ", ".join(f'\"{factor}\"' for factor in source_ids)
                minute_ctes.append(f"minute_{source_index} AS (SELECT trade_date, ts_code, {selected} FROM read_parquet('{_sql(minute_glob)}') WHERE trade_date BETWEEN DATE '{lower}' AND DATE '{upper}')")
                minute_joins.append(f"LEFT JOIN minute_{source_index} m{source_index} USING(trade_date, ts_code)")
            minute_sql = ", " + ", ".join(minute_ctes) if minute_ctes else ""
            joins_sql = " ".join(minute_joins)
            query = f"""WITH universe AS (SELECT DISTINCT trade_date, ts_code FROM index_trading_universe
                             WHERE index_code IN ('000300.SH','000905.SH') AND trade_date BETWEEN DATE '{lower}' AND DATE '{upper}'), factors AS ({factors_cte}){minute_sql}
                         SELECT u.trade_date, u.ts_code, {columns} FROM universe u LEFT JOIN factors f USING(trade_date, ts_code)
                         {joins_sql}
                         GROUP BY u.trade_date, u.ts_code ORDER BY u.trade_date, u.ts_code"""
            part = pl.from_arrow(conn.execute(query).arrow())
            rows += part.height
            dest = feature_root / f"year={year}"; dest.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix="features.", suffix=".parquet", dir=dest); os.close(fd)
            part.write_parquet(tmp, compression="zstd"); os.replace(tmp, dest / "features.parquet")
    finally: conn.close()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"cache_hit": False, "rows": rows, "years": sorted(expected_years), "factor_hash": digest, "factor_count": len(all_factor_ids)}


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


def run_oos(catalog: Path, feature_root: Path, output: Path, start: str | None = None, end: str | None = None, settings: LgbmSettings = LgbmSettings(), ensemble: EnsembleSettings = EnsembleSettings(), selection: SelectionSettings = SelectionSettings(), oos_start: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True); models = output / "models"; models.mkdir(exist_ok=True)
    files = sorted(feature_root.glob("year=*/features.parquet"));
    if not files: raise FileNotFoundError("Feature cache is empty; run build-features first")
    feature_manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    factor_ids = tuple(feature_manifest["factor_ids"])
    features = pl.concat([pl.read_parquet(p) for p in files]).with_columns(pl.col("trade_date").cast(pl.Date)).filter(~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    labels = build_labels(catalog, start, end)
    panel = standardize_features(features, factor_ids).join(labels, on=["trade_date", "ts_code"], how="left")
    dates = sorted(str(x) for x in panel.select("trade_date").unique().to_series().to_list())
    invalid_models=set(ensemble.models)-{"lgbm","mlp","transformer"}
    if invalid_models or not ensemble.models: raise ValueError(f"unsupported models: {sorted(invalid_models)}")
    windows = rolling_windows(dates); records=[]; timings=[]; importance=[]; selections=[]; neural_losses=[]
    for window_index, (signal, train_dates) in enumerate(windows):
        if window_index % ensemble.refit_months: continue
        if oos_start and signal < oos_start: continue
        month = signal[:7]
        all_months=sorted({date[:7] for date in dates}); month_index=all_months.index(month)
        test_months=set(all_months[month_index:month_index + ensemble.refit_months])
        month_dates=[d for d in dates if d[:7] in test_months and (not end or d<=end)]
        train = panel.filter(pl.col("trade_date").cast(pl.String).is_in(train_dates)); test = panel.filter(pl.col("trade_date").cast(pl.String).is_in(month_dates))
        if test.height == 0: continue
        model_dir=models / f"month={month}"; model_dir.mkdir(parents=True, exist_ok=True)
        month_start=time.perf_counter()
        out=test.select("trade_date","ts_code")
        selected, diagnostics = select_features(train, factor_ids, "excess_h1_h5_mean", selection)
        selections.extend({"model_month":month, "feature":row["feature"], "selected":row["feature"] in selected, **row} for row in diagnostics)
        fit_h1=winsorize_labels(train.filter(pl.col("excess_h1").is_not_null()), "excess_h1")
        fit_h5=winsorize_labels(train.filter(pl.col("excess_h5").is_not_null()), "excess_h5")
        if "lgbm" in ensemble.models:
            for horizon, target, fit in (("h1", "excess_h1", fit_h1), ("h5", "excess_h5", fit_h5)):
                model=lgb.train(settings.params(), lgb.Dataset(fit.select(selected).to_numpy(), label=fit[target].to_numpy(), feature_name=list(selected)), num_boost_round=settings.num_boost_round)
                model.save_model(str(model_dir / f"lgbm_{horizon}.txt"))
                out=out.with_columns(pl.Series(f"lgbm_{horizon}", model.predict(test.select(selected).to_numpy())))
                importance.extend({"model_month":month,"horizon":horizon,"feature":f,"importance":float(v)} for f,v in zip(selected, model.feature_importance()))
        neural_fit=winsorize_labels(winsorize_labels(train.drop_nulls(["excess_h1", "excess_h5"]), "excess_h1"), "excess_h5")
        neural_metadata=[]
        for model_name in (name for name in ensemble.models if name in {"mlp", "transformer"}):
            prediction, metadata=fit_predict_neural(
                model_name, neural_fit.select(selected).to_numpy(), neural_fit.select("excess_h1", "excess_h5").to_numpy(),
                test.select(selected).to_numpy(), settings.seed + window_index, ensemble.neural_epochs, ensemble.neural_max_samples,
                neural_fit.get_column("trade_date").to_numpy(),
            )
            out=out.with_columns(pl.Series(f"{model_name}_h1", prediction[:,0]), pl.Series(f"{model_name}_h5", prediction[:,1]))
            neural_metadata.append({"model":model_name, **metadata})
            neural_losses.extend({"model_month":month, "model":model_name, **point} for point in metadata["loss_history"])
        out=out.with_columns(pl.lit(month).alias("model_month"))
        # execution date is market-calendar based, not per-stock; repair via date mapping.
        next_map={dates[i]:dates[i+1] for i in range(len(dates)-1)}; out=out.with_columns(pl.col("trade_date").cast(pl.String).replace_strict(next_map, default=None).str.to_date().alias("execution_date"))
        records.append(out); timings.append({"model_month":month,"seconds":time.perf_counter()-month_start,"training_start":train_dates[0],"training_end":train_dates[-1],"training_days":len(train_dates),"selected_features":len(selected)})
        (models / f"month={month}" / "manifest.json").write_text(json.dumps({"training_start":train_dates[0],"training_end":train_dates[-1],"factor_ids":selected,"feature_hash":hashlib.sha256("|".join(selected).encode()).hexdigest(),"targets":["excess_h1","excess_h5"],"lgbm_settings":asdict(settings),"ensemble_settings":asdict(ensemble),"selection_settings":asdict(selection),"neural":neural_metadata},indent=2))
    pred=pl.concat(records) if records else pl.DataFrame()
    prediction_columns=[f"{name}_{horizon}" for name in ensemble.models for horizon in ("h1","h5")]
    zscores=[]
    for column in prediction_columns:
        zscores.append(pl.when(pl.col(column).std().over("trade_date") > 1e-12).then((pl.col(column)-pl.col(column).mean().over("trade_date"))/pl.col(column).std().over("trade_date")).otherwise(0.0).alias(f"__z_{column}"))
    if pred.height:
        pred=pred.with_columns(zscores).with_columns(
            (sum((pl.col(f"__z_{name}_h1") for name in ensemble.models), pl.lit(0.0)) / len(ensemble.models)).alias("pred_h1"),
            (sum((pl.col(f"__z_{name}_h5") for name in ensemble.models), pl.lit(0.0)) / len(ensemble.models)).alias("pred_h5"),
        ).with_columns(((pl.col("pred_h1") + pl.col("pred_h5")) / 2.0).alias("alpha_daily"))
    pred.write_parquet(output / "predictions.parquet", compression="zstd")
    evaluation=pred.join(labels,on=["trade_date","ts_code"],how="left")
    summary={"lgbm_settings":asdict(settings),"ensemble_settings":asdict(ensemble),"selection_settings":asdict(selection),"factor_ids":factor_ids,"benchmark":"CSI500 open-to-open","oos_start":str(pred["trade_date"].min()) if pred.height else None,"oos_end":str(pred["trade_date"].max()) if pred.height else None,"mean_alpha":_metrics(evaluation,"alpha_daily","excess_h1_h5_mean"),"h1":_metrics(evaluation,"pred_h1","excess_h1"),"h5":_metrics(evaluation,"pred_h5","excess_h5")}
    pl.DataFrame(timings, schema={"model_month":pl.String,"seconds":pl.Float64,"training_start":pl.String,"training_end":pl.String,"training_days":pl.Int64,"selected_features":pl.Int64}).write_parquet(output / "model_timings.parquet")
    pl.DataFrame(importance, schema={"model_month":pl.String,"horizon":pl.String,"feature":pl.String,"importance":pl.Float64}).write_parquet(output / "feature_importance.parquet")
    pl.DataFrame(selections).write_parquet(output / "rolling_feature_selection.parquet")
    if neural_losses:
        loss_frame=pl.DataFrame(neural_losses).sort("model_month", "model", "epoch")
        loss_frame.write_parquet(output / "neural_loss_history.parquet", compression="zstd")
        loss_frame.write_csv(output / "neural_loss_history.csv")
    (output / "summary.json").write_text(json.dumps(summary,indent=2,default=str)); return summary


def project_capped_simplex(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Normalize non-negative scores under a fully invested single-name cap."""
    if len(weights) * max_weight < 1 - 1e-12:
        raise ValueError("max_weight makes a fully invested portfolio infeasible")
    result = np.maximum(weights.astype(float, copy=True), 0.0)
    if result.sum() <= 0:
        raise ValueError("at least one positive target weight is required")
    result /= result.sum()
    free = result > 0
    remaining = 1.0
    while True:
        if not free.any() or result[free].sum() <= 0:
            raise ValueError("too few retained names for the requested maximum weight")
        proposal = result[free] / result[free].sum() * remaining
        overflow = proposal > max_weight + 1e-15
        if not overflow.any():
            result[free] = proposal
            return result
        indices = np.flatnonzero(free)[overflow]
        result[indices] = max_weight
        free[indices] = False
        remaining = 1.0 - result[~free].sum()


def optimize_dual_alpha_targets(predictions: Path, output: Path, h1_weight: float = .5, turnover_cap: float = .30, temperature: float = 2.0, max_weight: float = .10, min_weight: float = 0.0, top_fraction: float = 1.0, weighting: str = "softmax") -> dict:
    """Blend h1/h5 alpha, select a score fraction, then turnover-project targets.

    ``equal`` weighting is deliberately available as an optimizer diagnostic:
    it measures the simple top-quantile signal without softmax concentration.
    """
    if not (0 <= h1_weight <= 1 and 0 < turnover_cap <= 1 and temperature > 0 and 0 <= min_weight <= max_weight <= 1 and 0 < top_fraction <= 1 and weighting in {"softmax", "equal"}):
        raise ValueError("invalid dual-alpha optimizer parameters")
    source=pl.read_parquet(predictions).with_columns(pl.col("trade_date").cast(pl.Date),pl.col("execution_date").cast(pl.Date)).filter(pl.col("execution_date").is_not_null() & pl.col("pred_h1").is_finite() & pl.col("pred_h5").is_finite() & ~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    previous: dict[str,float]={}; rows=[]
    for key,frame in source.partition_by("trade_date",as_dict=True).items():
        signal_date=key[0] if isinstance(key,tuple) else key; execution_date=frame["execution_date"][0]
        frame=frame.sort("ts_code"); codes=frame["ts_code"].to_list(); a1=frame["pred_h1"].to_numpy(); a5=frame["pred_h5"].to_numpy()
        def z(value: np.ndarray) -> np.ndarray:
            std=value.std(); return (value-value.mean())/std if std > 1e-12 else np.zeros_like(value)
        score=h1_weight*z(a1)+(1-h1_weight)*z(a5)
        selected_count = max(1, int(np.ceil(len(codes) * top_fraction)))
        selected_index = np.argsort(score)[-selected_count:]
        desired = np.zeros_like(score)
        if weighting == "equal":
            desired[selected_index] = project_capped_simplex(np.ones(selected_count), max_weight)
        else:
            exp = np.exp(np.clip(score[selected_index] / temperature, -30, 30))
            desired[selected_index] = project_capped_simplex(exp, max_weight)
        old=np.array([previous.get(code,0.0) for code in codes]); l1=float(np.abs(desired-old).sum()); scale=1.0 if not previous else min(1.0,2*turnover_cap/l1)
        weights=old+scale*(desired-old); previous=dict(zip(codes,weights))
        # The minimum is applied to the executed target, not merely the
        # unconstrained alpha target.  Removed micro-positions are redistributed
        # proportionally and the single-name cap remains hard.
        if min_weight:
            keep = weights >= min_weight
            minimum_names = int(np.ceil(1 / max_weight))
            if int(keep.sum()) < minimum_names:
                keep[np.argsort(weights)[-minimum_names:]] = True
            weights = project_capped_simplex(np.where(keep, weights, 0.0), max_weight)
            previous = dict(zip(codes, weights))
        for code,w,s1,s5 in zip(codes,weights,a1,a5):
            if w > 0:
                rows.append({"trade_date":signal_date,"execution_date":execution_date,"ts_code":code,"target_weight":float(w),"pred_h1":float(s1),"pred_h5":float(s5),"alpha_daily":float(h1_weight*s1+(1-h1_weight)*s5),"optimizer":"dual_alpha_turnover_projection"})
    result=pl.DataFrame(rows); output.parent.mkdir(parents=True,exist_ok=True); result.write_parquet(output,compression="zstd")
    return {"output":str(output),"days":result.select("trade_date").n_unique(),"rows":result.height,"h1_weight":h1_weight,"h5_weight":1-h1_weight,"turnover_cap_one_way":turnover_cap,"temperature":temperature,"max_weight":max_weight,"min_weight":min_weight,"top_fraction":top_fraction,"weighting":weighting}


def staggered_top_quantile_targets(predictions: Path, output: Path, top_fraction: float = .10, holding_days: int = 5) -> dict:
    """Build equal-weight, H5-ranked sleeves that rebalance one vintage a day.

    Each vintage receives ``1 / holding_days`` of NAV.  On signal date *T*,
    only the corresponding sleeve is refreshed and executes at *T+1* open.
    The aggregate target is therefore directly consumable by the normal
    portfolio backtester while keeping each sleeve for five market sessions.
    """
    if not (0 < top_fraction <= 1 and holding_days >= 1):
        raise ValueError("top_fraction must be in (0, 1] and holding_days must be positive")
    source = pl.read_parquet(predictions).with_columns(
        pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date)
    ).filter(
        pl.col("execution_date").is_not_null() & pl.col("pred_h5").is_finite() & ~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES)
    )
    by_date = {
        (key[0] if isinstance(key, tuple) else key): frame
        for key, frame in source.partition_by("trade_date", as_dict=True).items()
    }
    sleeves: list[dict[str, float]] = [dict() for _ in range(holding_days)]
    rows: list[dict[str, object]] = []
    selected_counts: list[int] = []
    for index, signal_date in enumerate(sorted(by_date)):
        frame = by_date[signal_date].sort("pred_h5", descending=True)
        count = max(1, int(np.ceil(frame.height * top_fraction)))
        chosen = frame.head(count)
        sleeve = index % holding_days
        sleeve_weight = 1.0 / holding_days / count
        sleeves[sleeve] = {code: sleeve_weight for code in chosen.get_column("ts_code").to_list()}
        aggregate: dict[str, float] = {}
        for vintage in sleeves:
            for code, weight in vintage.items():
                aggregate[code] = aggregate.get(code, 0.0) + weight
        execution_date = frame.get_column("execution_date")[0]
        for code, weight in sorted(aggregate.items()):
            rows.append({"trade_date": signal_date, "execution_date": execution_date, "ts_code": code, "target_weight": weight, "optimizer": "staggered_h5_top_quantile_equal_weight"})
        selected_counts.append(count)
    result = pl.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    return {"output": str(output), "days": len(by_date), "rows": result.height, "top_fraction": top_fraction, "holding_days": holding_days, "mean_names_per_sleeve": float(np.mean(selected_counts))}


def staggered_dual_hysteresis_targets(predictions: Path, output: Path, h1_allocation: float = .20, entry_fraction: float = .08, exit_fraction: float = .12, holding_days: int = 5) -> dict:
    """Combine a daily H1 sleeve with staggered H5 sleeves using rank buffers.

    A name enters when it reaches the top ``entry_fraction`` and remains until
    it falls below ``exit_fraction``.  The H1 sleeve receives ``h1_allocation``
    and the remaining capital is split evenly across H5 vintages.
    """
    if not (0 < h1_allocation < 1 and 0 < entry_fraction <= exit_fraction <= 1 and holding_days >= 1):
        raise ValueError("invalid sleeve allocation or rank-buffer parameters")
    source = pl.read_parquet(predictions).with_columns(
        pl.col("trade_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date)
    ).filter(
        pl.col("execution_date").is_not_null() & pl.col("pred_h1").is_finite() & pl.col("pred_h5").is_finite() & ~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES)
    )
    by_date = {
        (key[0] if isinstance(key, tuple) else key): frame
        for key, frame in source.partition_by("trade_date", as_dict=True).items()
    }

    def buffered_members(frame: pl.DataFrame, score: str, old: set[str]) -> set[str]:
        ordered = frame.sort(score, descending=True)
        enter_count = max(1, int(np.ceil(ordered.height * entry_fraction)))
        exit_count = max(enter_count, int(np.ceil(ordered.height * exit_fraction)))
        entrants = set(ordered.head(enter_count).get_column("ts_code").to_list())
        eligible = set(ordered.head(exit_count).get_column("ts_code").to_list())
        return entrants | (old & eligible)

    h1_members: set[str] = set()
    h5_members: list[set[str]] = [set() for _ in range(holding_days)]
    h5_allocation = 1.0 - h1_allocation
    rows: list[dict[str, object]] = []
    h1_counts: list[int] = []
    h5_counts: list[int] = []
    for index, signal_date in enumerate(sorted(by_date)):
        frame = by_date[signal_date]
        available = set(frame.get_column("ts_code").to_list())
        h5_members = [vintage & available for vintage in h5_members]
        h1_members = buffered_members(frame, "pred_h1", h1_members)
        sleeve = index % holding_days
        h5_members[sleeve] = buffered_members(frame, "pred_h5", h5_members[sleeve])
        aggregate: dict[str, float] = {}
        if h1_members:
            h1_weight = h1_allocation / len(h1_members)
            for code in h1_members:
                aggregate[code] = aggregate.get(code, 0.0) + h1_weight
        for vintage in h5_members:
            if vintage:
                h5_weight = h5_allocation / holding_days / len(vintage)
                for code in vintage:
                    aggregate[code] = aggregate.get(code, 0.0) + h5_weight
        execution_date = frame.get_column("execution_date")[0]
        for code, weight in sorted(aggregate.items()):
            rows.append({"trade_date": signal_date, "execution_date": execution_date, "ts_code": code, "target_weight": weight, "optimizer": "staggered_h1_h5_rank_buffer"})
        h1_counts.append(len(h1_members))
        h5_counts.append(sum(len(vintage) for vintage in h5_members))
    result = pl.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    return {"output": str(output), "days": len(by_date), "rows": result.height, "h1_allocation": h1_allocation, "h5_allocation": h5_allocation, "entry_fraction": entry_fraction, "exit_fraction": exit_fraction, "holding_days": holding_days, "mean_h1_names": float(np.mean(h1_counts)), "mean_h5_names_across_sleeves": float(np.mean(h5_counts))}


def backtest_targets(catalog: Path, target_weights: Path, output: Path, buy_bps: float = 2.1, sell_bps: float = 7.1, initial_capital: float = 10_000_000.0, lot_size: int = 100, return_basis: str = "qfq", rebalance_band: float = 0.0) -> dict:
    """Open-to-open backtest using real-price lots and qfq or raw-price marking."""
    if initial_capital <= 0 or lot_size <= 0 or return_basis not in {"qfq", "raw"} or not 0 <= rebalance_band < 1:
        raise ValueError("initial_capital and lot_size must be positive; return_basis must be qfq or raw")
    output.mkdir(parents=True, exist_ok=True)
    targets = pl.read_parquet(target_weights).with_columns(pl.col("execution_date").cast(pl.Date)).filter(~pl.col("ts_code").is_in(INFEASIBLE_EXECUTION_CODES))
    execution_dates = sorted(targets["execution_date"].drop_nulls().unique().to_list())
    if len(execution_dates) < 2: raise ValueError("Need at least two execution dates")
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        prices = pl.from_arrow(conn.execute(f"SELECT trade_date, ts_code, open AS raw_open, qfq_open FROM daily_qfq WHERE trade_date BETWEEN DATE '{execution_dates[0]}' AND DATE '{execution_dates[-1]}' AND open > 0 AND qfq_open > 0").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
        benchmark = pl.from_arrow(conn.execute(f"SELECT trade_date, open FROM index_daily WHERE index_code='000905.SH' AND trade_date BETWEEN DATE '{execution_dates[0]}' AND DATE '{execution_dates[-1]}' AND open > 0").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    by_date = {(day[0] if isinstance(day, tuple) else day): frame for day, frame in targets.partition_by("execution_date", as_dict=True).items()}
    price = {(day, code): (raw, qfq / raw) for day, code, raw, qfq in prices.select("trade_date", "ts_code", "raw_open", "qfq_open").iter_rows()}
    benchmark_price = dict(benchmark.select("trade_date", "open").iter_rows())
    rows=[]; positions=[]; executions=[]; shares: dict[str, float] = {}; factors: dict[str, float] = {}; last_value: dict[str, float] = {}; cash=initial_capital; nav=1.0; bench_nav=1.0; previous_ending_value=initial_capital
    for i, day in enumerate(execution_dates[:-1]):
        next_day=execution_dates[i+1]; frame=by_date[day]
        quote = {code: price[(day, code)] for code in set(shares) | set(frame["ts_code"].to_list()) if (day, code) in price}
        for code, quantity in list(shares.items()):
            if return_basis == "qfq" and code in quote and code in factors:
                shares[code] = quantity * quote[code][1] / factors[code]
                factors[code] = quote[code][1]
        equity = cash + sum(quantity * quote[code][0] if code in quote else last_value.get(code, 0.0) for code, quantity in shares.items())
        if equity <= 0: raise RuntimeError(f"non-positive equity on {day}")
        targets = {code: float(weight) for code, weight in frame.select("ts_code", "target_weight").iter_rows() if weight > 0 and code in quote}
        desired = {code: float(np.floor(equity * weight / quote[code][0] / lot_size) * lot_size) for code, weight in targets.items()}
        # The band applies only to continuing positions, never entries/exits.
        for code in desired.keys() & shares.keys():
            if abs(shares[code] * quote[code][0] / equity - targets[code]) <= rebalance_band:
                desired[code] = shares[code]
        # Sells are financed first.  Quantities produced by a corporate action
        # may be fractional; they remain sellable even though new buys are lots.
        sold = 0.0
        for code, quantity in list(shares.items()):
            if code not in quote:
                continue
            delta = max(quantity - desired.get(code, 0.0), 0.0)
            if delta:
                value = delta * quote[code][0]; cash += value * (1 - sell_bps / 10_000); sold += value
                shares[code] = quantity - delta
                executions.append({"execution_date": day, "ts_code": code, "side": "sell", "raw_open": quote[code][0], "shares": delta, "notional": value})
        buys = {code: float(np.floor(max(desired[code] - shares.get(code, 0.0), 0.0) / lot_size) * lot_size) for code in desired}
        buy_cost = sum(quantity * quote[code][0] * (1 + buy_bps / 10_000) for code, quantity in buys.items())
        if buy_cost > cash and buy_cost > 0:
            scale = cash / buy_cost
            buys = {code: float(np.floor(quantity * scale / lot_size) * lot_size) for code, quantity in buys.items()}
            buy_cost = sum(quantity * quote[code][0] * (1 + buy_bps / 10_000) for code, quantity in buys.items())
        bought = 0.0
        for code, quantity in buys.items():
            if quantity:
                value = quantity * quote[code][0]; shares[code] = shares.get(code, 0.0) + quantity
                cash -= value * (1 + buy_bps / 10_000); bought += value
                executions.append({"execution_date": day, "ts_code": code, "side": "buy", "raw_open": quote[code][0], "shares": quantity, "notional": value})
        shares = {code: quantity for code, quantity in shares.items() if quantity > 1e-10}
        for code, quantity in shares.items():
            if code in quote:
                factors[code] = quote[code][1]
                last_value[code] = quantity * quote[code][0]
                positions.append({"execution_date": day, "ts_code": code, "raw_open": quote[code][0], "qfq_ratio": quote[code][1], "shares": quantity, "market_value": last_value[code], "target_weight": targets.get(code, 0.0), "realized_weight": last_value[code] / equity})
        transaction_cost = bought * buy_bps / 10_000 + sold * sell_bps / 10_000
        # qfq/raw is the quantity adjustment needed to keep actual-share value
        # economically continuous through splits, dividends, and rights issues.
        ending_value = cash
        for code, quantity in shares.items():
            current_quote, next_quote = quote.get(code), price.get((next_day, code))
            if current_quote and next_quote:
                adjusted_quantity = quantity * next_quote[1] / current_quote[1] if return_basis == "qfq" else quantity
                ending_value += adjusted_quantity * next_quote[0]
            else:
                ending_value += last_value.get(code, quantity * current_quote[0] if current_quote else 0.0)
        p0,p1=benchmark_price.get(day),benchmark_price.get(next_day)
        bench=p1/p0-1 if p0 and p1 else 0.0
        # Returns must chain from the preceding end-of-day portfolio value.  The
        # tradable pre-trade value may differ around corporate actions because
        # qfq share-equivalence is refreshed at the open; using it as a return
        # denominator made compounded daily NAV disagree with the cash ledger.
        gross=(ending_value - previous_ending_value + transaction_cost) / previous_ending_value
        net=ending_value / previous_ending_value - 1
        nav=ending_value / initial_capital; bench_nav*=1+bench
        rows.append({"execution_date":day,"next_execution_date":next_day,"gross_return":gross,"transaction_cost":transaction_cost / previous_ending_value,"net_return":net,"csi500_return":bench,"active_return":net-bench,"buy_turnover":bought / equity,"sell_turnover":sold / equity,"nav":nav,"csi500_nav":bench_nav,"holding_count":len(shares),"cash":cash,"cash_weight":cash / equity,"equity":ending_value})
        previous_ending_value=ending_value
    daily=pl.DataFrame(rows); daily.write_parquet(output/"portfolio_daily.parquet",compression="zstd")
    pl.DataFrame(positions).write_parquet(output/"executed_positions.parquet", compression="zstd")
    pl.DataFrame(executions).write_parquet(output/"executions.parquet", compression="zstd")
    net=daily["net_return"].to_numpy(); active=daily["active_return"].to_numpy(); years=len(net)/252
    ann_return=nav**(1/years)-1 if years else 0.0; ann_vol=float(np.std(net,ddof=1)*np.sqrt(252)); active_vol=float(np.std(active,ddof=1)*np.sqrt(252)); running=np.maximum.accumulate(daily["nav"].to_numpy()); max_dd=float(np.min(daily["nav"].to_numpy()/running-1))
    summary={"days":len(rows),"initial_capital":initial_capital,"lot_size":lot_size,"return_basis":return_basis,"gross_total_return":float(np.prod(1+daily["gross_return"].to_numpy())-1),"net_total_return":nav-1,"csi500_total_return":bench_nav-1,"annualized_return":ann_return,"annualized_volatility":ann_vol,"sharpe":ann_return/ann_vol if ann_vol else None,"information_ratio":float(np.mean(active)/np.std(active,ddof=1)*np.sqrt(252)) if np.std(active,ddof=1) else None,"max_drawdown":max_dd,"average_buy_turnover":float(daily["buy_turnover"].mean()),"average_sell_turnover":float(daily["sell_turnover"].mean()),"average_cash_weight":float(daily["cash_weight"].mean()),"final_cash":float(cash),"buy_cost_bps":buy_bps,"sell_cost_bps":sell_bps,"total_transaction_cost":float(daily["transaction_cost"].sum())}
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
    gross_nav=np.cumprod(1+gross); net_nav=frame["nav"].to_numpy() if "nav" in frame.columns else np.cumprod(1+net); csi_nav=np.cumprod(1+csi); fee=[]; nav=1.0; paid=0.0
    for ret, fee_rate in zip(gross,cost):
        paid += nav*fee_rate; nav *= 1+ret-fee_rate; fee.append(paid)
    drawdown=net_nav/np.maximum.accumulate(net_nav)-1
    years=len(net)/252
    def ann(value: np.ndarray) -> float: return float(np.prod(1+value)**(1/years)-1) if years else 0.0
    def vol(value: np.ndarray) -> float: return float(np.std(value,ddof=1)*np.sqrt(252)) if len(value)>1 else 0.0
    net_excess_wealth = float(net_nav[-1] / csi_nav[-1] - 1)
    net_ann = float(net_nav[-1] ** (1 / years) - 1) if years else 0.0
    metrics={"days":len(frame),"start":str(frame["execution_date"][0]),"end":str(frame["execution_date"][-1]),"gross_total_return":float(gross_nav[-1]-1),"net_total_return":float(net_nav[-1]-1),"csi500_total_return":float(csi_nav[-1]-1),"net_excess_wealth_vs_csi500":net_excess_wealth,"gross_annualized_return":ann(gross),"net_annualized_return":net_ann,"csi500_annualized_return":ann(csi),"net_annualized_excess_vs_csi500":float((net_nav[-1] / csi_nav[-1]) ** (1 / years) - 1) if years else 0.0,"average_daily_active_return_bps":float(np.mean(net-csi)*10_000),"annualized_tracking_error":vol(net-csi),"net_sharpe":net_ann/vol(net) if vol(net) else None,"information_ratio":float(np.mean(net-csi)/np.std(net-csi,ddof=1)*np.sqrt(252)) if np.std(net-csi,ddof=1) else None,"max_drawdown":float(drawdown.min()),"average_buy_turnover":float(frame["buy_turnover"].mean()),"average_sell_turnover":float(frame["sell_turnover"].mean()),"average_fee_bps":float(np.mean(cost)*10_000),"cumulative_fee_paid_on_initial_nav":float(fee[-1]),"average_holding_count":float(frame["holding_count"].mean())}
    holdings = summarize_holdings(target_weights) if target_weights else None
    dates=frame["execution_date"].to_list()
    plt.style.use("seaborn-v0_8-whitegrid")
    gross_net_gap = gross_nav - net_nav
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[3,1]}); axes[0].plot(dates,gross_nav,label="Gross NAV",lw=1.8,color="#457b9d"); axes[0].plot(dates,net_nav,label="Net NAV",lw=1.8,color="#e76f51"); axes[0].plot(dates,csi_nav,label="CSI500 NAV",lw=1.5,color="#6b7280"); axes[0].fill_between(dates,net_nav,gross_nav,color="#e9c46a",alpha=.35,label="Cost drag"); axes[0].set_title(title+" — NAV and transaction-cost drag"); axes[0].set_ylabel("Initial NAV = 1"); axes[0].legend(ncol=4,fontsize=9); axes[1].plot(dates,gross_net_gap,label="Gross − Net NAV",lw=1.5,color="#e9c46a"); axes[1].plot(dates,fee,label="Cumulative fee paid",lw=1.2,ls="--",color="#9b5de5"); axes[1].set_ylabel("Initial NAV"); axes[1].legend(ncol=2,fontsize=9); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output/"nav_and_fee.png",dpi=160); plt.close(fig)
    excess_nav = net_nav - csi_nav
    cumulative_active = np.cumsum(net - csi)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[2,1]}); axes[0].plot(dates,excess_nav,color="#2a9d8f",lw=1.8,label="Net NAV − CSI500 NAV"); axes[0].fill_between(dates,excess_nav,0,color="#2a9d8f",alpha=.18); axes[0].axhline(0,color="#6b7280",lw=.8); axes[0].set_ylabel("Excess NAV (initial NAV)"); axes[0].set_title(title+" — CSI500 excess return / NAV"); axes[0].legend(fontsize=9); axes[1].plot(dates,cumulative_active,color="#457b9d",lw=1.3,label="Cumulative daily active return"); axes[1].axhline(0,color="#6b7280",lw=.8); axes[1].set_ylabel("Sum of active returns"); axes[1].legend(fontsize=9); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output/"excess_performance.png",dpi=160); plt.close(fig)
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
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#18212f}}table{{border-collapse:collapse}}th,td{{padding:7px 12px;border:1px solid #d9e1ea;text-align:left}}img{{display:block;max-width:1100px;width:100%;margin:20px 0}}</style></head><body><h1>{html.escape(title)}</h1><p>Return basis: T+1 open to T+2 open; benchmark: CSI500; costs use the portfolio daily ledger.</p><h2>Performance and CSI500 excess</h2><table>{table}</table>{holdings_table}<img src='nav_and_fee.png'><img src='excess_performance.png'><img src='drawdown_turnover_fee.png'></body></html>"""
    report = {**metrics, "holdings": holdings} if holdings else metrics
    (output/"report.html").write_text(page,encoding="utf-8"); (output/"report_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return {"output":str(output),"report":str(output/"report.html"),**report}


def main(argv: list[str] | None = None) -> None:
    p=argparse.ArgumentParser(prog="quant-predict"); sub=p.add_subparsers(dest="command",required=True)
    def common(x): x.add_argument("--catalog",type=Path,required=True); x.add_argument("--start"); x.add_argument("--end")
    b=sub.add_parser("build-features"); common(b); b.add_argument("--factor-root",type=Path,required=True); b.add_argument("--feature-root",type=Path,required=True); b.add_argument("--factor-ids-file",type=Path); b.add_argument("--all-factor-artifacts",action="store_true"); b.add_argument("--no-daily-factors",action="store_true"); b.add_argument("--minute-factor-dataset",type=Path,action="append"); b.add_argument("--minute-factor-ids-file",type=Path,action="append"); b.add_argument("--replace",action="store_true")
    r=sub.add_parser("run-oos"); common(r); r.add_argument("--oos-start"); r.add_argument("--feature-root",type=Path,required=True); r.add_argument("--output",type=Path,required=True); r.add_argument("--models",nargs="+",choices=["lgbm","mlp","transformer"],default=["lgbm"]); r.add_argument("--refit-months",type=int,default=3); r.add_argument("--neural-epochs",type=int,default=20); r.add_argument("--neural-max-samples",type=int,default=60_000); r.add_argument("--max-features",type=int,default=40); r.add_argument("--min-coverage",type=float,default=.85); r.add_argument("--min-abs-icir",type=float,default=.5); r.add_argument("--max-abs-correlation",type=float,default=.90); r.add_argument("--use-all-features",action="store_true",help="train with every feature meeting the coverage and variance requirements")
    d=sub.add_parser("optimize-dual-alpha"); d.add_argument("--predictions",type=Path,required=True); d.add_argument("--output",type=Path,required=True); d.add_argument("--h1-weight",type=float,default=.5); d.add_argument("--turnover-cap",type=float,default=.30); d.add_argument("--temperature",type=float,default=2.0); d.add_argument("--max-weight",type=float,default=.10); d.add_argument("--min-weight",type=float,default=0.0); d.add_argument("--top-fraction",type=float,default=1.0); d.add_argument("--weighting",choices=["softmax","equal"],default="softmax")
    s=sub.add_parser("staggered-top-quantile"); s.add_argument("--predictions",type=Path,required=True); s.add_argument("--output",type=Path,required=True); s.add_argument("--top-fraction",type=float,default=.10); s.add_argument("--holding-days",type=int,default=5)
    q=sub.add_parser("staggered-dual-hysteresis"); q.add_argument("--predictions",type=Path,required=True); q.add_argument("--output",type=Path,required=True); q.add_argument("--h1-allocation",type=float,default=.20); q.add_argument("--entry-fraction",type=float,default=.08); q.add_argument("--exit-fraction",type=float,default=.12); q.add_argument("--holding-days",type=int,default=5)
    bt=sub.add_parser("backtest-portfolio"); bt.add_argument("--catalog",type=Path,required=True); bt.add_argument("--target-weights",type=Path,required=True); bt.add_argument("--output",type=Path,required=True); bt.add_argument("--buy-bps",type=float,default=2.1); bt.add_argument("--sell-bps",type=float,default=7.1); bt.add_argument("--initial-capital",type=float,default=10_000_000); bt.add_argument("--lot-size",type=int,default=100); bt.add_argument("--return-basis",choices=["qfq","raw"],default="qfq")
    bp=sub.add_parser("backtest-policy", help="real-holdings limited-replacement daily policy")
    bp.add_argument("--catalog",type=Path,required=True); bp.add_argument("--predictions",type=Path,required=True); bp.add_argument("--output",type=Path,required=True)
    bp.add_argument("--target-holdings",type=int,default=80); bp.add_argument("--entry-rank",type=int,default=80); bp.add_argument("--exit-rank",type=int,default=96); bp.add_argument("--max-daily-replacements",type=int,default=5)
    bp.add_argument("--max-weight",type=float,default=.03); bp.add_argument("--rebalance-to-weight",type=float,default=.028); bp.add_argument("--min-new-weight",type=float,default=.005); bp.add_argument("--cash-reserve",type=float,default=.02)
    bp.add_argument("--daily-buy-budget",type=float,default=.10); bp.add_argument("--daily-sell-budget",type=float,default=.10); bp.add_argument("--h1-weight",type=float,default=.5); bp.add_argument("--entry-sizing",choices=["equal","rank_tilt"],default="equal"); bp.add_argument("--rank-tilt",type=float,default=.15); bp.add_argument("--initial-capital",type=float,default=10_000_000); bp.add_argument("--lot-size",type=int,default=100); bp.add_argument("--buy-bps",type=float,default=2.1); bp.add_argument("--sell-bps",type=float,default=7.1); bp.add_argument("--no-report",action="store_true")
    rp=sub.add_parser("render-backtest-report"); rp.add_argument("--portfolio-daily",type=Path,required=True); rp.add_argument("--output",type=Path,required=True); rp.add_argument("--title",default="Portfolio backtest"); rp.add_argument("--target-weights",type=Path)
    ro=sub.add_parser("research-oos", help="fixed-factor, leakage-safe rolling model comparison")
    ro.add_argument("--config",type=Path,required=True); ro.add_argument("--start"); ro.add_argument("--end"); ro.add_argument("--skip-backtests",action="store_true")
    args=p.parse_args(argv)
    if args.command=="build-features":
        if sum(bool(value) for value in (args.factor_ids_file, args.all_factor_artifacts, args.no_daily_factors)) > 1: raise ValueError("choose one daily-factor source: factor ids file, all artifacts, or no daily factors")
        daily_ids = () if args.no_daily_factors else (all_factor_ids(args.factor_root) if args.all_factor_artifacts else read_factor_ids(args.factor_ids_file))
        minute_datasets = args.minute_factor_dataset or []
        minute_id_files = args.minute_factor_ids_file or []
        if len(minute_datasets) != len(minute_id_files): raise ValueError("provide one --minute-factor-ids-file for each --minute-factor-dataset")
        minute_sources = tuple((dataset, read_factor_ids(ids_file)) for dataset, ids_file in zip(minute_datasets, minute_id_files))
        result=build_features(args.catalog,args.factor_root,args.feature_root,args.start,args.end,args.replace,daily_ids,minute_sources)
    elif args.command=="run-oos": result=run_oos(args.catalog,args.feature_root,args.output,args.start,args.end,ensemble=EnsembleSettings(tuple(args.models),args.refit_months,args.neural_epochs,args.neural_max_samples),selection=SelectionSettings(args.min_coverage,args.min_abs_icir,args.max_features,args.max_abs_correlation,use_all_features=args.use_all_features),oos_start=args.oos_start)
    elif args.command=="optimize-dual-alpha": result=optimize_dual_alpha_targets(args.predictions,args.output,args.h1_weight,args.turnover_cap,args.temperature,args.max_weight,args.min_weight,args.top_fraction,args.weighting)
    elif args.command=="staggered-top-quantile": result=staggered_top_quantile_targets(args.predictions,args.output,args.top_fraction,args.holding_days)
    elif args.command=="staggered-dual-hysteresis": result=staggered_dual_hysteresis_targets(args.predictions,args.output,args.h1_allocation,args.entry_fraction,args.exit_fraction,args.holding_days)
    elif args.command=="backtest-portfolio": result=backtest_targets(args.catalog,args.target_weights,args.output,args.buy_bps,args.sell_bps,args.initial_capital,args.lot_size,args.return_basis)
    elif args.command=="backtest-policy":
        from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
        config=LimitedReplacementConfig(target_holdings=args.target_holdings,entry_rank=args.entry_rank,exit_rank=args.exit_rank,max_daily_replacements=args.max_daily_replacements,max_weight=args.max_weight,rebalance_to_weight=args.rebalance_to_weight,min_new_weight=args.min_new_weight,cash_reserve=args.cash_reserve,daily_buy_budget=args.daily_buy_budget,daily_sell_budget=args.daily_sell_budget,h1_weight=args.h1_weight,entry_sizing=args.entry_sizing,rank_tilt=args.rank_tilt,initial_capital=args.initial_capital,lot_size=args.lot_size,buy_bps=args.buy_bps,sell_bps=args.sell_bps)
        result=run_limited_replacement_policy(args.catalog,args.predictions,args.output,config)
        if not args.no_report:
            result["charged_report"]=render_backtest_report(args.output/"portfolio_daily.parquet",args.output/"report","Limited-replacement v2 — charged account")
            result["zero_cost_report"]=render_backtest_report(args.output/"zero_cost"/"portfolio_daily.parquet",args.output/"zero_cost"/"report","Limited-replacement v2 — independent zero-cost account")
    elif args.command=="research-oos":
        from a_share_data.research_oos import run_research_oos
        result=run_research_oos(args.config,args.start,args.end,args.skip_backtests)
    else: result=render_backtest_report(args.portfolio_daily,args.output,args.title,args.target_weights)
    print(json.dumps(result,ensure_ascii=False,default=str))

if __name__ == "__main__": main()
