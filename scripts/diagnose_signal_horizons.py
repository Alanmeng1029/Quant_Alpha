#!/usr/bin/env python3
"""Diagnose h1/h5 signal structure: horizon information, persistence, and z5-vs-smoothed-z1 equivalence.

Answers, per prediction file (lstm/lgbm/blend/hybrid):
  1. cross-sectional corr(z1, z5) per day          -> how much independent horizon info exists;
  2. IC decay of z1/z5/composite over k=1..10 days -> information horizon and specialization;
  3. rank persistence in trading time (lag 1/2/5/10) and top-100 churn -> natural turnover;
  4. corr(z5, EWMA(z1, half-life h))               -> whether h5 is just a smoothed h1;
  5. pooled daily OLS slope excess_k ~ z           -> z-score to expected-return calibration.

Forward returns follow the executable label convention: entry next trading day's
qfq open, exit k trading days later, excess versus CSI500 open-to-open.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

HORIZONS = tuple(range(1, 11))
EWMA_HALF_LIVES = (1.0, 2.0, 3.0, 5.0, 8.0, 13.0)
LAGS = (1, 2, 5, 10)
TOP_N = 100
INDEX_CODE = "000905.SH"


def _daily_spearman(frame: pl.DataFrame, x: str, y: str, min_n: int = 30) -> pl.DataFrame:
    """Per-trade-date Spearman correlation between two columns."""
    return (
        frame.drop_nulls(subset=[x, y])
        .group_by("trade_date")
        .agg(pl.corr(pl.col(x).rank(), pl.col(y).rank()).alias("corr"), pl.len().alias("n"))
        .filter(pl.col("n") >= min_n)
        .drop("n")
    )


def _mean_corr(daily: pl.DataFrame) -> dict[str, Any]:
    if daily.is_empty():
        return {"mean": None, "std": None, "days": 0}
    stats = daily.select(
        pl.col("corr").mean().alias("mean"), pl.col("corr").std().alias("std"), pl.len().alias("days")
    ).to_dicts()[0]
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in stats.items()}


def _forward_excess(catalog: Path, signals: pl.DataFrame) -> pl.DataFrame:
    """Executable k-day open-to-open excess returns for every (trade_date, ts_code) row."""
    start = signals["trade_date"].min()
    end = signals["trade_date"].max() + timedelta(days=60)
    start_sql, end_sql = f"'{start}'", f"'{end}'"
    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        calendar = pl.from_arrow(connection.execute(
            f"SELECT trade_date, row_number() over(order by trade_date) n FROM observed_calendar "
            f"WHERE is_observed_market_day AND trade_date BETWEEN {start_sql} AND {end_sql}"
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date), pl.col("n").cast(pl.Int32))
        codes = signals["ts_code"].unique().to_list()
        code_list = ",".join(f"'{code}'" for code in codes)
        opens = pl.from_arrow(connection.execute(
            f"SELECT ts_code, trade_date, qfq_open FROM daily_qfq WHERE ts_code IN ({code_list}) "
            f"AND trade_date BETWEEN {start_sql} AND {end_sql} AND qfq_open > 0"
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
        index_opens = pl.from_arrow(connection.execute(
            f"SELECT trade_date, open FROM index_daily WHERE index_code = '{INDEX_CODE}' "
            f"AND trade_date BETWEEN {start_sql} AND {end_sql}"
        ).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        connection.close()

    day_index = calendar.select("trade_date", pl.col("n").alias("day_index"))
    stock_open = (opens.join(day_index, on="trade_date", how="inner")
                  .select("ts_code", pl.col("day_index").alias("gidx"),
                          pl.col("qfq_open").cast(pl.Float64).alias("sopen")))
    index_open = (index_opens.join(day_index, on="trade_date", how="inner")
                  .select(pl.col("day_index").alias("gidx"),
                          pl.col("open").cast(pl.Float64).alias("iopen")))

    panel = (signals.select("trade_date", "ts_code").unique()
             .join(day_index, on="trade_date", how="inner")
             .with_columns((pl.col("day_index") + 1).alias("entry_gidx")))
    panel = (panel.join(stock_open.rename({"gidx": "entry_gidx", "sopen": "entry_sopen"}),
                        on=["ts_code", "entry_gidx"], how="left")
             .join(index_open.rename({"gidx": "entry_gidx", "iopen": "entry_iopen"}),
                   on="entry_gidx", how="left"))
    for horizon in HORIZONS:
        panel = panel.with_columns((pl.col("day_index") + 1 + horizon).alias("exit_gidx"))
        panel = (panel.join(stock_open.rename({"gidx": "exit_gidx", "sopen": f"sopen_{horizon}"}),
                            on=["ts_code", "exit_gidx"], how="left")
                 .join(index_open.rename({"gidx": "exit_gidx", "iopen": f"iopen_{horizon}"}),
                       on="exit_gidx", how="left").drop("exit_gidx"))
        panel = panel.with_columns(
            (pl.col(f"sopen_{horizon}") / pl.col("entry_sopen")
             - pl.col(f"iopen_{horizon}") / pl.col("entry_iopen")).alias(f"excess_{horizon}"))
    keep = ["trade_date", "ts_code"] + [f"excess_{horizon}" for horizon in HORIZONS]
    return panel.select(keep)


def _ewma(values: np.ndarray, half_life: float) -> np.ndarray:
    alpha = 1.0 - np.exp(np.log(0.5) / half_life)
    out = np.empty_like(values)
    state = np.nan
    for position, value in enumerate(values):
        if np.isnan(value):
            out[position] = state
            continue
        state = value if np.isnan(state) else alpha * value + (1.0 - alpha) * state
        out[position] = state
    return out


def _add_ewma_columns(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Per-stock EWMA of a column, sorted by trade date, one output column per half-life."""
    frame = frame.sort("ts_code", "trade_date")
    codes = frame["ts_code"].to_numpy()
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    segments = np.split(np.arange(frame.height), boundaries)
    values_all = frame[column].to_numpy()
    additions = {f"{column}_hl{half_life:g}": np.full(frame.height, np.nan) for half_life in EWMA_HALF_LIVES}
    for segment in segments:
        values = values_all[segment]
        for half_life in EWMA_HALF_LIVES:
            additions[f"{column}_hl{half_life:g}"][segment] = _ewma(values, half_life)
    return frame.with_columns([pl.Series(name, values) for name, values in additions.items()])


def _ols_slope(daily_frame: pl.DataFrame, x: str, y: str) -> pl.DataFrame:
    """Daily through-origin OLS slope of y on x (z-score to expected-return calibration)."""
    return daily_frame.drop_nulls(subset=[x, y]).group_by("trade_date").agg(
        ((pl.col(x) * pl.col(y)).sum() / (pl.col(x) * pl.col(x)).sum()).alias("slope"),
        pl.len().alias("n")).filter(pl.col("n") >= 30).drop("n")


def _top_churn(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    """Average daily exits from the top-N set under an unconstrained rank policy."""
    ranked = frame.drop_nulls(subset=[column]).select("trade_date", "ts_code", column).sort(
        column, descending=True)
    top = ranked.group_by("trade_date").agg(pl.col("ts_code").head(TOP_N).alias("codes"))
    sets = {row["trade_date"]: set(row["codes"]) for row in top.iter_rows(named=True)}
    dates = sorted(sets)
    churns = [len(sets[dates[i]] - sets[dates[i - 1]]) for i in range(1, len(dates))
              if sets[dates[i]] and sets[dates[i - 1]]]
    if not churns:
        return {"top_n": TOP_N, "avg_daily_exits": None, "days": 0}
    return {"top_n": TOP_N, "avg_daily_exits": round(float(np.mean(churns)), 2), "days": len(churns)}


def diagnose(predictions: dict[str, pl.DataFrame], panel: pl.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, frame in predictions.items():
        data = frame.join(panel, on=["trade_date", "ts_code"], how="inner").with_columns(
            ((pl.col("pred_h1") + pl.col("pred_h5")) / 2.0).alias("composite"))
        result: dict[str, Any] = {}

        # 1. z1 vs z5 cross-sectional correlation.
        result["corr_z1_z5"] = _mean_corr(_daily_spearman(data, "pred_h1", "pred_h5"))

        # 2. IC decay across horizons.
        result["ic_curve"] = [
            {"column": column, "horizon": horizon,
             **_mean_corr(_daily_spearman(data, column, f"excess_{horizon}"))}
            for column in ("pred_h1", "pred_h5", "composite") for horizon in HORIZONS]

        # 3. Rank persistence in trading time and top-N churn.
        dates = data["trade_date"].unique().sort()
        lag_frames = {lag: pl.DataFrame({"trade_date": dates, "prev_date": dates.shift(lag)})
                      for lag in LAGS}
        persistence = []
        for column in ("pred_h1", "pred_h5", "composite"):
            ranks = data.select("trade_date", "ts_code", column)
            for lag in LAGS:
                lagged = (ranks.join(lag_frames[lag], on="trade_date", how="inner").drop_nulls()
                          .join(ranks.rename({"trade_date": "prev_date", column: f"prev_{column}"}),
                                left_on=["ts_code", "prev_date"], right_on=["ts_code", "prev_date"],
                                how="inner"))
                persistence.append({"column": column, "lag_days": lag,
                                    **_mean_corr(_daily_spearman(lagged, column, f"prev_{column}"))})
        result["rank_persistence"] = persistence
        result["top_churn"] = {column: _top_churn(data, column)
                               for column in ("pred_h1", "pred_h5", "composite")}

        # 4. Is z5 an EWMA of z1?
        smoothed = _add_ewma_columns(data.select("trade_date", "ts_code", "pred_h1"), "pred_h1")
        data = data.join(smoothed, on=["trade_date", "ts_code"], how="inner")
        result["ewma_equivalence"] = [
            {"half_life": half_life,
             "corr_with_z5": _mean_corr(_daily_spearman(data, f"pred_h1_hl{half_life:g}", "pred_h5"))["mean"],
             "ic5": _mean_corr(_daily_spearman(data, f"pred_h1_hl{half_life:g}", "excess_5"))["mean"]}
            for half_life in EWMA_HALF_LIVES]
        result["ewma_reference"] = {
            "corr_z1_z5": result["corr_z1_z5"]["mean"],
            "ic5_of_z5": _mean_corr(_daily_spearman(data, "pred_h5", "excess_5"))["mean"],
            "ic5_of_z1": _mean_corr(_daily_spearman(data, "pred_h1", "excess_5"))["mean"]}

        # 5. z-score to expected-return calibration (bp per unit z).
        slope_h1 = _ols_slope(data, "pred_h1", "excess_1")
        slope_h5 = _ols_slope(data, "pred_h5", "excess_5")
        result["calibration_bp_per_z"] = {
            "h1_vs_excess1": round(1e4 * slope_h1["slope"].mean(), 2) if slope_h1.height else None,
            "h5_vs_excess5": round(1e4 * slope_h5["slope"].mean(), 2) if slope_h5.height else None}

        # 6. Yearly IC at native horizons.
        yearly = (_daily_spearman(data, "pred_h1", "excess_1")
                  .with_columns(pl.col("trade_date").dt.year().alias("year"))
                  .group_by("year").agg(pl.col("corr").mean().alias("ic1_of_h1")).sort("year"))
        yearly5 = (_daily_spearman(data, "pred_h5", "excess_5")
                   .with_columns(pl.col("trade_date").dt.year().alias("year"))
                   .group_by("year").agg(pl.col("corr").mean().alias("ic5_of_h5")).sort("year"))
        result["yearly_ic"] = yearly.join(yearly5, on="year", how="full", coalesce=True).sort("year").to_dicts()
        summary[name] = result
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    prediction_dir = args.prediction_dir.expanduser()
    predictions = {path.stem.removesuffix("_csi500"): pl.read_parquet(path)
                   for path in sorted(prediction_dir.glob("*_csi500.parquet"))}
    if not predictions:
        raise FileNotFoundError(f"no *_csi500.parquet under {prediction_dir}")
    union = pl.concat([frame.select("trade_date", "ts_code") for frame in predictions.values()]).unique()
    panel = _forward_excess(args.catalog.expanduser(), union)
    summary = diagnose(predictions, panel)
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prediction_dir": str(prediction_dir),
        "catalog": str(args.catalog.expanduser().resolve()),
        "rows": {name: frame.height for name, frame in predictions.items()},
        "dates": [str(union["trade_date"].min()), str(union["trade_date"].max())],
        "convention": "entry next trading day qfq open, exit k trading days later, excess vs CSI500 o2o",
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    frames = []
    for name, result in summary.items():
        frames.append(pl.DataFrame([{"model": name, **result["corr_z1_z5"], "seq": 0}]))
        frames.append(pl.DataFrame(result["ic_curve"]).with_columns(seq=pl.lit(1, dtype=pl.Int64), model=pl.lit(name)))
        frames.append(pl.DataFrame(result["rank_persistence"]).with_columns(seq=pl.lit(2, dtype=pl.Int64), model=pl.lit(name)))
        frames.append(pl.DataFrame(result["ewma_equivalence"]).with_columns(seq=pl.lit(3, dtype=pl.Int64), model=pl.lit(name)))
    pl.concat(frames, how="diagonal").write_csv(output / "metrics.csv")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path,
                        default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/weekly_comparison/predictions"))
    parser.add_argument("--catalog", type=Path,
                        default=Path("A_stock_database/lake/catalog/a_share.duckdb"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/horizon_diagnostics"))
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
