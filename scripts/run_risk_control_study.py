#!/usr/bin/env python3
"""Compare the limited-replacement policy with risk-neutralized alpha ranks.

This keeps the trading rule fixed (CSI500, Top100, three replacements/day,
equal entries and the same costs).  Each day's blended LGBM alpha is projected
off point-in-time style exposures, optionally plus static industry dummies.
It is a selection-risk control, not a claim of exact portfolio neutrality.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "A_stock_database/lake/catalog/a_share.duckdb"
PREDICTIONS = ROOT / "results/predict/research-oos-daily60-minute45-v2/predictions/lgbm_default105.parquet"
EXPOSURES = ROOT / "results/predict/risk-attribution-csi500-top100-swap3-with-size/exposures.parquet"
OUT = ROOT / "results/predict/risk-control-study"
STYLE = ["size", "beta_60", "residual_vol_60", "momentum_20", "liquidity_log_amount_20", "turnover_20"]


def stats(frame: pl.DataFrame) -> dict[str, float]:
    r = frame["net_return"].to_numpy(); b = frame["csi500_return"].to_numpy()
    nav = np.cumprod(1 + r); benchmark = np.cumprod(1 + b)
    return {"days": len(r), "net_return": float(nav[-1] - 1), "annual_return": float(nav[-1] ** (252 / len(r)) - 1),
            "annual_excess": float((nav[-1] / benchmark[-1]) ** (252 / len(r)) - 1),
            "max_drawdown": float((nav / np.maximum.accumulate(np.r_[1., nav])[1:] - 1).min()),
            "information_ratio": float((r - b).mean() / (r - b).std(ddof=1) * np.sqrt(252)),
            "average_buy_turnover": float(frame["buy_turnover"].mean())}


def residualize(source: pl.DataFrame, controls: list[str], use_industry: bool) -> pl.DataFrame:
    """Project a daily standardized H1/H5 blend off disclosed risk controls."""
    rows: list[pl.DataFrame] = []
    for key, frame in source.partition_by("trade_date", as_dict=True).items():
        frame = frame.sort("ts_code")
        y1, y5 = frame["pred_h1"].to_numpy(), frame["pred_h5"].to_numpy()
        def z(x: np.ndarray) -> np.ndarray:
            sd = x.std(); return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)
        y = .5 * z(y1) + .5 * z(y5)
        values = [np.ones(frame.height)]
        values.extend(frame[name].to_numpy() for name in controls)
        if use_industry:
            labels = frame["industry"].fill_null("Unknown").to_numpy()
            # Dropping one dummy keeps the regression full rank with intercept.
            for label in sorted(set(labels))[:-1]: values.append((labels == label).astype(float))
        x = np.column_stack(values)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        residual = np.full(frame.height, np.nan)
        if valid.sum() >= x.shape[1] + 10:
            residual[valid] = y[valid] - x[valid] @ np.linalg.lstsq(x[valid], y[valid], rcond=None)[0]
        rows.append(frame.with_columns(pl.Series("pred_h1", residual), pl.Series("pred_h5", residual)))
    return pl.concat(rows).filter(pl.col("pred_h1").is_finite()).sort(["trade_date", "ts_code"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CATALOG), read_only=True)
    try:
        universe = pl.from_arrow(con.execute("SELECT DISTINCT trade_date,ts_code FROM index_trading_universe WHERE index_code='000905.SH'").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        con.close()
    pred = pl.read_parquet(PREDICTIONS).with_columns(pl.col("trade_date").cast(pl.Date)).join(universe, on=["trade_date", "ts_code"], how="semi")
    exp = pl.read_parquet(EXPOSURES).select("trade_date", "ts_code", "industry", *STYLE)
    joined = pred.join(exp, on=["trade_date", "ts_code"], how="inner")
    if joined.height != pred.height:
        raise RuntimeError(f"missing risk data for {pred.height - joined.height} prediction rows")
    eligible = joined.filter(*[pl.col(name).is_finite() for name in STYLE])
    config = LimitedReplacementConfig(target_holdings=100, entry_rank=100, exit_rank=120, max_daily_replacements=3)
    variants = {"baseline": ([], False), "style_neutral": (STYLE, False), "style_industry_neutral": (STYLE, True)}
    report = {"protocol": {"predictions": str(PREDICTIONS), "prediction_sha256": hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(), "universe": "CSI500 historical constituents", "risk_input": str(EXPOSURES), "risk_control": "daily OLS alpha residual; style z-scores and optional industry dummies", "policy": asdict(config)}}
    results = []
    for name, (controls, industries) in variants.items():
        target = OUT / name
        # Every variant uses the same risk-observable stock/date panel.  This
        # prevents missing vendor share data from being mistaken for an effect
        # of the risk control itself.
        if name == "baseline": transformed = eligible.select(pred.columns)
        else: transformed = residualize(eligible, controls, industries).select(pred.columns)
        transformed.write_parquet(target.with_suffix(".predictions.parquet"), compression="zstd")
        run_limited_replacement_policy(CATALOG, target.with_suffix(".predictions.parquet"), target, config)
        row = {"name": name, "style_controls": controls, "industry_controls": industries,
               "prediction_rows": transformed.height, "metrics": stats(pl.read_parquet(target / "portfolio_daily.parquet"))}
        results.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    report["variants"] = results
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
