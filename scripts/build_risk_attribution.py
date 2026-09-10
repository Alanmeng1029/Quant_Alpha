#!/usr/bin/env python3
"""Build point-in-time risk proxies and attribute active portfolio return.

This is a transparent Barra-style proxy, not a licensed Barra model.  It uses
only information available before each exposure date.  An optional daily input
with ``trade_date``, ``ts_code`` and ``float_mv`` or ``total_mv`` adds size.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl


STYLE = ("size", "beta_60", "residual_vol_60", "momentum_20", "liquidity_log_amount_20", "turnover_20")


def zscore(values: np.ndarray) -> np.ndarray:
    good = np.isfinite(values)
    out = np.full(len(values), np.nan)
    if good.sum() < 2:
        return out
    x = values[good]
    lo, hi = np.quantile(x, [.01, .99])
    x = np.clip(x, lo, hi)
    sd = x.std(ddof=0)
    out[good] = 0 if sd <= 1e-12 else (x - x.mean()) / sd
    return out


def read_optional_size(path: Path | None) -> pl.DataFrame | None:
    if path is None:
        return None
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    value = next((name for name in ("float_mv", "total_mv", "float_market_cap", "market_cap") if name in frame.columns), None)
    if value is None or not {"trade_date", "ts_code"}.issubset(frame.columns):
        raise ValueError("daily risk input needs trade_date, ts_code, and one of float_mv/total_mv/float_market_cap/market_cap")
    return frame.select("trade_date", "ts_code", pl.col(value).cast(pl.Float64).alias("size_value")).with_columns(pl.col("trade_date").cast(pl.Date))


def build_exposures(catalog: Path, start: str, end: str, size: pl.DataFrame | None) -> tuple[pl.DataFrame, list[str]]:
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        raw = pl.from_arrow(conn.execute(f"""
            SELECT u.trade_date,u.ts_code,d.qfq_return,d.amount_cny,b.turn,i.industry
            FROM index_trading_universe u
            JOIN daily_qfq d USING(trade_date,ts_code)
            LEFT JOIN baostock_qfq_daily b USING(trade_date,ts_code)
            LEFT JOIN instruments i USING(ts_code)
            WHERE u.index_code='000905.SH'
              AND u.trade_date BETWEEN DATE '2018-01-01' AND DATE '{end}'
            ORDER BY u.trade_date,u.ts_code
        """).arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
        market = pl.from_arrow(conn.execute("SELECT trade_date,close_return FROM index_daily WHERE index_code='000905.SH' ORDER BY trade_date").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally:
        conn.close()
    table = raw.to_pandas(); dates = np.array(sorted(table.trade_date.unique())); codes = np.array(sorted(table.ts_code.unique()))
    date_i = {d: i for i, d in enumerate(dates)}; code_i = {c: i for i, c in enumerate(codes)}
    ret = np.full((len(dates), len(codes)), np.nan); amount = ret.copy(); turn = ret.copy(); industry = np.full(len(codes), "Unknown", dtype=object)
    for row in table.itertuples(index=False):
        i, j = date_i[row.trade_date], code_i[row.ts_code]; ret[i, j] = row.qfq_return; amount[i, j] = row.amount_cny; turn[i, j] = row.turn
        if row.industry is not None: industry[j] = row.industry
    mkt_map = dict(zip(market["trade_date"].to_list(), market["close_return"].to_list()))
    mkt = np.array([mkt_map.get(pd.Timestamp(d).date(), np.nan) for d in dates])
    fields = {name: np.full_like(ret, np.nan) for name in STYLE}
    for t in range(60, len(dates)):
        rr, mm = ret[t-60:t], mkt[t-60:t]
        for j in range(len(codes)):
            y = rr[:, j]; valid = np.isfinite(y) & np.isfinite(mm)
            if valid.sum() >= 40 and np.var(mm[valid]) > 1e-12:
                beta = np.cov(y[valid], mm[valid], ddof=1)[0, 1] / np.var(mm[valid], ddof=1)
                fields["beta_60"][t, j] = beta
                fields["residual_vol_60"][t, j] = np.std(y[valid] - beta * mm[valid], ddof=1)
            recent = ret[t-20:t, j]
            if np.isfinite(recent).sum() >= 15: fields["momentum_20"][t, j] = np.prod(1 + recent[np.isfinite(recent)]) - 1
            recent_amount = amount[t-20:t, j]
            if np.isfinite(recent_amount).sum() >= 15: fields["liquidity_log_amount_20"][t, j] = np.log(np.nanmean(recent_amount)) if np.nanmean(recent_amount) > 0 else np.nan
            recent_turn = turn[t-20:t, j]
            if np.isfinite(recent_turn).sum() >= 15: fields["turnover_20"][t, j] = np.nanmean(recent_turn)
    rows = []
    for t, d in enumerate(dates):
        if str(d) < start: continue
        for name in STYLE:
            fields[name][t] = zscore(fields[name][t])
        for j, c in enumerate(codes):
            row = {"trade_date": d, "ts_code": c, "industry": industry[j]}
            row.update({name: fields[name][t, j] for name in STYLE})
            rows.append(row)
    exposure = pl.from_dicts(rows).with_columns(pl.col("trade_date").cast(pl.Date))
    available = [name for name in STYLE if exposure.select(pl.col(name).is_finite().any()).item()]
    if size is not None:
        exposure = exposure.join(size, on=["trade_date", "ts_code"], how="left").with_columns(pl.col("size_value").log().alias("size")).drop("size_value")
        exposure = exposure.with_columns(pl.col("size").map_batches(lambda x: pl.Series(zscore(x.to_numpy())), return_dtype=pl.Float64).over("trade_date"))
        available = [name for name in STYLE if name != "size"] + ["size"]
    # Keep unavailable lookbacks in Arrow-null form for downstream consumers;
    # NaN is otherwise easy to accidentally aggregate as a real observation.
    exposure = exposure.with_columns(pl.col(pl.Float64).fill_nan(None))
    return exposure, available


def factor_returns(exposure: pl.DataFrame, catalog: Path, available: list[str]) -> pl.DataFrame:
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        returns = pl.from_arrow(conn.execute("SELECT trade_date,ts_code,qfq_return FROM daily_qfq WHERE qfq_return IS NOT NULL").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    panel = exposure.join(returns, on=["trade_date", "ts_code"], how="inner").drop_nulls([*available, "qfq_return"])
    panel = panel.filter(pl.col("qfq_return").is_finite(), *[pl.col(name).is_finite() for name in available])
    industries = sorted(panel["industry"].drop_nulls().unique().to_list()); base = industries[-1:]
    rows = []
    for day, group in panel.group_by("trade_date", maintain_order=True):
        d = day[0] if isinstance(day, tuple) else day
        if group.height < len(available) + 10: continue
        x = [np.ones(group.height), *[group[name].to_numpy() for name in available]]
        for name in industries:
            if name not in base: x.append((group["industry"].to_numpy() == name).astype(float))
        x = np.column_stack(x); y = group["qfq_return"].to_numpy(); coef = np.linalg.lstsq(x, y, rcond=None)[0]
        row = {"trade_date": d, "factor_return_market": float(coef[0])}; row.update({f"factor_return_{name}": float(coef[i + 1]) for i, name in enumerate(available)})
        row.update({f"factor_return_industry::{name}": float(coef[1 + len(available) + i]) for i, name in enumerate(industries) if name not in base})
        rows.append(row)
    return pl.from_dicts(rows).with_columns(pl.col("trade_date").cast(pl.Date))


def attribute(holdings: Path, exposure: pl.DataFrame, factor_ret: pl.DataFrame, catalog: Path, available: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    h = pl.read_parquet(holdings).select("execution_date", "ts_code", "weight").rename({"execution_date": "trade_date"}).with_columns(pl.col("trade_date").cast(pl.Date))
    conn = duckdb.connect(str(catalog), read_only=True)
    try:
        benchmark = pl.from_arrow(conn.execute("SELECT trade_date,ts_code FROM index_trading_universe WHERE index_code='000905.SH'").arrow()).with_columns(pl.col("trade_date").cast(pl.Date))
    finally: conn.close()
    benchmark_exposure = benchmark.join(exposure, on=["trade_date", "ts_code"], how="inner")
    bench = benchmark_exposure.group_by("trade_date").agg(*[pl.col(name).mean().alias(name) for name in available])
    port = h.join(exposure, on=["trade_date", "ts_code"], how="inner").group_by("trade_date").agg(*[(pl.col(name) * pl.col("weight")).sum().alias(name) for name in available], pl.col("industry").value_counts().alias("industry_weight"))
    active = port.join(bench, on="trade_date", suffix="_bench").join(factor_ret, on="trade_date", how="inner")
    for name in available:
        active = active.with_columns((pl.col(name) - pl.col(f"{name}_bench")).alias(f"active_{name}"), ((pl.col(name) - pl.col(f"{name}_bench")) * pl.col(f"factor_return_{name}")).alias(f"contribution_{name}"))
    industry_port = h.join(exposure.select("trade_date", "ts_code", "industry"), on=["trade_date", "ts_code"], how="inner").group_by("trade_date", "industry").agg(pl.col("weight").sum().alias("portfolio_weight"))
    industry_bench = benchmark_exposure.group_by("trade_date", "industry").len().with_columns((pl.col("len") / pl.col("len").sum().over("trade_date")).alias("benchmark_weight")).drop("len")
    industry = industry_port.join(industry_bench, on=["trade_date", "industry"], how="full", coalesce=True).fill_null(0).join(factor_ret, on="trade_date", how="left")
    industry_names = [name.removeprefix("factor_return_industry::") for name in factor_ret.columns if name.startswith("factor_return_industry::")]
    # A row belongs to exactly one industry.  Sum the conditional terms instead
    # of overwriting ``contribution`` once for every industry.
    industry_terms = [
        pl.when(pl.col("industry") == name)
        .then((pl.col("portfolio_weight") - pl.col("benchmark_weight")) * pl.col(f"factor_return_industry::{name}"))
        .otherwise(0.0)
        for name in industry_names
    ]
    industry = industry.with_columns(
        (pl.sum_horizontal(industry_terms) if industry_terms else pl.lit(0.0)).alias("contribution")
    )
    industry = industry.group_by("trade_date", "industry").agg(pl.col("portfolio_weight").first(), pl.col("benchmark_weight").first(), pl.col("contribution").sum()).with_columns((pl.col("portfolio_weight") - pl.col("benchmark_weight")).alias("active_weight"))
    summary = {"days": active.height, "available_style_factors": available,
               "style_valid_days": {name: active.select(pl.col(f"active_{name}").is_finite().sum()).item() for name in available},
               "mean_active_exposure": {name: active.select(pl.col(f"active_{name}").filter(pl.col(f"active_{name}").is_finite()).mean()).item() for name in available},
               "cumulative_style_contribution": {name: active.select(pl.col(f"contribution_{name}").filter(pl.col(f"contribution_{name}").is_finite()).sum()).item() for name in available},
               "top_industry_active_weights": industry.group_by("industry").agg(pl.col("active_weight").mean().alias("mean_active_weight"), pl.col("contribution").sum().alias("cumulative_contribution")).sort("mean_active_weight", descending=True).head(10).to_dicts()}
    return active, industry, summary


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--catalog", type=Path, required=True); p.add_argument("--holdings", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--start", default="2021-04-01"); p.add_argument("--end", default="2026-08-28"); p.add_argument("--daily-risk-input", type=Path); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); size = read_optional_size(args.daily_risk_input)
    exposures, available = build_exposures(args.catalog, args.start, args.end, size); exposures.write_parquet(args.output / "exposures.parquet", compression="zstd")
    factors = factor_returns(exposures, args.catalog, available); factors.write_parquet(args.output / "factor_returns.parquet", compression="zstd")
    attribution, industry, summary = attribute(args.holdings, exposures, factors, args.catalog, available); attribution.write_parquet(args.output / "attribution.parquet", compression="zstd"); industry.write_parquet(args.output / "industry_attribution.parquet", compression="zstd")
    summary.update({"method": "proxy model; close-to-close daily factor returns; exposures lag their input windows", "size_input": str(args.daily_risk_input) if size is not None else None, "limitation": "industry is static instruments.industry; no PIT industry history or commercial Barra covariance/specific risk"})
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    style_lines = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.5f}</td><td>{summary['cumulative_style_contribution'][k]:.5f}</td><td>{summary['style_valid_days'][k]}</td></tr>" for k, v in summary["mean_active_exposure"].items())
    industry_summary = industry.group_by("industry").agg(pl.col("active_weight").mean().alias("mean_active_weight"), pl.col("contribution").sum().alias("cumulative_contribution")).sort("cumulative_contribution", descending=True)
    industry_lines = "".join(f"<tr><td>{html.escape(str(r['industry']))}</td><td>{r['mean_active_weight']:.3%}</td><td>{r['cumulative_contribution']:.3%}</td></tr>" for r in industry_summary.head(15).to_dicts())
    (args.output / "report.html").write_text(f"""<!doctype html><html><meta charset='utf-8'><title>CSI500 风险代理归因</title><style>body{{font:15px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:1000px;margin:32px auto;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #d7dce5;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3fb}}.note{{color:#596579;line-height:1.6}}</style><body><h1>CSI500 Top100（换仓上限3）风险代理归因</h1><p>样本日数：{summary['days']}；规模输入：{'已提供' if size is not None else '缺失，尚未计算 Size'}。</p><h2>风格暴露与贡献</h2><table><tr><th>因子</th><th>平均主动暴露</th><th>累计日收益贡献</th><th>有效日</th></tr>{style_lines}</table><h2>行业主动暴露与贡献（前15）</h2><table><tr><th>行业</th><th>平均主动权重</th><th>累计日收益贡献</th></tr>{industry_lines}</table><p class='note'>方法：每日截面回归估计收盘到收盘的代理因子收益，再以策略相对 CSI500 等权的暴露计算贡献。它用于诊断风格和行业偏离，不等同于商业 Barra 模型，也不会逐笔复现开盘到开盘的交易账本收益。{html.escape(summary['limitation'])}</p></body></html>""", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__": main()
