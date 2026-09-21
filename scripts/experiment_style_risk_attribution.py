#!/usr/bin/env python3
"""Build three CSI500 style factors and run four-factor portfolio attribution."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-alpha-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from a_share_data.risk_attribution import (
    MarketRiskConfig,
    estimate_cross_sectional_factor_returns,
    rolling_factor_risk_attribution,
    summarise_attribution,
)
from experiment_market_risk_attribution import DEFAULT_INPUTS, ROOT


FEATURES = ROOT / "A_stock_database/lake/derived/predict_features_o2o_daily60_minute45_v2"
MARKET_CAP = (
    ROOT
    / "A_stock_database/lake/derived/risk_inputs/csi500_akshare_monthly_float_mv"
    / "akshare_daily_market_cap.parquet"
)
PREDICTIONS = (
    ROOT
    / "results/predict/sequence-lstm-residual-daily60-minute45-v2/full"
    / "weekly_comparison/predictions/blend_50_50_csi500.parquet"
)
OUTPUT = ROOT / "results/risk/active_style_factor_v1"
EXPOSURES = ["size", "volatility", "liquidity"]
FACTORS = ["market", "size", "volatility", "liquidity"]


def _standardize_expression(column: str) -> pl.Expr:
    raw = pl.col(column)
    lower = raw.quantile(0.01).over("date")
    upper = raw.quantile(0.99).over("date")
    clipped = raw.clip(lower, upper)
    mean = clipped.mean().over("date")
    std = clipped.std().over("date")
    return ((clipped - mean) / std).alias(column)


def load_style_panel() -> pd.DataFrame:
    feature_files = str(FEATURES / "year=*/features.parquet")
    features = pl.scan_parquet(feature_files).select(
        pl.col("trade_date").alias("date"),
        "ts_code",
        "mf_realized_volatility",
        "mf_amihud_intraday_w20",
    )
    cap = pl.scan_parquet(MARKET_CAP).select(
        pl.col("trade_date").cast(pl.Date).alias("date"),
        "ts_code",
        "float_mv",
    )
    prediction = pl.scan_parquet(PREDICTIONS).select(
        pl.col("trade_date").alias("date"),
        "ts_code",
        pl.col("raw_h1").alias("future_excess_return"),
    )
    panel = (
        prediction.join(features, on=["date", "ts_code"], how="left")
        .join(cap, on=["date", "ts_code"], how="left")
        .with_columns(
            pl.col("float_mv").median().over("date").alias("daily_median_float_mv")
        )
        .with_columns(
            pl.col("float_mv").fill_null(pl.col("daily_median_float_mv")),
        )
        .with_columns(
            pl.col("float_mv").log().alias("size"),
            pl.col("mf_realized_volatility").clip(1e-12, None).log().alias("volatility"),
            (-pl.col("mf_amihud_intraday_w20").clip(1e-18, None).log()).alias("liquidity"),
            pl.col("float_mv").sqrt().alias("regression_weight"),
        )
        .with_columns(*[_standardize_expression(column) for column in EXPOSURES])
        .select("date", "ts_code", "future_excess_return", "regression_weight", *EXPOSURES)
        .collect()
    )
    return panel.to_pandas()


def load_portfolio(path: Path, factor_returns: pd.DataFrame) -> pd.DataFrame:
    portfolio = (
        pl.read_parquet(path)
        .select(
            pl.col("signal_date").alias("date"),
            (pl.col("gross_return") - pl.col("csi500_return")).alias("portfolio_return"),
            pl.col("csi500_return").alias("market_factor_return"),
        )
        .to_pandas()
    )
    result = portfolio.merge(factor_returns, on="date", how="inner", validate="one_to_one")
    return result.rename(
        columns={
            "market_factor_return": "market",
            "size_factor_return": "size",
            "volatility_factor_return": "volatility",
            "liquidity_factor_return": "liquidity",
        }
    )


def calculate_direct_exposures(
    strategy: str,
    holdings_path: Path,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compare portfolio style exposures with a float-cap weighted CSI500 proxy."""

    benchmark_source = panel[["date", "regression_weight", *EXPOSURES]].copy()
    benchmark_source["cap_weight"] = benchmark_source["regression_weight"] ** 2
    benchmark_rows = []
    for date, group in benchmark_source.groupby("date", sort=True):
        weights = group["cap_weight"].to_numpy(dtype=np.float64)
        row: dict[str, object] = {"date": date}
        for exposure in EXPOSURES:
            row[f"benchmark_{exposure}"] = float(
                np.average(group[exposure].to_numpy(dtype=np.float64), weights=weights)
            )
        benchmark_rows.append(row)
    benchmark = pd.DataFrame(benchmark_rows)

    holdings = (
        pl.read_parquet(holdings_path)
        .select(
            pl.col("signal_date").alias("date"),
            "ts_code",
            "weight",
        )
        .to_pandas()
    )
    joined = holdings.merge(
        panel[["date", "ts_code", *EXPOSURES]],
        on=["date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    rows = []
    for date, group in joined.groupby("date", sort=True):
        valid = group.dropna(subset=["weight", *EXPOSURES])
        total_weight = float(valid["weight"].sum())
        if total_weight <= 0.0:
            continue
        row: dict[str, object] = {
            "strategy": strategy,
            "date": date,
            "holding_rows": int(len(group)),
            "matched_holding_rows": int(len(valid)),
        }
        weights = valid["weight"].to_numpy(dtype=np.float64) / total_weight
        for exposure in EXPOSURES:
            row[f"portfolio_{exposure}"] = float(
                np.dot(weights, valid[exposure].to_numpy(dtype=np.float64))
            )
        rows.append(row)
    result = pd.DataFrame(rows).merge(benchmark, on="date", how="left", validate="one_to_one")
    for exposure in EXPOSURES:
        result[f"active_{exposure}"] = (
            result[f"portfolio_{exposure}"] - result[f"benchmark_{exposure}"]
        )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = MarketRiskConfig(lookback=252, half_life=60.0, forward_window=20, minimum_observations=126)
    panel = load_style_panel()
    factor_returns = estimate_cross_sectional_factor_returns(panel, EXPOSURES)
    factor_returns.to_csv(OUTPUT / "style_factor_returns.csv", index=False)

    daily_frames = []
    one_factor_daily_frames = []
    direct_exposure_frames = []
    summary_rows = []
    one_factor_summary_rows = []
    for strategy, path in DEFAULT_INPUTS.items():
        portfolio = load_portfolio(path, factor_returns)
        daily = rolling_factor_risk_attribution(portfolio, FACTORS, config)
        daily.insert(0, "strategy", strategy)
        daily_frames.append(daily)
        one_factor_daily = rolling_factor_risk_attribution(portfolio, ["market"], config)
        one_factor_daily.insert(0, "strategy", strategy)
        one_factor_daily_frames.append(one_factor_daily)
        one_factor_summary_rows.append(
            {"strategy": strategy, **summarise_attribution(one_factor_daily)}
        )
        summary = {"strategy": strategy, **summarise_attribution(daily)}
        for factor in FACTORS:
            summary[f"mean_beta_{factor}"] = float(daily[f"beta_{factor}"].mean())
            summary[f"mean_risk_share_{factor}"] = float(daily[f"risk_share_{factor}"].mean())
        direct = calculate_direct_exposures(strategy, path.parent / "holdings.parquet", panel)
        direct_exposure_frames.append(direct)
        for exposure in EXPOSURES:
            summary[f"mean_active_exposure_{exposure}"] = float(
                direct[f"active_{exposure}"].mean()
            )
            summary[f"std_active_exposure_{exposure}"] = float(
                direct[f"active_{exposure}"].std()
            )
        summary_rows.append(summary)

    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_one_factor_daily = pd.concat(one_factor_daily_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    one_factor_summary = pd.DataFrame(one_factor_summary_rows)
    pl.from_pandas(all_daily).write_parquet(OUTPUT / "daily_attribution.parquet")
    pl.from_pandas(all_one_factor_daily).write_parquet(
        OUTPUT / "one_factor_daily_attribution.parquet"
    )
    direct_exposures = pd.concat(direct_exposure_frames, ignore_index=True)
    direct_exposures.to_csv(OUTPUT / "direct_style_exposures.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    one_factor_summary.to_csv(OUTPUT / "one_factor_active_summary.csv", index=False)
    comparison = summary.merge(
        one_factor_summary, on="strategy", suffixes=("_four_factor", "_one_factor")
    )
    comparison.to_csv(OUTPUT / "one_vs_four_factor.csv", index=False)

    share_columns = [f"mean_risk_share_{factor}" for factor in FACTORS] + ["mean_specific_risk_share"]
    chart = summary.set_index("strategy")[share_columns].copy()
    chart.columns = FACTORS + ["specific"]
    axis = chart.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20c")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean active-risk Euler contribution share")
    axis.set_title("Four-factor active risk attribution versus CSI500")
    axis.legend(ncol=5, loc="upper center")
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT / "factor_risk_shares.png", dpi=170)
    plt.close(axis.figure)

    report_lines = [
        "# CSI500 四因子主动风险归因实验",
        "",
        "风险目标为组合毛收益减 CSI500 收益；交易成本不进入风险模型。",
        "因子包括 CSI500 市场收益、流通市值规模、分钟实现波动率和 Amihud 反向流动性。",
        "后三个风格因子每日按 1%/99% 去极值并标准化，用流通市值平方根加权的截面回归估计因子收益。",
        "组合风险模型每天仅使用此前最多 252 日因子收益，EWMA 半衰期为 60 日。",
        "",
        summary[["strategy", "mean_beta_market", "mean_r_squared", "volatility_calibration_ratio", "coverage_95", *share_columns]].to_markdown(index=False),
        "",
        "## 相对流通市值加权 CSI500 代理的平均主动暴露",
        "",
        summary[["strategy", *[f"mean_active_exposure_{name}" for name in EXPOSURES]]].to_markdown(index=False),
        "",
        "市场 beta 是主动收益对市场的暴露，因此约等于组合总 beta 减 1。",
        "Euler 因子贡献允许为负，表示该因子与其他风险暴露形成对冲；所有因子贡献加特异风险严格等于预测主动方差。",
        "该版本仍未使用行业因子，因为当前行业字段没有完成历史时点审计。",
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "model": "csi500_active_market_plus_three_style_factors",
        "risk_target": "gross_return_minus_csi500_return",
        "style_exposures": {
            "size": "log(point_in_time_float_market_cap)",
            "volatility": "log(mf_realized_volatility)",
            "liquidity": "-log(mf_amihud_intraday_w20)",
        },
        "cross_sectional_target": "raw_h1_excess_return",
        "config": config.__dict__,
        "factor_return_days": int(len(factor_returns)),
        "panel_rows": int(len(panel)),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
