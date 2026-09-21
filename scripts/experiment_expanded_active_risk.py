#!/usr/bin/env python3
"""Run an expanded, point-in-time CSI500 active-risk attribution experiment."""

from __future__ import annotations

import json
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
QFQ_DAILY = ROOT / "A_stock_database/lake/canonical/baostock_qfq_csi300_csi500_v1"
INDEX_DAILY = ROOT / "A_stock_database/lake/canonical/index_daily/index_daily.parquet"
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
OUTPUT = ROOT / "results/risk/active_expanded_factor_v2"
EXPOSURES = ["size", "beta", "residual_volatility", "momentum", "liquidity"]
FACTORS = ["market", *EXPOSURES]


def _standardize_expression(column: str) -> pl.Expr:
    raw = pl.col(column)
    lower = raw.quantile(0.01).over("date")
    upper = raw.quantile(0.99).over("date")
    clipped = raw.clip(lower, upper)
    return ((clipped - clipped.mean().over("date")) / clipped.std().over("date")).alias(
        column
    )


def build_historical_exposures() -> pl.LazyFrame:
    market = (
        pl.scan_parquet(INDEX_DAILY)
        .filter(pl.col("index_code") == "000905.SH")
        .select(
            pl.col("trade_date").alias("date"),
            pl.col("close_return").alias("market_return"),
        )
    )
    stock = (
        pl.scan_parquet(str(QFQ_DAILY / "ts_code=*/daily.parquet"), hive_partitioning=False)
        .select(
            pl.col("trade_date").alias("date"),
            "ts_code",
            (pl.col("pct_chg") / 100.0).alias("stock_return"),
        )
        .join(market, on="date", how="left")
        .sort(["ts_code", "date"])
        .with_columns(
            (pl.col("stock_return") - pl.col("market_return")).alias(
                "market_adjusted_return"
            ),
            pl.col("stock_return").clip(-0.999999, None).log1p().alias("log_return"),
        )
    )
    return stock.with_columns(
        (
            pl.rolling_cov(
                "stock_return", "market_return", window_size=120, min_samples=60
            ).over("ts_code")
            / pl.col("market_return")
            .rolling_var(window_size=120, min_samples=60)
            .over("ts_code")
        ).alias("beta"),
        pl.col("market_adjusted_return")
        .rolling_std(window_size=60, min_samples=40)
        .over("ts_code")
        .alias("residual_volatility"),
        pl.col("log_return")
        .shift(21)
        .rolling_sum(window_size=231, min_samples=126)
        .over("ts_code")
        .alias("momentum"),
    ).select("date", "ts_code", "beta", "residual_volatility", "momentum")


def load_style_panel() -> pd.DataFrame:
    prediction = pl.scan_parquet(PREDICTIONS).select(
        pl.col("trade_date").alias("date"),
        "ts_code",
        pl.col("raw_h1").alias("future_excess_return"),
    )
    features = pl.scan_parquet(str(FEATURES / "year=*/features.parquet")).select(
        pl.col("trade_date").alias("date"),
        "ts_code",
        "mf_amihud_intraday_w20",
    )
    cap = pl.scan_parquet(MARKET_CAP).select(
        pl.col("trade_date").cast(pl.Date).alias("date"), "ts_code", "float_mv"
    )
    panel = (
        prediction.join(build_historical_exposures(), on=["date", "ts_code"], how="left")
        .join(features, on=["date", "ts_code"], how="left")
        .join(cap, on=["date", "ts_code"], how="left")
        .with_columns(
            pl.col("float_mv").median().over("date").alias("median_float_mv")
        )
        .with_columns(pl.col("float_mv").fill_null(pl.col("median_float_mv")))
        .with_columns(
            pl.col("float_mv").log().alias("size"),
            pl.col("residual_volatility")
            .clip(1e-12, None)
            .log()
            .alias("residual_volatility"),
            (-pl.col("mf_amihud_intraday_w20").clip(1e-18, None).log()).alias(
                "liquidity"
            ),
            pl.col("float_mv").sqrt().alias("regression_weight"),
        )
        .with_columns(*[_standardize_expression(column) for column in EXPOSURES])
        .select(
            "date",
            "ts_code",
            "future_excess_return",
            "regression_weight",
            *EXPOSURES,
        )
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
    rename = {"market_factor_return": "market"}
    rename.update({f"{name}_factor_return": name for name in EXPOSURES})
    return result.rename(columns=rename)


def calculate_direct_exposures(
    strategy: str, holdings_path: Path, panel: pd.DataFrame
) -> pd.DataFrame:
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
        .select(pl.col("signal_date").alias("date"), "ts_code", "weight")
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
        weight_sum = float(valid["weight"].sum())
        if weight_sum <= 0.0:
            continue
        weights = valid["weight"].to_numpy(dtype=np.float64) / weight_sum
        row: dict[str, object] = {"strategy": strategy, "date": date}
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
    coverage = panel[["future_excess_return", "regression_weight", *EXPOSURES]].notna().mean()
    factor_returns = estimate_cross_sectional_factor_returns(panel, EXPOSURES)
    factor_returns.to_csv(OUTPUT / "factor_returns.csv", index=False)

    daily_frames = []
    direct_frames = []
    summaries = []
    for strategy, path in DEFAULT_INPUTS.items():
        daily = rolling_factor_risk_attribution(load_portfolio(path, factor_returns), FACTORS, config)
        daily.insert(0, "strategy", strategy)
        daily_frames.append(daily)
        direct = calculate_direct_exposures(strategy, path.parent / "holdings.parquet", panel)
        direct_frames.append(direct)
        row = {"strategy": strategy, **summarise_attribution(daily)}
        for factor in FACTORS:
            row[f"mean_beta_{factor}"] = float(daily[f"beta_{factor}"].mean())
            row[f"mean_risk_share_{factor}"] = float(daily[f"risk_share_{factor}"].mean())
        for exposure in EXPOSURES:
            row[f"mean_active_exposure_{exposure}"] = float(
                direct[f"active_{exposure}"].mean()
            )
        summaries.append(row)

    all_daily = pd.concat(daily_frames, ignore_index=True)
    direct_exposures = pd.concat(direct_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    pl.from_pandas(all_daily).write_parquet(OUTPUT / "daily_attribution.parquet")
    direct_exposures.to_csv(OUTPUT / "direct_exposures.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    v1 = pd.read_csv(ROOT / "results/risk/active_style_factor_v1/summary.csv")
    comparison = summary.merge(v1, on="strategy", suffixes=("_v2", "_v1"))
    comparison["r_squared_increment"] = (
        comparison["mean_r_squared_v2"] - comparison["mean_r_squared_v1"]
    )
    comparison.to_csv(OUTPUT / "v1_vs_v2.csv", index=False)

    share_columns = [f"mean_risk_share_{factor}" for factor in FACTORS] + [
        "mean_specific_risk_share"
    ]
    chart = summary.set_index("strategy")[share_columns]
    chart.columns = FACTORS + ["specific"]
    axis = chart.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    axis.set_title("Expanded active-risk attribution versus CSI500")
    axis.set_ylabel("Mean Euler variance contribution share")
    axis.legend(ncol=4, loc="upper center")
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT / "expanded_active_risk.png", dpi=170)
    plt.close(axis.figure)

    report_columns = [
        "strategy",
        "mean_predicted_volatility_ann",
        "mean_r_squared",
        "volatility_calibration_ratio",
        "predicted_vs_forward_volatility_corr",
        *share_columns,
    ]
    direct_columns = ["strategy", *[f"mean_active_exposure_{x}" for x in EXPOSURES]]
    report = [
        "# 扩展主动风险模型 V2",
        "",
        "目标为组合毛收益减 CSI500 收益。因子包括市场、规模、个股滚动 beta、市场调整残差波动率、12-1 月动量和流动性。",
        "所有股票暴露只使用信号日收盘时已经可见的数据；行业因子因缺少历史时点分类仍未加入。",
        "",
        summary[report_columns].to_markdown(index=False),
        "",
        "## 平均主动暴露",
        "",
        summary[direct_columns].to_markdown(index=False),
        "",
        "## 相对 V1 的解释度增量",
        "",
        comparison[["strategy", "mean_r_squared_v1", "mean_r_squared_v2", "r_squared_increment"]].to_markdown(index=False),
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "model": "expanded_active_factor_v2",
        "risk_target": "gross_return_minus_csi500_return",
        "exposures": EXPOSURES,
        "config": config.__dict__,
        "panel_rows": int(len(panel)),
        "factor_return_days": int(len(factor_returns)),
        "input_coverage": {name: float(value) for name, value in coverage.items()},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparison[["strategy", "mean_r_squared_v1", "mean_r_squared_v2", "r_squared_increment"]].to_string(index=False))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
