#!/usr/bin/env python3
"""Build the first stock-level CSI500 factor-risk snapshot."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-alpha-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from a_share_data.factor_risk_model import (
    FactorRiskConfig,
    build_factor_risk_snapshot,
    estimate_cross_sectional_factor_model,
    estimate_factor_covariance,
    write_factor_risk_snapshot,
)
from experiment_expanded_active_risk import (
    EXPOSURES,
    FEATURES,
    INDEX_DAILY,
    MARKET_CAP,
    QFQ_DAILY,
    ROOT,
    _standardize_expression,
    build_historical_exposures,
)


PRODUCTION_PREDICTIONS = (
    ROOT
    / "results/predict/research-oos-raw-daily60-minute45-dos20-csi300-csi500-h1-h5-h10-v1"
    / "predictions.parquet"
)


def load_full_exposure_panel() -> pd.DataFrame:
    """Load point-in-time exposures without restricting dates to one model run."""

    features = pl.scan_parquet(str(FEATURES / "year=*/features.parquet")).select(
        pl.col("trade_date").alias("date"),
        "ts_code",
        "mf_amihud_intraday_w20",
    )
    cap = pl.scan_parquet(MARKET_CAP).select(
        pl.col("trade_date").cast(pl.Date).alias("date"), "ts_code", "float_mv"
    )
    panel = (
        build_historical_exposures()
        .join(features, on=["date", "ts_code"], how="inner")
        .join(cap, on=["date", "ts_code"], how="inner")
        .with_columns(
            pl.col("float_mv").log().alias("size"),
            pl.col("residual_volatility").clip(1e-12, None).log().alias("residual_volatility"),
            (-pl.col("mf_amihud_intraday_w20").clip(1e-18, None).log()).alias("liquidity"),
            pl.col("float_mv").sqrt().alias("regression_weight"),
        )
        .with_columns(*[_standardize_expression(column) for column in EXPOSURES])
        .with_columns(
            (pl.col("float_mv") / pl.col("float_mv").sum().over("date")).alias(
                "benchmark_weight"
            )
        )
        .select(
            "date",
            "ts_code",
            "regression_weight",
            "benchmark_weight",
            *EXPOSURES,
        )
        .collect()
    )
    return panel.to_pandas()


def build_daily_optimizer_inputs(
    output: Path,
    exposure_panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    specific_returns: pd.DataFrame,
    prediction_path: Path,
    config: FactorRiskConfig,
) -> dict[str, int]:
    """Build point-in-time daily risk inputs consumed by the sparse QP."""

    risk_dates = set(
        pl.read_parquet(prediction_path, columns=["trade_date"])
        .get_column("trade_date")
        .unique()
        .to_list()
    )
    exposures = exposure_panel.loc[
        pd.to_datetime(exposure_panel["date"]).dt.date.isin(risk_dates),
        ["date", "ts_code", "benchmark_weight", *EXPOSURES],
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()
    exposures.insert(2, "market", 1.0)
    exposures.to_parquet(output / "optimizer_exposures.parquet", index=False)

    factor_names = ["market", *EXPOSURES]
    factor_columns = ["market_factor_return", *[f"{name}_factor_return" for name in EXPOSURES]]
    covariance_rows: list[dict[str, object]] = []
    usable_dates: set[object] = set()
    for date in sorted(pd.to_datetime(list(risk_dates))):
        try:
            covariance, _ = estimate_factor_covariance(
                factor_returns, factor_columns, date, config
            )
        except ValueError:
            continue
        usable_dates.add(date.date())
        for left, left_name in enumerate(factor_names):
            for right, right_name in enumerate(factor_names):
                covariance_rows.append(
                    {
                        "date": date,
                        "factor_left": left_name,
                        "factor_right": right_name,
                        "covariance": float(covariance[left, right]),
                    }
                )
    pd.DataFrame(covariance_rows).to_parquet(
        output / "factor_covariance_daily.parquet", index=False
    )

    residual = specific_returns.copy()
    residual["date"] = pd.to_datetime(residual["date"])
    pivot = residual.pivot(index="date", columns="ts_code", values="specific_return").sort_index()
    values = pivot.to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    squared = np.where(valid, values * values, 0.0)
    code_index = {str(code): index for index, code in enumerate(pivot.columns)}
    exposure_by_date = {
        pd.Timestamp(date).date(): group
        for date, group in exposures.groupby("date", sort=False)
    }
    decay = math.exp(-math.log(2.0) / config.specific_half_life)
    expired_scale = decay**config.lookback
    numerator = np.zeros(values.shape[1], dtype=np.float64)
    denominator = np.zeros(values.shape[1], dtype=np.float64)
    cumulative_counts = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.int32), np.cumsum(valid, axis=0, dtype=np.int32)]
    )
    specific_rows: list[dict[str, object]] = []
    for row_index, date in enumerate(pivot.index):
        date_key = date.date()
        if date_key in usable_dates and date_key in exposure_by_date:
            counts = cumulative_counts[row_index] - cumulative_counts[max(0, row_index - config.lookback)]
            raw = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0.0,
            )
            prior_pool = raw[(counts >= config.minimum_specific_observations) & np.isfinite(raw)]
            if len(prior_pool):
                prior = float(np.median(prior_pool))
                lower = max(
                    float(np.quantile(prior_pool, 0.10)) * config.specific_floor_multiplier,
                    1e-12,
                )
                upper = max(
                    float(np.quantile(prior_pool, 0.90)) * config.specific_cap_multiplier,
                    lower,
                )
                for item in exposure_by_date[date_key].itertuples(index=False):
                    column = code_index.get(str(item.ts_code))
                    observations = int(counts[column]) if column is not None else 0
                    raw_variance = raw[column] if column is not None else math.nan
                    if not np.isfinite(raw_variance):
                        raw_variance = prior
                    confidence = observations / (
                        observations + config.specific_prior_observations
                    )
                    variance = float(
                        np.clip(
                            confidence * raw_variance + (1.0 - confidence) * prior,
                            lower,
                            upper,
                        )
                    )
                    specific_rows.append(
                        {"date": date, "ts_code": str(item.ts_code), "specific_variance": variance}
                    )
        numerator *= decay
        denominator *= decay
        numerator += squared[row_index]
        denominator += valid[row_index]
        expired = row_index - config.lookback
        if expired >= 0:
            numerator -= expired_scale * squared[expired]
            denominator -= expired_scale * valid[expired]
    pivot_dates = {date.date() for date in pivot.index}
    trailing_dates = sorted(
        (usable_dates & exposure_by_date.keys()).difference(pivot_dates)
    )
    if trailing_dates:
        last_return_date = pivot.index.max().date()
        if any(date <= last_return_date for date in trailing_dates):
            raise ValueError(
                "specific-return history has internal gaps on requested risk dates"
            )
        row_count = len(pivot.index)
        counts = cumulative_counts[row_count] - cumulative_counts[
            max(0, row_count - config.lookback)
        ]
        raw = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0.0,
        )
        prior_pool = raw[
            (counts >= config.minimum_specific_observations) & np.isfinite(raw)
        ]
        if not len(prior_pool):
            raise ValueError("no trailing-date specific-variance prior is available")
        prior = float(np.median(prior_pool))
        lower = max(
            float(np.quantile(prior_pool, 0.10)) * config.specific_floor_multiplier,
            1e-12,
        )
        upper = max(
            float(np.quantile(prior_pool, 0.90)) * config.specific_cap_multiplier,
            lower,
        )
        for date_key in trailing_dates:
            for item in exposure_by_date[date_key].itertuples(index=False):
                column = code_index.get(str(item.ts_code))
                observations = int(counts[column]) if column is not None else 0
                raw_variance = raw[column] if column is not None else math.nan
                if not np.isfinite(raw_variance):
                    raw_variance = prior
                confidence = observations / (
                    observations + config.specific_prior_observations
                )
                variance = float(
                    np.clip(
                        confidence * raw_variance + (1.0 - confidence) * prior,
                        lower,
                        upper,
                    )
                )
                specific_rows.append(
                    {
                        "date": pd.Timestamp(date_key),
                        "ts_code": str(item.ts_code),
                        "specific_variance": variance,
                    }
                )
    pd.DataFrame(specific_rows).to_parquet(
        output / "specific_variance_daily.parquet", index=False
    )
    return {
        "risk_dates_requested": len(risk_dates),
        "risk_dates_built": len(usable_dates),
        "optimizer_exposure_rows": len(exposures),
        "daily_specific_rows": len(specific_rows),
    }


def load_open_to_open_returns() -> pd.DataFrame:
    """Build T-signal labels from T+1 open to T+2 open without lookahead joins."""

    calendar = (
        pl.read_parquet(INDEX_DAILY)
        .filter(pl.col("index_code") == "000905.SH")
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .with_columns(
            pl.col("trade_date").shift(1).alias("signal_date"),
            pl.col("trade_date").shift(-1).alias("exit_date"),
        )
        .drop_nulls()
        .rename({"trade_date": "entry_date"})
    )
    opens = (
        pl.scan_parquet(str(QFQ_DAILY / "ts_code=*/daily.parquet"), hive_partitioning=False)
        .select("trade_date", "ts_code", pl.col("open").cast(pl.Float64).alias("qfq_open"))
        .filter(pl.col("qfq_open") > 0.0)
        .collect()
    )
    entry = calendar.join(
        opens.rename({"trade_date": "entry_date", "qfq_open": "entry_open"}),
        on="entry_date",
        how="inner",
    )
    realised = (
        entry.join(
            opens.rename({"trade_date": "exit_date", "qfq_open": "exit_open"}),
            on=["exit_date", "ts_code"],
            how="inner",
        )
        .with_columns(
            (pl.col("exit_open") / pl.col("entry_open") - 1.0).alias("future_stock_return")
        )
        .select(pl.col("signal_date").alias("date"), "ts_code", "future_stock_return")
    )
    return realised.to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/risk/stock_factor_risk_v1",
    )
    parser.add_argument("--as-of", type=pd.Timestamp)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--half-life", type=float, default=63.0)
    parser.add_argument("--shrinkage", type=float, default=0.20)
    parser.add_argument("--daily-optimizer-inputs", action="store_true")
    parser.add_argument("--prediction-path", type=Path, default=PRODUCTION_PREDICTIONS)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    exposure_panel = load_full_exposure_panel()
    panel = exposure_panel.merge(
        load_open_to_open_returns(),
        on=["date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    factor_returns, specific_returns = estimate_cross_sectional_factor_model(
        panel,
        EXPOSURES,
        return_column="future_stock_return",
        minimum_observations=100,
    )
    factor_returns.to_parquet(args.output / "factor_returns.parquet", index=False)
    specific_returns.to_parquet(args.output / "specific_returns.parquet", index=False)

    available_dates = pd.to_datetime(exposure_panel["date"]).drop_duplicates().sort_values()
    as_of = args.as_of if args.as_of is not None else available_dates.iloc[-1]
    current = exposure_panel.loc[
        pd.to_datetime(exposure_panel["date"]) == as_of, ["ts_code", *EXPOSURES]
    ]
    if current.empty:
        raise ValueError(f"no exposures for requested as-of date {as_of.date()}")
    config = FactorRiskConfig(
        lookback=args.lookback,
        half_life=args.half_life,
        specific_half_life=args.half_life,
        covariance_shrinkage=args.shrinkage,
    )
    daily_diagnostics = (
        build_daily_optimizer_inputs(
            args.output,
            exposure_panel,
            factor_returns,
            specific_returns,
            args.prediction_path,
            config,
        )
        if args.daily_optimizer_inputs
        else {}
    )
    snapshot, specific_frame = build_factor_risk_snapshot(
        as_of,
        current,
        EXPOSURES,
        factor_returns,
        specific_returns,
        config,
    )
    snapshot_dir = args.output / "snapshots" / f"date={as_of.date()}"
    write_factor_risk_snapshot(snapshot, specific_frame, snapshot_dir, config)
    standard_deviation = np.sqrt(np.diag(snapshot.factor_covariance))
    factor_correlation = snapshot.factor_covariance / np.outer(
        standard_deviation, standard_deviation
    )
    specific_volatility = np.sqrt(specific_frame["specific_variance"] * 252.0)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    image = axes[0].imshow(factor_correlation, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    axes[0].set_xticks(range(len(snapshot.factor_names)), snapshot.factor_names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(snapshot.factor_names)), snapshot.factor_names)
    axes[0].set_title("Factor-return correlation")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].hist(specific_volatility, bins=35, color="#3769a5", alpha=0.85)
    axes[1].axvline(float(specific_volatility.median()), color="#b6423c", linestyle="--")
    axes[1].set_title("Annualized stock-specific volatility")
    axes[1].set_xlabel("Volatility")
    axes[1].set_ylabel("Stocks")
    figure.tight_layout()
    figure.savefig(args.output / "risk_snapshot_diagnostics.png", dpi=170)
    plt.close(figure)

    equal_weight = np.full(len(snapshot.securities), 1.0 / len(snapshot.securities))
    factor_exposure = snapshot.exposures.T @ equal_weight
    factor_variance = float(
        factor_exposure @ snapshot.factor_covariance @ factor_exposure
    )
    specific_variance = float(
        np.dot(equal_weight * equal_weight, snapshot.specific_variance)
    )
    equal_weight_volatility = math.sqrt(
        (factor_variance + specific_variance) * 252.0
    )
    manifest = {
        "model": "stock_factor_risk_v1",
        "as_of": str(as_of.date()),
        "factors": list(snapshot.factor_names),
        "point_in_time_rule": "risk histories use dates strictly before as_of",
        "return_target": "realized qfq open-to-open from T+1 open to T+2 open",
        "industry": "excluded: current industry source has no audited effective dates",
        "snapshot": str(snapshot_dir),
        **daily_diagnostics,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "report.md").write_text(
        f"""# CSI500 股票因子风险模型 V1

快照日：{as_of.date()}。收益口径为信号日 T 对应的 T+1 开盘至 T+2 开盘复权收益；风险历史严格早于快照日。

| 指标 | 数值 |
| --- | ---: |
| 股票数 | {len(snapshot.securities)} |
| 因子数 | {len(snapshot.factor_names)} |
| 因子历史观测 | {snapshot.diagnostics['factor_observations']} |
| 因子协方差最小特征值 | {snapshot.diagnostics['factor_covariance_min_eigenvalue']:.8g} |
| 因子协方差条件数 | {snapshot.diagnostics['factor_covariance_condition_number']:.3f} |
| 缺失特异风险历史股票 | {snapshot.diagnostics['specific_missing_histories']} |
| 等权组合预测年化波动率 | {equal_weight_volatility:.2%} |
| 等权组合因子风险占比 | {factor_variance / (factor_variance + specific_variance):.2%} |
| 个股特异年化波动率中位数 | {specific_volatility.median():.2%} |

![风险快照诊断](risk_snapshot_diagnostics.png)

## 当前边界

- 因子为市场截距、规模、Beta、残差波动率、12-1 月动量和流动性。
- 行业因子尚未加入，因为当前行业字段没有完成历史生效日期审计。
- 当前结果是最新时点风险快照，不代表已经完成逐日滚动的风险预测校准。
- 接入 optimizer 前仍需用历史持仓验证预测 tracking error 与未来实现 tracking error 的校准关系。
""",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "diagnostics": snapshot.diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
