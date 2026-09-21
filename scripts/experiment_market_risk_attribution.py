#!/usr/bin/env python3
"""Run a minimal CSI500 market-factor risk attribution experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-alpha-matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from a_share_data.risk_attribution import (
    MarketRiskConfig,
    rolling_market_risk_attribution,
    summarise_attribution,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/predict/sequence-lstm-residual-daily60-minute45-v2/full"
DEFAULT_INPUTS = {
    "lgbm": RUN_ROOT / "weekly_comparison/backtests/daily_swap3/lgbm/portfolio_daily.parquet",
    "lstm": RUN_ROOT / "weekly_comparison/backtests/daily_swap3/lstm/portfolio_daily.parquet",
    "blend_50_50": RUN_ROOT
    / "weekly_comparison/backtests/daily_swap3/blend_50_50/portfolio_daily.parquet",
    "rank_tilt_10": RUN_ROOT
    / "rank_tilt_experiment/daily_swap3/rank_tilt_10/portfolio_daily.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/risk/simple_market_factor_v1",
    )
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--half-life", type=float, default=60.0)
    parser.add_argument("--forward-window", type=int, default=20)
    parser.add_argument("--minimum-observations", type=int, default=126)
    return parser.parse_args()


def load_strategy(path: Path) -> pd.DataFrame:
    frame = pl.read_parquet(path).select(
        pl.col("execution_date").alias("date"),
        pl.col("gross_return").alias("portfolio_return"),
        pl.col("csi500_return").alias("market_return"),
    )
    return frame.to_pandas()


def yearly_summary(strategy: str, daily: pd.DataFrame) -> list[dict[str, object]]:
    work = daily.copy()
    work["year"] = pd.to_datetime(work["date"]).dt.year
    output = []
    for year, group in work.groupby("year", sort=True):
        row: dict[str, object] = {"strategy": strategy, "year": int(year)}
        row.update(summarise_attribution(group))
        output.append(row)
    return output


def make_chart(all_daily: pd.DataFrame, output_path: Path) -> None:
    strategies = list(all_daily["strategy"].drop_duplicates())
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for strategy in strategies:
        frame = all_daily[all_daily["strategy"] == strategy]
        axes[0].plot(
            frame["date"],
            frame["predicted_volatility_ann"].rolling(20).mean(),
            linewidth=1.2,
            label=strategy,
        )
        axes[1].plot(
            frame["date"],
            frame["factor_risk_share"].rolling(20).mean(),
            linewidth=1.2,
            label=strategy,
        )
    axes[0].set_ylabel("Annualized predicted volatility")
    axes[0].set_title("Lagged CSI500 one-factor risk model")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2)
    axes[1].set_ylabel("Market-factor risk share")
    axes[1].set_xlabel("Execution date")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_path: Path, config: MarketRiskConfig) -> None:
    columns = [
        "strategy",
        "mean_beta",
        "mean_r_squared",
        "mean_predicted_volatility_ann",
        "volatility_calibration_ratio",
        "predicted_vs_forward_volatility_corr",
        "coverage_95",
        "mean_factor_risk_share",
    ]
    display = summary[columns].copy()
    for column in columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    lines = [
        "# CSI500 单因子风险归因实验",
        "",
        f"每个交易日只使用此前最多 {config.lookback} 日数据，EWMA 半衰期 "
        f"{config.half_life:g} 日；最少 {config.minimum_observations} 个观测。",
        "收益使用组合毛收益，以 CSI500 同期收益为市场因子。",
        "",
        display.to_markdown(index=False),
        "",
        "## 解释",
        "",
        "- `volatility_calibration_ratio` 为实际收益冲击方差与预测方差之比的平方根；接近 1 较理想。",
        "- `coverage_95` 是实际组合收益落在预测均值加减 1.96 倍预测波动内的比例。",
        "- `mean_factor_risk_share` 是 CSI500 市场因子对总预测方差的平均贡献，其余为特异风险。",
        "- 该模型适合先验证组合层风险校准与归因。它没有行业、规模等股票截面暴露，不能直接充当完整优化器协方差模型。",
        "",
        "## 当前可用的扩展原料",
        "",
        "- 历史 CSI500 收益：已用于本实验。",
        "- 公告日可用的流通市值缓存：可构造规模因子和截面回归权重。",
        "- 105 因子中的波动率、流动性和市场残差类特征：可整理为干净的风险暴露，但不应直接把全部 Alpha 当风险因子。",
        "- `instruments.industry` 缺少历史生效日期，本实验未将其当作历史时点行业。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = MarketRiskConfig(
        lookback=args.lookback,
        half_life=args.half_life,
        forward_window=args.forward_window,
        minimum_observations=args.minimum_observations,
    )

    daily_frames = []
    summaries = []
    annual = []
    for strategy, path in DEFAULT_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        result = rolling_market_risk_attribution(load_strategy(path), config)
        result.insert(0, "strategy", strategy)
        daily_frames.append(result)
        summary = {"strategy": strategy, **summarise_attribution(result)}
        summaries.append(summary)
        annual.extend(yearly_summary(strategy, result))

    all_daily = pd.concat(daily_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    annual_frame = pd.DataFrame(annual)
    pl.from_pandas(all_daily).write_parquet(output_dir / "daily_attribution.parquet")
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    annual_frame.to_csv(output_dir / "annual_summary.csv", index=False)
    make_chart(all_daily, output_dir / "risk_attribution.png")
    write_report(summary_frame, output_dir / "report.md", config)
    manifest = {
        "model": "lagged_csi500_one_factor",
        "return": "gross_return",
        "config": config.__dict__,
        "inputs": {key: str(value.resolve()) for key, value in DEFAULT_INPUTS.items()},
        "outputs": [
            "daily_attribution.parquet",
            "summary.csv",
            "annual_summary.csv",
            "risk_attribution.png",
            "report.md",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
