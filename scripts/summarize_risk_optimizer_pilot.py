#!/usr/bin/env python3
"""Summarize the aligned no-industry risk-aware optimizer pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-alpha-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/predict/risk-multiperiod-optimizer-pilot"
BASELINE = (
    ROOT
    / "results/predict/new125-multiperiod-from-2025-2bps-v1/h1h5_5period/backtest"
    / "portfolio_daily.parquet"
)
RISK_RUN = OUTPUT / "aligned_100/lambda_0_01"


def metrics(name: str, frame: pl.DataFrame, dates: pl.DataFrame) -> dict[str, object]:
    frame = frame.join(dates, on="execution_date", how="inner").sort("execution_date")
    benchmark_column = (
        "csi500_return" if "csi500_return" in frame.columns else "benchmark_return"
    )
    net = frame["net_return"].to_numpy()
    benchmark = frame[benchmark_column].to_numpy()
    active = net - benchmark
    nav = np.cumprod(1.0 + net)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    return {
        "name": name,
        "days": len(frame),
        "start": str(frame["execution_date"][0]),
        "end": str(frame["execution_date"][-1]),
        "net_return": float(nav[-1] - 1.0),
        "relative_return": float(nav[-1] / benchmark_nav[-1] - 1.0),
        "information_ratio": float(
            active.mean() / active.std(ddof=1) * np.sqrt(252.0)
        ),
        "tracking_error": float(active.std(ddof=1) * np.sqrt(252.0)),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "average_buy_turnover": float(frame["buy_turnover"].mean()),
        "average_holding_count": float(frame["holding_count"].mean()),
    }


def main() -> None:
    risk_daily = pl.read_parquet(RISK_RUN / "backtest/portfolio_daily.parquet")
    dates = risk_daily.select("execution_date")
    baseline_daily = pl.read_parquet(BASELINE)
    rows = [
        metrics("no_risk_baseline", baseline_daily, dates),
        metrics("risk_lambda_0_01", risk_daily, dates),
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "aligned_comparison.csv", index=False)

    baseline_aligned = baseline_daily.join(dates, on="execution_date", how="inner").sort(
        "execution_date"
    )
    risk_aligned = risk_daily.sort("execution_date")
    optimizer_daily = pl.read_parquet(RISK_RUN / "optimizer_daily.parquet").sort(
        "execution_date"
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(
        baseline_aligned["execution_date"],
        np.cumprod(1.0 + baseline_aligned["net_return"].to_numpy()),
        label="No-risk baseline",
    )
    axes[0].plot(
        risk_aligned["execution_date"],
        np.cumprod(1.0 + risk_aligned["net_return"].to_numpy()),
        label="Risk-aware lambda=0.01",
    )
    axes[0].set_ylabel("Net NAV")
    axes[0].legend()
    axes[0].set_title("Aligned 100-day risk-aware optimizer pilot")
    axes[1].plot(
        optimizer_daily["execution_date"],
        optimizer_daily["predicted_active_volatility_ann"],
        color="#b6423c",
    )
    axes[1].set_ylabel("Predicted active vol")
    axes[1].set_xlabel("Execution date")
    figure.tight_layout()
    figure.savefig(OUTPUT / "aligned_pilot.png", dpi=170)
    plt.close(figure)

    baseline, risk = rows
    solver_status = optimizer_daily.group_by("solver_status").len().sort("solver_status")
    report = f"""# 无行业风险模型接入五期 Optimizer：对齐 Pilot

区间：{risk['start']} 至 {risk['end']}，共 {risk['days']} 个完全一致的执行日。两组均使用原 125 因子 LGBM H1/H5 预测、五期滚动规划、买卖各 2bp、98% 投资和单票 1% 上限。

| 指标 | 无风险基线 | 风险版 λ=0.01 | 变化 |
| --- | ---: | ---: | ---: |
| 净收益 | {baseline['net_return']:.2%} | {risk['net_return']:.2%} | {risk['net_return'] - baseline['net_return']:+.2%} |
| 相对收益 | {baseline['relative_return']:.2%} | {risk['relative_return']:.2%} | {risk['relative_return'] - baseline['relative_return']:+.2%} |
| 信息比率 | {baseline['information_ratio']:.3f} | {risk['information_ratio']:.3f} | {risk['information_ratio'] - baseline['information_ratio']:+.3f} |
| 实现跟踪误差 | {baseline['tracking_error']:.2%} | {risk['tracking_error']:.2%} | {risk['tracking_error'] - baseline['tracking_error']:+.2%} |
| 最大回撤 | {baseline['max_drawdown']:.2%} | {risk['max_drawdown']:.2%} | {risk['max_drawdown'] - baseline['max_drawdown']:+.2%} |
| 平均买入换手 | {baseline['average_buy_turnover']:.2%} | {risk['average_buy_turnover']:.2%} | {risk['average_buy_turnover'] - baseline['average_buy_turnover']:+.2%} |
| 平均持股数 | {baseline['average_holding_count']:.2f} | {risk['average_holding_count']:.2f} | {risk['average_holding_count'] - baseline['average_holding_count']:+.2f} |

![对齐 Pilot](aligned_pilot.png)

## 结论

风险项按预期轻微降低了实现跟踪误差，但下降幅度只有 {baseline['tracking_error'] - risk['tracking_error']:.2%}，同时净收益下降 {baseline['net_return'] - risk['net_return']:.2%}、信息比率下降 {baseline['information_ratio'] - risk['information_ratio']:.3f}。该版本不应替代生产无风险 optimizer，也不值得直接扩大到全历史参数搜索。

当前风险基准使用当日流通市值权重代理 CSI500 官方权重，且没有行业因子。后续若继续，应优先校准 alpha 与风险项的单位、研究风险预算/软约束，而不是继续增大 λ。

## 求解审计

- 最大原始约束残差：{optimizer_daily['primal_residual'].max():.3g}
- 最大缩放后对偶残差：{optimizer_daily['dual_residual_scaled'].max():.3g}
- Solver 状态计数：`{json.dumps(dict(zip(solver_status['solver_status'], solver_status['len'])), ensure_ascii=False)}`
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
