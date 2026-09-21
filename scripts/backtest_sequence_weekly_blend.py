#!/usr/bin/env python3
"""Compare daily and weekly CSI500 Top100 policies for LGBM/LSTM/blend."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-alpha-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/quant-alpha-cache")

import numpy as np
import polars as pl

from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.research_oos import daily_normalize
from a_share_data.sequence_oos import _csi500


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _blend(lstm: pl.DataFrame, lgbm: pl.DataFrame) -> pl.DataFrame:
    keys = ["trade_date", "ts_code"]
    left = daily_normalize(lstm, "raw_h1", "raw_h5")
    right = daily_normalize(lgbm, "raw_h1", "raw_h5").select(
        *keys, "raw_h1", "raw_h5", "pred_h1", "pred_h5"
    ).rename({
        "raw_h1": "lgbm_raw_h1", "raw_h5": "lgbm_raw_h5",
        "pred_h1": "lgbm_pred_h1", "pred_h5": "lgbm_pred_h5",
    })
    return left.join(right, on=keys, how="inner").with_columns(
        ((pl.col("raw_h1") + pl.col("lgbm_raw_h1")) / 2).alias("raw_h1"),
        ((pl.col("raw_h5") + pl.col("lgbm_raw_h5")) / 2).alias("raw_h5"),
        ((pl.col("pred_h1") + pl.col("lgbm_pred_h1")) / 2).alias("pred_h1"),
        ((pl.col("pred_h5") + pl.col("lgbm_pred_h5")) / 2).alias("pred_h5"),
    ).select("trade_date", "ts_code", "execution_date", "raw_h1", "raw_h5", "pred_h1", "pred_h5")


def _horizon_hybrid(lstm: pl.DataFrame, lgbm: pl.DataFrame) -> pl.DataFrame:
    """Use LGBM for H1 and LSTM for H5 after same-day normalization."""
    keys = ["trade_date", "ts_code"]
    lstm_z = daily_normalize(lstm, "raw_h1", "raw_h5").select(
        *keys, pl.col("raw_h5"), pl.col("pred_h5"), pl.col("execution_date"))
    lgbm_z = daily_normalize(lgbm, "raw_h1", "raw_h5").select(
        *keys, pl.col("raw_h1"), pl.col("pred_h1"))
    return lgbm_z.join(lstm_z, on=keys, how="inner").select(
        "trade_date", "ts_code", "execution_date", "raw_h1", "raw_h5", "pred_h1", "pred_h5")


def _drawdown(nav: np.ndarray) -> float:
    return float(np.min(nav / np.maximum.accumulate(nav) - 1.0))


def _metrics(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    daily = pl.read_parquet(path / "portfolio_daily.parquet").sort("execution_date")
    executions = pl.read_parquet(path / "executions.parquet")
    summary = result["charged"]
    years = daily.height / 252
    final_nav = float(daily["nav"][-1])
    return {
        "days": daily.height,
        "final_nav": final_nav,
        "net_total_return": final_nav - 1,
        "net_annualized_return": final_nav ** (1 / years) - 1,
        "information_ratio": summary["information_ratio"],
        "max_drawdown": _drawdown(daily["nav"].to_numpy()),
        "total_buy_turnover": float(daily["buy_turnover"].sum()),
        "total_sell_turnover": float(daily["sell_turnover"].sum()),
        "transaction_cost_rate_sum": summary["sum_daily_transaction_cost_rate"],
        "average_holding_count": summary["average_holding_count"],
        "average_cash_weight": summary["average_cash_weight"],
        "filled_orders": executions.height,
    }


def _chart(root: Path, variants: list[str], models: list[str]) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(variants), 1, figsize=(12, 12), sharex=True)
    for axis, variant in zip(axes, variants):
        for model in models:
            frame = pl.read_parquet(root / "backtests" / variant / model / "portfolio_daily.parquet").sort("execution_date")
            axis.plot(frame["execution_date"].to_list(), frame["nav"].to_list(), label=model)
        benchmark = pl.read_parquet(root / "backtests" / variant / models[0] / "portfolio_daily.parquet").sort("execution_date")
        axis.plot(benchmark["execution_date"].to_list(), benchmark["csi500_nav"].to_list(), label="CSI500", color="black", alpha=.65)
        axis.set_title(variant)
        axis.grid(alpha=.25)
        axis.legend(ncol=4)
    fig.suptitle("CSI500 Top100: daily vs weekly replacement", fontsize=14)
    fig.tight_layout()
    output = root / "nav_comparison.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def _markdown(root: Path, rows: list[dict[str, Any]], annual_rows: list[dict[str, Any]], chart: Path) -> Path:
    lines = [
        "# 105 因子日频与周频调仓对照", "",
        "共同样本：CSI500，Top100，退出 Top120，H1/H5 各 50%，买卖成本 2.1/7.1 bp。",
        "周频在每周最后一个可交易信号日做普通排名替换；退池和单票权重上限仍每日处理。", "",
        "| 调仓 | 模型 | 年化收益 | 总收益 | IR | 最大回撤 | 买入换手合计 | 成本率合计 | 成交笔数 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['model']} | {row['net_annualized_return']:.2%} | "
            f"{row['net_total_return']:.2%} | {row['information_ratio']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['total_buy_turnover']:.2f} | {row['transaction_cost_rate_sum']:.2%} | {row['filled_orders']} |"
        )
    lines += ["", "## 分年度净收益", "", "| 调仓 | 模型 | 年份 | 净收益 |", "|---|---|---:|---:|"]
    for row in annual_rows:
        lines.append(f"| {row['variant']} | {row['model']} | {row['year']} | {row['net_return']:.2%} |")
    lines += ["", f"![净值对照]({chart.name})", ""]
    output = root / "report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    sequence = _json(args.sequence_config)
    strategy = _json(args.strategy_config)
    catalog = Path(sequence["catalog"]).expanduser()
    prediction_root = Path(sequence["output"]).expanduser() / "full" / "predictions"
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "predictions"
    prepared.mkdir(exist_ok=True)

    lstm = pl.read_parquet(prediction_root / "lstm105.parquet")
    lgbm = pl.read_parquet(prediction_root / "lgbm_default105_common.parquet")
    common = lstm.select("trade_date", "ts_code").join(lgbm.select("trade_date", "ts_code"), on=["trade_date", "ts_code"], how="inner")
    frames = {
        "lstm": lstm.join(common, on=["trade_date", "ts_code"], how="semi"),
        "lgbm": lgbm.join(common, on=["trade_date", "ts_code"], how="semi"),
        "blend_50_50": _blend(lstm, lgbm),
        "lgbm_h1_lstm_h5": _horizon_hybrid(lstm, lgbm),
    }
    prediction_paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = prepared / f"{name}_csi500.parquet"
        _csi500(catalog, frame).sort("trade_date", "ts_code").write_parquet(path, compression="zstd")
        prediction_paths[name] = path

    policy = {**strategy["portfolio"], **strategy["costs"]}
    policy.pop("strategy", None)
    base = LimitedReplacementConfig(**policy)
    variants = {
        "daily_swap3": base,
        "weekly_swap3": replace(base, rebalance_frequency="weekly"),
        "weekly_swap15": replace(base, rebalance_frequency="weekly", max_daily_replacements=15,
                                 daily_buy_budget=.50, daily_sell_budget=.50),
    }
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}
    for variant, config in variants.items():
        raw_results[variant] = {}
        for model, predictions in prediction_paths.items():
            target = output / "backtests" / variant / model
            result = run_limited_replacement_policy(catalog, predictions, target, config)
            raw_results[variant][model] = result
            rows.append({"variant": variant, "model": model, **_metrics(target, result)})
            annual = pl.read_parquet(target / "annual_metrics.parquet")
            annual_rows.extend({"variant": variant, "model": model, **row}
                               for row in annual.to_dicts())

    pl.DataFrame(rows).write_csv(output / "summary.csv")
    pl.DataFrame(annual_rows).write_csv(output / "annual_summary.csv")
    (output / "summary.json").write_text(json.dumps({"rows": rows, "annual_rows": annual_rows, "runs": raw_results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    chart = _chart(output, list(variants), list(prediction_paths))
    report = _markdown(output, rows, annual_rows, chart)
    return {"output": str(output), "summary": str(output / "summary.csv"),
            "annual_summary": str(output / "annual_summary.csv"), "report": str(report),
            "chart": str(chart), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-config", type=Path, default=Path("configs/prediction_sequence_lstm_residual_105_v2.json"))
    parser.add_argument("--strategy-config", type=Path, default=Path("configs/production_strategy_csi500_top100_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/predict/sequence-lstm-residual-daily60-minute45-v2/full/weekly_comparison"))
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
