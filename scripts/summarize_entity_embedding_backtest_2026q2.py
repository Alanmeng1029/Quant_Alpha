"""Create the matched 2026Q2 backtest comparison table and NAV chart."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


ROOT = Path("results/predict/sequence-lstm-residual-raw125-v1/entity_embedding_pilot_v2/month=2026-04/backtest_comparison")
NAMES = {"baseline": "Baseline LSTM", "entity": "Industry + liquidity LSTM", "lgbm": "Formal LGBM"}
metrics = []
daily = {}
for key, label in NAMES.items():
    report = json.loads((ROOT / key / "blend/backtest/report/report_summary.json").read_text())
    metrics.append({
        "model": label,
        "net_total_return": report["net_total_return"],
        "benchmark_total_return": report["benchmark_total_return"],
        "excess_total_return": report["excess_curve_total_return"],
        "net_sharpe": report["net_sharpe"],
        "excess_sharpe_243": report["excess_sharpe_243"],
        "max_drawdown": report["max_drawdown"],
        "average_buy_turnover": report["average_buy_turnover"],
        "average_fee_bps": report["average_fee_bps"],
    })
    daily[label] = pl.read_parquet(ROOT / key / "blend/backtest/portfolio_daily.parquet").sort("execution_date")

comparison = pl.DataFrame(metrics)
comparison.write_csv(ROOT / "portfolio_comparison.csv")
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for label, frame in daily.items():
    axes[0].plot(frame["execution_date"], frame["nav"], label=label)
    axes[1].plot(frame["execution_date"], frame["buy_turnover"].rolling_mean(5), label=label)
axes[0].set_title("2026Q2 net NAV, seed 20260908")
axes[1].set_title("5-day average buy turnover")
for axis in axes:
    axis.grid(alpha=.3)
    axis.legend()
fig.tight_layout()
fig.savefig(ROOT / "portfolio_comparison.png", dpi=160)
plt.close(fig)

rows = []
for row in comparison.iter_rows(named=True):
    rows.append("| " + " | ".join([
        row["model"], f'{row["net_total_return"]:.2%}', f'{row["excess_total_return"]:.2%}',
        f'{row["net_sharpe"]:.3f}', f'{row["excess_sharpe_243"]:.3f}',
        f'{row["max_drawdown"]:.2%}', f'{row["average_buy_turnover"]:.2%}',
        f'{row["average_fee_bps"]:.3f}',
    ]) + " |")
(ROOT / "README.md").write_text("\n".join([
    "# 2026Q2 embedding LSTM 成本后回测",
    "",
    "使用默认种子 `20260908`，未做多种子集成。三个模型采用完全相同的测试样本和组合构造：80% 中证500核心仓（1%单名上限）＋20%联合池集中仓（5%单名上限），总投资比例98%，买卖各2bp。回测为2026-04-02至2026-06-30，共59个收益日。",
    "",
    "| Model | Net return | Excess return | Net Sharpe | Excess Sharpe | Max drawdown | Buy turnover | Fee bps/day |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    *rows,
    "",
    "![Portfolio comparison](portfolio_comparison.png)",
    "",
    "embedding LSTM 相对配对基线的净收益提高1.28个百分点，超额Sharpe提高0.499，最大回撤改善0.24个百分点，并略微降低换手。该结论只覆盖一个季度，不能替代完整滚动样本外回测。",
]) + "\n")
print(comparison)
