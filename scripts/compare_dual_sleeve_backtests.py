"""Compare one dual-sleeve candidate with two CSI500-only baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def metrics(name: str, path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = pl.read_parquet(path).sort("execution_date")
    returns = frame["net_return"].to_numpy()
    benchmark = frame["csi500_return"].to_numpy()
    active = returns - benchmark
    nav = np.cumprod(1.0 + returns)
    years = len(frame) / 252.0
    row = {
        "strategy": name,
        "days": len(frame),
        "total_return": float(nav[-1] - 1.0),
        "annualized_return": float(nav[-1] ** (1.0 / years) - 1.0),
        "max_drawdown": float((nav / np.maximum.accumulate(np.r_[1.0, nav])[1:] - 1.0).min()),
        "daily_active_bps": float(active.mean() * 10_000),
        "tracking_error": float(active.std(ddof=1) * np.sqrt(252)),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252)),
        "average_buy_turnover": float(frame["buy_turnover"].mean()),
        "average_cash_weight": float(frame["cash_weight"].mean()),
        "average_holding_count": float(frame["holding_count"].mean()),
    }
    annual: list[dict[str, object]] = []
    grouped = frame.with_columns(pl.col("execution_date").dt.year().alias("year")).partition_by("year", as_dict=True)
    for key, sample in grouped.items():
        year = key[0] if isinstance(key, tuple) else key
        annual_active = sample["net_return"].to_numpy() - sample["csi500_return"].to_numpy()
        annual.append({
            "strategy": name,
            "year": year,
            "days": len(sample),
            "daily_active_bps": float(annual_active.mean() * 10_000),
            "tracking_error": float(annual_active.std(ddof=1) * np.sqrt(252)),
            "information_ratio": float(annual_active.mean() / annual_active.std(ddof=1) * np.sqrt(252)),
        })
    return row, annual


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--top100", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        ("raw 80/20 Top100/swap3 revised", args.candidate),
        ("CSI500 Top100/swap3", args.top100),
        ("CSI500 formal Top80/max5", args.formal),
    ]
    rows, annual = [], []
    frames: dict[str, pl.DataFrame] = {}
    for name, path in inputs:
        row, year_rows = metrics(name, path)
        rows.append(row)
        annual.extend(year_rows)
        frames[name] = pl.read_parquet(path).sort("execution_date")
    args.output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.output / "comparison.csv")
    pl.DataFrame(annual).sort(["year", "strategy"]).write_csv(args.output / "annual_ir.csv")
    lines = [
        "# Raw 80/20 与两个 CSI500 baseline 对比",
        "",
        "统一口径：T+1 开盘成交、日收益减 CSI500 同期开盘收益、年化 252 日、含交易费。",
        "",
        "| 策略 | 超额Sharpe/IR | 日均超额 | 跟踪误差 | 总收益 | 年化收益 | 最大回撤 | 买入换手/日 | 平均现金 | 平均持股 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['information_ratio']:.3f} | {row['daily_active_bps']:.2f} bp | "
            f"{pct(row['tracking_error'])} | {pct(row['total_return'])} | {pct(row['annualized_return'])} | "
            f"{pct(row['max_drawdown'])} | {pct(row['average_buy_turnover'])} | "
            f"{pct(row['average_cash_weight'])} | {row['average_holding_count']:.1f} |"
        )
    lines.extend(["", "## 年度超额Sharpe/IR", "", "| 年份 | raw 80/20 | CSI500 Top100 | CSI500 formal Top80 |", "|---:|---:|---:|---:|"])
    annual_frame = pl.DataFrame(annual)
    for year in sorted(annual_frame["year"].unique().to_list()):
        sample = annual_frame.filter(pl.col("year") == year)
        values = dict(sample.select("strategy", "information_ratio").iter_rows())
        lines.append(f"| {year} | {values[inputs[0][0]]:.3f} | {values[inputs[1][0]]:.3f} | {values[inputs[2][0]]:.3f} |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidate_name, baseline_name = inputs[0][0], inputs[1][0]
    candidate = frames[candidate_name]
    baseline = frames[baseline_name]
    dates = candidate["execution_date"].to_list()
    candidate_nav = candidate["nav"].to_numpy()
    baseline_nav = baseline["nav"].to_numpy()
    benchmark_nav = candidate["csi500_nav"].to_numpy()

    def drawdown(nav: np.ndarray) -> np.ndarray:
        return nav / np.maximum.accumulate(nav) - 1.0

    def rolling_ir(frame: pl.DataFrame, window: int = 252) -> np.ndarray:
        active = frame["net_return"].to_numpy() - frame["csi500_return"].to_numpy()
        result = np.full(len(active), np.nan)
        for index in range(window - 1, len(active)):
            sample = active[index - window + 1:index + 1]
            std = sample.std(ddof=1)
            result[index] = sample.mean() / std * np.sqrt(252) if std > 0 else np.nan
        return result

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].plot(dates, candidate_nav, label="Raw 80/20 revised", lw=1.8, color="#d05a3a")
    axes[0, 0].plot(dates, baseline_nav, label="CSI500 Top100/swap3", lw=1.6, color="#457b9d")
    axes[0, 0].plot(dates, benchmark_nav, label="CSI500", lw=1.2, color="#6b7280")
    axes[0, 0].set_title("Net NAV (after costs)"); axes[0, 0].legend(fontsize=9)
    axes[0, 1].plot(dates, candidate_nav / benchmark_nav, label="Raw 80/20 revised", lw=1.8, color="#d05a3a")
    axes[0, 1].plot(dates, baseline_nav / benchmark_nav, label="CSI500 Top100/swap3", lw=1.6, color="#457b9d")
    axes[0, 1].axhline(1.0, color="#6b7280", lw=.8)
    axes[0, 1].set_title("Relative wealth vs CSI500"); axes[0, 1].legend(fontsize=9)
    axes[1, 0].plot(dates, drawdown(candidate_nav), label="Raw 80/20 revised", lw=1.5, color="#d05a3a")
    axes[1, 0].plot(dates, drawdown(baseline_nav), label="CSI500 Top100/swap3", lw=1.4, color="#457b9d")
    axes[1, 0].set_title("Drawdown"); axes[1, 0].legend(fontsize=9)
    axes[1, 1].plot(dates, rolling_ir(candidate), label="Raw 80/20 revised", lw=1.5, color="#d05a3a")
    axes[1, 1].plot(dates, rolling_ir(baseline), label="CSI500 Top100/swap3", lw=1.4, color="#457b9d")
    axes[1, 1].axhline(0.0, color="#6b7280", lw=.8)
    axes[1, 1].set_title("Rolling 252-day excess Sharpe / IR"); axes[1, 1].legend(fontsize=9)
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Raw CSI500/CSI1000 80/20 vs CSI500 Top100/swap3", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
