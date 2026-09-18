#!/usr/bin/env python3
"""Compare CSI500 and CSI1000 on the exact open-to-next-open backtest calendar."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


ANNUAL_DAYS = 243


def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peaks))


def metrics(frame: pl.DataFrame) -> dict[str, float]:
    r500 = frame["csi500_return"].to_numpy()
    r1000 = frame["csi1000_return"].to_numpy()
    active = r1000 - r500
    years = len(frame) / ANNUAL_DAYS
    excess_wealth = np.cumprod(1.0 + active)
    return {
        "days": len(frame),
        "csi500_total_return": float(np.prod(1.0 + r500) - 1.0),
        "csi1000_total_return": float(np.prod(1.0 + r1000) - 1.0),
        "csi500_annual_return": float(np.prod(1.0 + r500) ** (1.0 / years) - 1.0),
        "csi1000_annual_return": float(np.prod(1.0 + r1000) ** (1.0 / years) - 1.0),
        "csi500_max_drawdown": max_drawdown(r500),
        "csi1000_max_drawdown": max_drawdown(r1000),
        "csi1000_minus_csi500_total": float(excess_wealth[-1] - 1.0),
        "csi1000_minus_csi500_annual": float(excess_wealth[-1] ** (1.0 / years) - 1.0),
        "csi1000_minus_csi500_sharpe": float(active.mean() / active.std(ddof=1) * np.sqrt(ANNUAL_DAYS)),
        "csi1000_minus_csi500_max_drawdown": max_drawdown(active),
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def rolling_compound(returns: np.ndarray, window: int) -> np.ndarray:
    logs = np.log1p(returns)
    cumulative = np.concatenate(([0.0], np.cumsum(logs)))
    out = np.full(len(returns), np.nan)
    out[window - 1:] = np.expm1(cumulative[window:] - cumulative[:-window])
    return out


def load_aligned(args: argparse.Namespace) -> pl.DataFrame:
    portfolio = pl.read_parquet(args.portfolio).select("execution_date", "next_execution_date", "net_return")
    start = portfolio["execution_date"].min()
    end = portfolio["next_execution_date"].max()
    with duckdb.connect(args.catalog, read_only=True) as conn:
        csi500 = pl.from_arrow(
            conn.execute(
                """SELECT trade_date, open FROM index_daily
                   WHERE index_code='000905.SH' AND trade_date BETWEEN ? AND ? AND open>0
                   ORDER BY trade_date""",
                [start, end],
            ).arrow()
        ).with_columns(pl.col("trade_date").cast(pl.Date))
    csi1000 = pl.read_parquet(args.csi1000_bars).select("trade_date", "open").with_columns(
        pl.col("trade_date").cast(pl.Date)
    )
    return (
        portfolio
        .join(csi500.rename({"trade_date": "execution_date", "open": "csi500_open"}), on="execution_date")
        .join(csi500.rename({"trade_date": "next_execution_date", "open": "csi500_next_open"}), on="next_execution_date")
        .join(csi1000.rename({"trade_date": "execution_date", "open": "csi1000_open"}), on="execution_date")
        .join(csi1000.rename({"trade_date": "next_execution_date", "open": "csi1000_next_open"}), on="next_execution_date")
        .with_columns(
            (pl.col("csi500_next_open") / pl.col("csi500_open") - 1.0).alias("csi500_return"),
            (pl.col("csi1000_next_open") / pl.col("csi1000_open") - 1.0).alias("csi1000_return"),
        )
        .with_columns((pl.col("csi1000_return") - pl.col("csi500_return")).alias("active_return"))
        .sort("execution_date")
    )


def plot(frame: pl.DataFrame, output: Path) -> None:
    dates = frame["execution_date"].to_numpy()
    r500 = frame["csi500_return"].to_numpy()
    r1000 = frame["csi1000_return"].to_numpy()
    active = r1000 - r500
    nav500 = np.cumprod(1.0 + r500)
    nav1000 = np.cumprod(1.0 + r1000)
    active_nav = np.cumprod(1.0 + active)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].plot(dates, nav500, label="CSI500", lw=1.6)
    axes[0, 0].plot(dates, nav1000, label="CSI1000", lw=1.6)
    axes[0, 0].set_title("Full-period open-to-next-open cumulative wealth")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=.25)

    cutoff = np.datetime64("2025-01-01")
    mask = dates >= cutoff
    d25 = dates[mask]
    r500_25 = r500[mask]
    r1000_25 = r1000[mask]
    axes[0, 1].plot(d25, np.cumprod(1.0 + r500_25), label="CSI500", lw=1.6)
    axes[0, 1].plot(d25, np.cumprod(1.0 + r1000_25), label="CSI1000", lw=1.6)
    axes[0, 1].set_title("Rebased at 2025-01-01")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=.25)

    axes[1, 0].plot(dates, active_nav, color="#7b3294", lw=1.6)
    axes[1, 0].axvline(cutoff, color="#c43c39", ls="--", lw=1)
    axes[1, 0].axhline(1.0, color="#555", ls=":", lw=1)
    axes[1, 0].set_title("CSI1000 excess wealth vs CSI500: compound(r1000-r500)")
    axes[1, 0].grid(alpha=.25)

    roll500 = rolling_compound(r500, ANNUAL_DAYS)
    roll1000 = rolling_compound(r1000, ANNUAL_DAYS)
    axes[1, 1].plot(dates, roll500, label="CSI500 rolling 243d", lw=1.4)
    axes[1, 1].plot(dates, roll1000, label="CSI1000 rolling 243d", lw=1.4)
    axes[1, 1].axvline(cutoff, color="#c43c39", ls="--", lw=1)
    axes[1, 1].axhline(0.0, color="#555", ls=":", lw=1)
    axes[1, 1].set_title("Rolling 243-day compounded return")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=.25)
    fig.savefig(output / "csi500_csi1000_comparison.png", dpi=160)
    plt.close(fig)


def build_report(frame: pl.DataFrame, output: Path) -> None:
    yearly = []
    for (year,), group in frame.with_columns(pl.col("execution_date").dt.year().alias("year")).group_by("year", maintain_order=True):
        label = str(year)
        if year == 2021:
            label += " (from Apr)"
        elif year == 2026:
            label += " (through Aug)"
        row = {"period": label, **metrics(group)}
        yearly.append(row)
    pre = frame.filter(pl.col("execution_date") < pl.date(2025, 1, 1))
    post = frame.filter(pl.col("execution_date") >= pl.date(2025, 1, 1))
    periods = [
        {"period": "2021-04 to 2024-12", **metrics(pre)},
        {"period": "2025-01 onward", **metrics(post)},
        {"period": "Full OOS", **metrics(frame)},
    ]
    strategy_periods = []
    for label, sample in (("2021-04 to 2024-12", pre), ("2025-01 onward", post), ("Full OOS", frame)):
        strategy = sample["net_return"].to_numpy()
        r500 = sample["csi500_return"].to_numpy()
        r1000 = sample["csi1000_return"].to_numpy()
        active500 = strategy - r500
        active1000 = strategy - r1000
        strategy_periods.append({
            "period": label,
            "strategy_return": float(np.prod(1.0 + strategy) - 1.0),
            "excess_vs_csi500": float(np.prod(1.0 + active500) - 1.0),
            "excess_sharpe_vs_csi500": float(active500.mean() / active500.std(ddof=1) * np.sqrt(ANNUAL_DAYS)),
            "excess_vs_csi1000": float(np.prod(1.0 + active1000) - 1.0),
            "excess_sharpe_vs_csi1000": float(active1000.mean() / active1000.std(ddof=1) * np.sqrt(ANNUAL_DAYS)),
        })
    payload = {"periods": periods, "pure_csi1000_top100": strategy_periods, "annual": yearly}
    (output / "index_comparison_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    cols = [
        ("period", "Period", str),
        ("csi500_total_return", "CSI500 return", pct),
        ("csi1000_total_return", "CSI1000 return", pct),
        ("csi1000_minus_csi500_total", "1000 excess wealth", pct),
        ("csi1000_minus_csi500_sharpe", "1000 excess Sharpe", lambda x: f"{x:.3f}"),
        ("csi1000_minus_csi500_max_drawdown", "1000 excess MDD", pct),
    ]
    def table(rows: list[dict[str, float]]) -> str:
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in cols)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(row[key]))}</td>" for key, _, fmt in cols) + "</tr>" for row in rows)
        return f"<table><tr>{head}</tr>{body}</table>"

    strategy_cols = [
        ("period", "Period", str),
        ("strategy_return", "Top100 return", pct),
        ("excess_vs_csi500", "Excess vs CSI500", pct),
        ("excess_sharpe_vs_csi500", "Sharpe vs CSI500", lambda x: f"{x:.3f}"),
        ("excess_vs_csi1000", "Excess vs CSI1000", pct),
        ("excess_sharpe_vs_csi1000", "Sharpe vs CSI1000", lambda x: f"{x:.3f}"),
    ]
    def strategy_table(rows: list[dict[str, float]]) -> str:
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in strategy_cols)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(row[key]))}</td>" for key, _, fmt in strategy_cols) + "</tr>" for row in rows)
        return f"<table><tr>{head}</tr>{body}</table>"

    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>CSI500 vs CSI1000 index comparison</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#18212f}}table{{border-collapse:collapse}}th,td{{padding:7px 12px;border:1px solid #d9e1ea;text-align:left}}img{{display:block;max-width:1200px;width:100%;margin:20px 0}}</style></head><body><h1>CSI500 vs CSI1000 index return comparison</h1><p>Return basis: each backtest execution-date open to next execution-date open; annual factor: 243; excess curve compounds daily r(CSI1000)-r(CSI500).</p><h2>Key periods</h2>{table(periods)}<img src='csi500_csi1000_comparison.png'><h2>Pure CSI1000 Top100/swap5 decomposition</h2>{strategy_table(strategy_periods)}<h2>Annual breakdown</h2>{table(yearly)}</body></html>"""
    (output / "report.html").write_text(doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--csi1000-bars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_aligned(args)
    frame.write_parquet(args.output / "aligned_index_returns.parquet", compression="zstd")
    frame.select("execution_date", "next_execution_date", "csi500_return", "csi1000_return", "active_return").write_csv(
        args.output / "aligned_index_returns.csv"
    )
    plot(frame, args.output)
    build_report(frame, args.output)


if __name__ == "__main__":
    main()
