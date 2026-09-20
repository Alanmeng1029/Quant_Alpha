#!/usr/bin/env python3
"""Render the 105/125 execution-control comparison into one HTML report."""

from __future__ import annotations

import html
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1] / "results/predict/regression-105-125-execution-2bps-v1"


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    frame = pl.read_csv(ROOT / "comparison.csv")
    rows = []
    for row in frame.sort("period", "model", "method").iter_rows(named=True):
        report = f"{row['model']}/{row['method']}/report/report.html"
        cells = [
            row["period"], row["model"], row["method"], pct(row["net_return"]),
            pct(row["csi500_return"]), pct(row["relative_return"]),
            f"{row['information_ratio']:.3f}", pct(row["max_drawdown"]),
            pct(row["average_buy_turnover"]), pct(row["average_sell_turnover"]),
            f"{row['average_holding_count']:.1f}",
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells)
                    + f"<td><a href='{html.escape(report)}'>完整报告</a></td></tr>")
    headers = ["区间", "模型", "执行方式", "净收益", "CSI500", "净超额", "IR", "最大回撤",
               "日均买入换手", "日均卖出换手", "平均持仓", "详情"]
    page = f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>105/125执行方式对比</title>
<style>body{{font:15px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:1400px;margin:32px auto;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce2ea;text-align:right}}th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}th{{background:#eef3fb;position:sticky;top:0}}p{{line-height:1.65}}.scroll{{overflow:auto}}</style>
<body><h1>旧105 / 新125：执行与换手控制（双边各2bps）</h1>
<p>统一使用原回归模型 H1、信号日历史 CSI500 成分、T+1 开盘执行。Swap3 为 Top100/Exit120、每日最多替换3只；Optimizer 为无风险项的 mu 减双边换手成本、98%投资、单票上限1%；无控制为每日当日 Top100、98%等权并完整再平衡。</p>
<div class='scroll'><table><thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>所有数值均为扣除买入2bps和卖出2bps后的真实持仓账本结果。历史10bps结果未覆盖。</p></body></html>"""
    (ROOT / "report.html").write_text(page, encoding="utf-8")
    print(ROOT / "report.html")


if __name__ == "__main__":
    main()
