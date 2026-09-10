#!/usr/bin/env python3
"""Render the risk-control selection study as a compact local HTML report."""
from __future__ import annotations
import html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/predict/risk-control-study"

def pct(value: float) -> str: return f"{value:.2%}"

def main() -> None:
    report = json.loads((OUT / "summary.json").read_text())
    base = report["variants"][0]["metrics"]
    rows = []
    for variant in report["variants"]:
        m = variant["metrics"]
        rows.append(f"<tr><td>{html.escape(variant['name'])}</td><td>{pct(m['net_return'])}</td><td>{pct(m['annual_return'])}</td><td>{pct(m['annual_excess'])}</td><td>{pct(m['max_drawdown'])}</td><td>{m['information_ratio']:.3f}</td><td>{pct(m['average_buy_turnover'])}</td><td>{pct(m['annual_return']-base['annual_return'])}</td></tr>")
    risk = json.loads((OUT / "style_neutral/risk_attribution/summary.json").read_text())
    exposure = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.3f}</td></tr>" for k,v in risk["mean_active_exposure"].items())
    document = f"""<!doctype html><meta charset='utf-8'><title>风险控制选股研究</title><style>body{{font:15px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:1080px;margin:32px auto;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 30px}}td,th{{padding:9px;border:1px solid #d6deeb;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#edf3fc}}.note{{color:#536176;line-height:1.65}}</style><body><h1>CSI500 Top100 风险控制选股研究</h1><p>固定交易规则：Top100、最多每日换 3 只、等权入场、相同开盘成交和成本。样本：2021-04 至 2026-08。</p><table><tr><th>版本</th><th>累计净收益</th><th>年化收益</th><th>年化超额</th><th>最大回撤</th><th>IR</th><th>日买入换手</th><th>相对基准年化</th></tr>{''.join(rows)}</table><h2>风格中性版本的平均主动暴露</h2><table><tr><th>因子</th><th>主动暴露</th></tr>{exposure}</table><p class='note'>风格中性化是在每个信号日将 LGBM 的 H1/H5 合成 alpha 对 Size、Beta、特异波动、动量、流动性和换手做截面残差化；行业版本额外加入静态行业哑变量。它降低选股的风险相关部分，但有限换仓和权重漂移意味着它不是严格的持仓约束优化。</p></body>"""
    (OUT / "report.html").write_text(document, encoding="utf-8")

if __name__ == "__main__": main()
