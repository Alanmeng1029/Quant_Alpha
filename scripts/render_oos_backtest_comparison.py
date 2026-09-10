#!/usr/bin/env python3
"""Render a self-contained HTML comparison for two charged OOS backtests."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "98 factors": (ROOT / "results/predict/research-oos-daily60-minute38-v1-recheck-20260909", "lgbm_default98recheck"),
    "105 factors": (ROOT / "results/predict/research-oos-daily60-minute45-v2", "lgbm_default105"),
}
OUTPUT = ROOT / "results/predict/research-oos-daily60-minute45-v2/comparison-98-vs-105.html"


def _report(run: Path, model: str) -> dict:
    return json.loads((run / "report.json").read_text(encoding="utf-8"))["backtests"][model]["charged_report"]


def _daily(run: Path, model: str) -> pl.DataFrame:
    return pl.read_parquet(run / f"backtests/{model}/portfolio_daily.parquet").select(
        "execution_date", "nav", "csi500_nav", "active_return", "net_return"
    ).sort("execution_date")


def _series(frame: pl.DataFrame) -> list[dict[str, float | str]]:
    return [
        {"date": row[0].isoformat(), "nav": round(row[1], 8), "benchmark": round(row[2], 8)}
        for row in frame.iter_rows()
    ]


def _annual(frame: pl.DataFrame) -> list[dict[str, float | int]]:
    return [
        {"year": int(year[0]), "return": round(float((group["net_return"] + 1).product() - 1), 8)}
        for year, group in frame.with_columns(pl.col("execution_date").dt.year().alias("year")).group_by("year", maintain_order=True)
    ]


def main() -> None:
    reports = {name: _report(*source) for name, source in RUNS.items()}
    daily = {name: _daily(*source) for name, source in RUNS.items()}
    common_dates = set(daily["98 factors"]["execution_date"].to_list()) & set(daily["105 factors"]["execution_date"].to_list())
    daily = {name: frame.filter(pl.col("execution_date").is_in(common_dates)) for name, frame in daily.items()}
    payload = {
        "series": {name: _series(frame) for name, frame in daily.items()},
        "annual": {name: _annual(frame) for name, frame in daily.items()},
        "metrics": {
            name: {
                "net_total_return": report["net_total_return"],
                "net_annualized_return": report["net_annualized_return"],
                "net_annualized_excess": report["net_annualized_excess_vs_csi500"],
                "information_ratio": report["information_ratio"],
                "max_drawdown": report["max_drawdown"],
                "turnover": report["average_buy_turnover"],
                "cost": report["cumulative_fee_paid_on_initial_nav"],
            }
            for name, report in reports.items()
        },
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    OUTPUT.write_text(f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>98 vs 105 因子：滚动 OOS 回测对比</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--grid:#d9e2ef;--old:#577590;--new:#d05a3a;--bench:#8c95a5;--paper:#fff;--bg:#f6f8fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}main{{max-width:1180px;margin:auto;padding:34px 24px 56px}}h1{{margin:0;font-size:28px}}.sub{{color:var(--muted);margin:8px 0 26px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}.card{{background:var(--paper);border:1px solid var(--grid);border-radius:10px;padding:15px}}.label{{font-size:13px;color:var(--muted)}}.value{{font-size:24px;font-weight:700;margin-top:5px}}.delta{{font-size:13px;margin-top:6px;color:var(--new)}}section{{background:var(--paper);border:1px solid var(--grid);border-radius:10px;padding:18px;margin:16px 0}}h2{{font-size:17px;margin:0 0 12px}}.legend{{display:flex;gap:17px;font-size:13px;color:var(--muted);margin-bottom:6px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:99px;margin-right:5px}}svg{{width:100%;height:270px;display:block;overflow:visible}}.axis{{font-size:11px;fill:var(--muted)}}.line-old{{fill:none;stroke:var(--old);stroke-width:2}}.line-new{{fill:none;stroke:var(--new);stroke-width:2.4}}.line-bench{{fill:none;stroke:var(--bench);stroke-width:1.5;stroke-dasharray:5 4}}.grid{{stroke:var(--grid);stroke-width:1}}.tooltip{{position:fixed;display:none;background:#172033;color:#fff;padding:8px 10px;border-radius:6px;font-size:12px;pointer-events:none;line-height:1.5;z-index:2}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 8px;border-bottom:1px solid var(--grid);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600}}.old{{color:var(--old)}}.new{{color:var(--new);font-weight:700}}.note{{font-size:13px;color:var(--muted);line-height:1.6}}@media(max-width:720px){{main{{padding:22px 14px}}.cards{{grid-template-columns:repeat(2,1fr)}}svg{{height:230px}}h1{{font-size:23px}}}}
</style></head><body><main>
<h1>默认 LGBM：98 因子 vs 105 因子</h1><p class=\"sub\">相同数据快照、交易日历、季度滚动训练和 Optimizer V2 含成本口径 · 2021-04-02 至 2026-08-27</p>
<div class=\"cards\" id=\"cards\"></div>
<section><h2>含成本净值</h2><div class=\"legend\"><span><i class=\"dot\" style=\"background:var(--old)\"></i>旧 98 因子</span><span><i class=\"dot\" style=\"background:var(--new)\"></i>新 105 因子</span><span><i class=\"dot\" style=\"background:var(--bench)\"></i>CSI500</span></div><svg id=\"nav\" role=\"img\" aria-label=\"含成本净值比较\"></svg></section>
<section><h2>相对 CSI500 的累计超额净值</h2><svg id=\"active\" role=\"img\" aria-label=\"累计超额净值比较\"></svg></section>
<section><h2>净值回撤</h2><svg id=\"drawdown\" role=\"img\" aria-label=\"回撤比较\"></svg></section>
<section><h2>年度含成本收益</h2><svg id=\"annual\" role=\"img\" aria-label=\"年度收益比较\"></svg></section>
<section><h2>关键指标</h2><table id=\"metrics\"></table><p class=\"note\">新模型提高了收益、超额收益和信息比率；回撤扩大。因子是用同段时期的 ICIR 筛选，后续仍应以留出期验证稳定性。</p></section>
</main><div class=\"tooltip\" id=\"tip\"></div><script>const D={data};
const fmt=p=>`${{(p*100).toFixed(2)}}%`, num=x=>x.toFixed(3), old='98 factors', neu='105 factors';
const m=D.metrics, cards=[['净总收益',m[old].net_total_return,m[neu].net_total_return,fmt],['净年化收益',m[old].net_annualized_return,m[neu].net_annualized_return,fmt],['年化超额收益',m[old].net_annualized_excess,m[neu].net_annualized_excess,fmt],['信息比率',m[old].information_ratio,m[neu].information_ratio,num]];
document.getElementById('cards').innerHTML=cards.map(([l,a,b,f])=>`<div class=\"card\"><div class=\"label\">${{l}}</div><div class=\"value\">${{f(b)}}</div><div class=\"delta\">较旧版 ${{f(b-a)}} </div></div>`).join('');
const S='http://www.w3.org/2000/svg', mk=(tag,a={{}})=>{{let e=document.createElementNS(S,tag);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));return e}}, line=(svg,sets,field,zero=false)=>{{const W=svg.clientWidth||1000,H=svg.clientHeight||270,L=56,R=18,T=12,B=28,all=sets.flatMap(s=>s.values.map(field)),lo=Math.min(...all),hi=Math.max(...all);let min=zero?Math.min(0,lo):lo,max=zero?Math.max(0,hi):hi,p=(max-min)*.08||.1;min-=p;max+=p;svg.replaceChildren();for(let i=0;i<5;i++){{let v=min+(max-min)*i/4,y=T+(H-T-B)*(1-(v-min)/(max-min));svg.append(mk('line',{{x1:L,y1:y,x2:W-R,y2:y,class:'grid'}}));let t=mk('text',{{x:L-7,y:y+4,'text-anchor':'end',class:'axis'}});t.textContent=fmt(v);svg.append(t)}}const x=i=>L+i*(W-L-R)/(sets[0].values.length-1),y=v=>T+(H-T-B)*(1-(v-min)/(max-min));sets.forEach(s=>{{svg.append(mk('path',{{d:s.values.map((v,i)=>(i?'L':'M')+x(i)+','+y(field(v))).join(' '),class:s.cls}}))}});let hit=mk('rect',{{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'}});hit.addEventListener('mousemove',e=>{{let i=Math.max(0,Math.min(sets[0].values.length-1,Math.round((e.offsetX-L)/(W-L-R)*(sets[0].values.length-1))));let rows=sets.map(s=>`${{s.name}}: ${{fmt(field(s.values[i]))}}`).join('<br>');tip.innerHTML=sets[0].values[i].date+'<br>'+rows;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'}});hit.addEventListener('mouseleave',()=>tip.style.display='none');svg.append(hit)}};
const base=[{{name:'旧 98 因子',values:D.series[old],cls:'line-old'}},{{name:'新 105 因子',values:D.series[neu],cls:'line-new'}},{{name:'CSI500',values:D.series[old],cls:'line-bench'}}];line(document.getElementById('nav'),base,v=>v===base[2].values?0:v.nav); // replaced below
line(document.getElementById('nav'),[{{name:'旧 98 因子',values:D.series[old],cls:'line-old'}},{{name:'新 105 因子',values:D.series[neu],cls:'line-new'}},{{name:'CSI500',values:D.series[old].map(v=>({{...v,nav:v.benchmark}})),cls:'line-bench'}}],v=>v.nav);
line(document.getElementById('active'),[{{name:'旧 98 因子',values:D.series[old].map(v=>({{...v,nav:v.nav/v.benchmark-1}})),cls:'line-old'}},{{name:'新 105 因子',values:D.series[neu].map(v=>({{...v,nav:v.nav/v.benchmark-1}})),cls:'line-new'}}],v=>v.nav,true);
function dd(a){{let peak=-Infinity;return a.map(v=>{{peak=Math.max(peak,v.nav);return {{...v,nav:v.nav/peak-1}}}})}}line(document.getElementById('drawdown'),[{{name:'旧 98 因子',values:dd(D.series[old]),cls:'line-old'}},{{name:'新 105 因子',values:dd(D.series[neu]),cls:'line-new'}}],v=>v.nav,true);
function bars(){{const svg=document.getElementById('annual'),W=svg.clientWidth||1000,H=svg.clientHeight||270,L=56,R=18,T=12,B=32,years=D.annual[old].map(x=>x.year),vals=[...D.annual[old],...D.annual[neu]].map(x=>x.return),min=Math.min(0,...vals)*1.15,max=Math.max(0,...vals)*1.15,y=v=>T+(H-T-B)*(1-(v-min)/(max-min));svg.replaceChildren();[min,0,max].forEach(v=>{{let yy=y(v);svg.append(mk('line',{{x1:L,y1:yy,x2:W-R,y2:yy,class:'grid'}}));let t=mk('text',{{x:L-7,y:yy+4,'text-anchor':'end',class:'axis'}});t.textContent=fmt(v);svg.append(t)}});let step=(W-L-R)/years.length;years.forEach((yr,i)=>{{let a=D.annual[old][i].return,b=D.annual[neu][i].return,x=L+i*step;[[a,'var(--old)'],[b,'var(--new)']].forEach(([v,c],j)=>svg.append(mk('rect',{{x:x+step*(.19+j*.22),y:Math.min(y(v),y(0)),width:step*.17,height:Math.abs(y(v)-y(0)),fill:c}})));let t=mk('text',{{x:x+step/2,y:H-9,'text-anchor':'middle',class:'axis'}});t.textContent=yr;svg.append(t)}})}}bars();
const rows=[['H1 年化 Rank ICIR','icir_h1',num],['H5 年化 Rank ICIR','icir_h5',num],['最大回撤','max_drawdown',fmt],['日均买入换手','turnover',fmt],['累计交易成本','cost',fmt]];document.getElementById('metrics').innerHTML='<tr><th>指标</th><th class=\"old\">旧 98</th><th class=\"new\">新 105</th><th>差异</th></tr>'+rows.map(([n,k,f])=>`<tr><td>${{n}}</td><td>${{f(m[old][k])}}</td><td>${{f(m[neu][k])}}</td><td>${{f(m[neu][k]-m[old][k])}}</td></tr>`).join('');
new ResizeObserver(()=>{{line(document.getElementById('nav'),[{{name:'旧 98 因子',values:D.series[old],cls:'line-old'}},{{name:'新 105 因子',values:D.series[neu],cls:'line-new'}},{{name:'CSI500',values:D.series[old].map(v=>({{...v,nav:v.benchmark}})),cls:'line-bench'}}],v=>v.nav);line(document.getElementById('active'),[{{name:'旧 98 因子',values:D.series[old].map(v=>({{...v,nav:v.nav/v.benchmark-1}})),cls:'line-old'}},{{name:'新 105 因子',values:D.series[neu].map(v=>({{...v,nav:v.nav/v.benchmark-1}})),cls:'line-new'}}],v=>v.nav,true);line(document.getElementById('drawdown'),[{{name:'旧 98 因子',values:dd(D.series[old]),cls:'line-old'}},{{name:'新 105 因子',values:dd(D.series[neu]),cls:'line-new'}}],v=>v.nav,true);bars()}}).observe(document.body);
</script></body></html>""", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
