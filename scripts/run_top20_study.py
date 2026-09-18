"""Compare Top20 variants against the current production portfolio.
Run: PYTHONPATH=src python scripts/run_top20_study.py
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import numpy as np
import polars as pl
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import render_backtest_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/predict/top20-study'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    production = json.loads((ROOT / 'configs/production_strategy_csi500_top100_v1.json').read_text())
    base = LimitedReplacementConfig(**{k:v for k,v in production['portfolio'].items() if k != 'strategy'}, **production['costs'])
    source = ROOT / production['prediction_artifact']
    catalog = ROOT / 'A_stock_database/lake/catalog/a_share.duckdb'
    with duckdb.connect(str(catalog), read_only=True) as conn:
        universe = pl.from_arrow(conn.execute("SELECT DISTINCT trade_date,ts_code FROM index_trading_universe WHERE index_code='000905.SH'").arrow())
    raw = pl.read_parquet(source)
    predictions = raw.join(universe, on=['trade_date','ts_code'], how='semi').sort(['trade_date','ts_code'])
    assert predictions['trade_date'].unique().sort().equals(raw['trade_date'].unique().sort())
    filtered = OUT / 'csi500_predictions.parquet'
    predictions.write_parquet(filtered)
    configs = {'top100_production':base,
               'top20_scaled_cap':replace(base,target_holdings=20,entry_rank=20,exit_rank=24,max_weight=0.15,rebalance_to_weight=0.14),
               'top20_original_cap':replace(base,target_holdings=20,entry_rank=20,exit_rank=24)}
    (OUT/'protocol.json').write_text(json.dumps({'production':production,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cap_note':'Caps checked before buying; new entries can exceed cap until next session. Scaled caps preserve original cap/equal-entry ratio.','changes':'Top20 / exit24 / max3 replacements; all remaining settings inherited; no retraining.'},indent=2))
    rows=[]; annual=[]; frames={}; reference=None
    for name, config in configs.items():
        dest=OUT/name
        result=run_limited_replacement_policy(catalog,filtered,dest,config)
        f=pl.read_parquet(dest/'portfolio_daily.parquet')
        assert f.height>0 and f['holding_count'].max()<=config.target_holdings
        assert f['cash_weight'].min()>=-1e-8
        assert np.isfinite(f['nav'].to_numpy()).all()
        dates=f.select('execution_date','next_execution_date')
        if reference is None: reference=dates
        else: assert dates.equals(reference)
        frames[name]=f
        n=f['nav'].to_numpy(); r=f['net_return'].to_numpy(); b=f['csi500_return'].to_numpy()
        row={'name':name,'days':f.height,'total_return':n[-1]-1,'annual_return':n[-1]**(252/f.height)-1,'max_drawdown':(n/np.maximum.accumulate(np.r_[1,n])[1:]-1).min(),'information_ratio':(r-b).mean()/(r-b).std(ddof=1)*np.sqrt(252),'buy_turnover':f['buy_turnover'].mean(),'cash_weight':f['cash_weight'].mean(),'holding_count':f['holding_count'].mean(),'zero_cost_return':result['zero_cost']['net_total_return']}
        rows.append(row)
        for key,g in f.group_by(pl.col('execution_date').dt.year().alias('year'),maintain_order=True):
            annual.append({'name':name,'year':key[0],'return':float(np.prod(1+g['net_return'].to_numpy())-1),'csi500_return':float(np.prod(1+g['csi500_return'].to_numpy())-1)})
        render_backtest_report(dest/'portfolio_daily.parquet',dest/'report',name)
        print(json.dumps(row),flush=True)
    old=pl.read_parquet(ROOT/'results/predict/universe-size-study/csi500_top100_swap3/portfolio_daily.parquet')
    assert frames['top100_production'].select('execution_date','nav').equals(old.select('execution_date','nav')), 'Production baseline changed'
    pl.DataFrame(rows).write_csv(OUT/'summary.csv'); pl.DataFrame(annual).write_csv(OUT/'annual.csv')
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2))
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for name,f in frames.items():
        n=f['nav'].to_numpy(); dates=f['execution_date'].to_list()
        axs[0].plot(dates,n,label=name)
        axs[1].plot(dates,100*(n/np.maximum.accumulate(np.r_[1,n])[1:]-1))
        axs[2].plot(dates,100*f['cash_weight'].to_numpy(),alpha=.7)
    f=frames['top100_production'];axs[0].plot(f['execution_date'].to_list(),f['csi500_nav'],label='CSI500',color='gray',ls='--')
    for ax,label in zip(axs,['Net NAV','Drawdown (%)','Cash (%)']): ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0].legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    lines=['# Top20 与当前生产基线回测','',f"执行区间：{f['execution_date'].min()} 至 {f['execution_date'].max()}；最终估值：{f['next_execution_date'].max()}。",'', '复用105因子默认LGBM，信号日历史CSI500过滤。Top20入选、Top24退出，每日最多主动替换3只；初始1000万元，100股整手，买入2.1bp/卖出7.1bp，每日买卖预算各10%。生产配置未修改。','', '同比放大组：单票上限15%、减仓至14%，保留生产上限相对于等额新仓的比例；原上限组：3%/2.8%。上限在买入前检查，新仓可能超限至次日。实际仓位受现金、每日预算、整手和持仓漂移影响。','', '| 组合 | 净累计 | 净年化 | 最大回撤 | IR | 日均买入换手 | 平均现金 | 平均持股 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['name']} | {r['total_return']:.2%} | {r['annual_return']:.2%} | {r['max_drawdown']:.2%} | {r['information_ratio']:.3f} | {r['buy_turnover']:.2%} | {r['cash_weight']:.2%} | {r['holding_count']:.2f} |")
    lines+=['','![对照图](comparison.png)','','年度拆分见 annual.csv，汇总见 summary.csv；各组目录包含完整交易、持仓账本及标准HTML报告。','', '验证：三组执行日一致、持股不超过目标、现金非负、净值有限；重新回测Top100的每日净值与既有生产账本完全一致。','', '该区间已用于多轮研究，结果属于同样本探索。未完整模拟涨跌停、历史ST、市场冲击与容量；不是严格每日重选Top20。','', '复现：`MPLCONFIGDIR=/tmp/quant-alpha-mpl PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python scripts/run_top20_study.py`']
    (OUT/'report.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
