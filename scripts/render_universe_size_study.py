from pathlib import Path
import json
import numpy as np,polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'results/predict/universe-size-study'
rows=json.loads((P/'summary.json').read_text());assert len(rows)==8
fig,axs=plt.subplots(2,2,figsize=(14,8),sharex=True)
for col,swaps in enumerate([3,5]):
 for r in rows:
  if r['swaps']!=swaps:continue
  f=pl.read_parquet(P/r['name']/'portfolio_daily.parquet');n=f['nav'].to_numpy();label=f"{r['universe']} Top{r['top']}"
  axs[0,col].plot(f['execution_date'],n,label=label)
  axs[1,col].plot(f['execution_date'],(n/np.maximum.accumulate(np.r_[1,n])[1:]-1)*100)
 axs[0,col].set_title(f'Max {swaps} replacements/day');axs[0,col].legend(fontsize=9);axs[0,col].set_ylabel('Net NAV');axs[1,col].set_ylabel('Drawdown (%)')
 for a in axs[:,col]:a.grid(alpha=.2)
fig.tight_layout();fig.savefig(P/'comparison.png',dpi=150);plt.close(fig)
trs=[]
for r in rows:
 f=r['full'];vals=[('CSI300∪CSI500' if r['universe']=='union' else 'CSI500'),str(r['top']),str(r['swaps']),f"{f['net_return']:.2%}",f"{f['annual_return']:.2%}",f"{f['max_drawdown']:.2%}",f"{f['information_ratio']:.3f}",f"{f['buy_turnover']:.2%}",f"{f['holding_count']:.1f}",f"{f['cash_weight']:.2%}",f"{r['early']['net_return']:.2%}",f"{r['later']['net_return']:.2%}"]
 trs.append('<tr>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
(P/'report.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Top160 / 中证500 Top50、100</title><style>body{font-family:system-ui;max-width:1300px;margin:32px auto;padding:0 22px;color:#172033}img{width:100%}p{line-height:1.7}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:right;padding:9px;border-bottom:1px solid #ddd}</style><h1>交易股票池与持仓数量对比</h1><p>相同105因子LGBM预测，执行期2021-04-02至2026-08-27，最终估值至2026-08-28开盘。全池为CSI300∪CSI500，中证500使用信号日历史成分。保持等额建仓，存量权重随价格漂移。</p><img src="comparison.png"><div class="scroll"><table><tr><th>股票池</th><th>目标持仓</th><th>每日最多换仓</th><th>净累计</th><th>净年化</th><th>最大回撤</th><th>IR</th><th>日均买入换手</th><th>平均持仓</th><th>平均现金</th><th>2021–24累计</th><th>2025起累计</th></tr>'''+''.join(trs)+'''</table></div><p>退出阈值为入选名次的1.2倍，买入2.1bp、卖出7.1bp，初始资金1000万元，100股整手，目标现金2%，每日买卖预算各10%。单票上限3%，最低新仓0.5%。相同换仓只数在不同持仓规模下代表不同换手速度，表中同时列出实际换手。</p><p>这是交易池过滤实验，未重新训练中证500专属模型；H1/H5分数在可交易池内重新标准化后各占50%。所有结果为同段历史探索，分段结果用于稳定性诊断，不构成独立留出期。正式配置未修改。</p></html>''',encoding='utf-8')
