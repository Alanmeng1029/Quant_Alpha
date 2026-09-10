from pathlib import Path
import json
import polars as pl,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[1]/'results/predict/alpha-weight-study'
r=json.loads((p/'summary.json').read_text());names={'equal':'等权＋偏离阈值','rank':'排名倾斜 ±30%','alpha':'截尾 Alpha 倾斜'}
fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
for mode,label in [('equal','Equal weight'),('rank','Rank tilt'),('alpha','Clipped alpha tilt')]:
 f=pl.read_parquet(p/f'105_{mode}/portfolio_daily.parquet');n=f['nav'].to_numpy();axs[0].plot(f['execution_date'],n,label=label);axs[1].plot(f['execution_date'],(n/np.maximum.accumulate(np.r_[1,n])[1:]-1)*100)
axs[0].set_ylabel('Net NAV');axs[1].set_ylabel('Drawdown (%)');axs[0].legend();axs[0].grid(alpha=.2);axs[1].grid(alpha=.2);fig.tight_layout();fig.savefig(p/'comparison.png',dpi=150);plt.close(fig)
trs=[]
for x in r:
 vals=[x['model'],names[x['mode']],f"{x['net_total_return']:.2%}",f"{x['annualized_return']:.2%}",f"{x['max_drawdown']:.2%}",f"{x['information_ratio']:.3f}",f"{x['average_buy_turnover']:.2%}",f"{x['average_cash_weight']:.2%}",f"{x['later_return']:.2%}"]
 trs.append('<tr>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
(p/'report.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Top80 配权对比</title><style>body{font-family:system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#18243b}p{line-height:1.7}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}</style><h1>Top80 内：等权、排名倾斜、Alpha 倾斜</h1><p>当前105因子模型：本轮等权净年化12.19%，排名倾斜11.63%，Alpha倾斜11.43%。倾斜增加换手并扩大回撤。正式配置未切换。</p><img src="comparison.png"><div class="scroll"><table><tr><th>因子数</th><th>权重方式</th><th>净累计</th><th>净年化</th><th>最大回撤</th><th>IR</th><th>日均买入换手</th><th>平均现金</th><th>2025起累计</th></tr>'''+''.join(trs)+'''</table></div><h2>公平比较口径</h2><p>每个模型固定其原Top80有限换仓基线的实际股票名单，三种方法逐日目标代码集合与日期完全一致。名单含因交易约束暂留的股票，并非每天重选当天前80名。持仓股票缺少当日预测时放在最低分，不借用未来分数。最终日期仅用于计价。1311期执行，2021-04-02至2026-08-27，最后估值至2026-08-28开盘。</p><p>目标股票仓位98%；排名倾斜对选中股票按分数排序，权重从等权的1.3倍递减至0.7倍。Alpha使用当日全池H1/H5标准化分数各50%，在持仓截面按5%/95%截尾并标准化，z限制到[-1,1]，乘数为1+0.3z后归一化。三组都只在存量股票权重偏离目标超过NAV的0.25个百分点时调整；入场和退出不受阈值限制。</p><p>复用现有目标权重回测器，初始1000万元、100股整手、买入2.1bp/卖出7.1bp，包含现金约束和复权股数调整。实际现金会因不调仓区间和整手约束偏离2%。这是固定名单的权重对照，未套用原策略的每日10%买卖预算，也不构成涨跌停或市场冲击完全建模的成交保证。</p><p>等权版本管理存量权重，不能与旧“仅新买入等金额”版本混为一谈。两个模型均未发现本组倾斜设置的净收益优势；不代表所有配权参数必然无效。同段历史已用于研究，较晚时期仅为诊断，不能视为独立留出期。</p></html>''',encoding='utf-8')
