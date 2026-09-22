"""Serial matched-sample H1/H5 portfolio comparison after residual LSTM training.

Use the historical baseline's actual sleeve universes: CSI500 core (80%, 1%
name cap), union concentrated sleeve (20%, 5% cap), costs 2 bps each side.
"""
from pathlib import Path
import argparse
import base64
import html
import json
import os
import subprocess
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/quant-alpha-mpl')
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(config_path: Path) -> None:
    cfg=json.loads(config_path.read_text());root=Path(cfg['output'])/'full'
    report=json.loads((root/'report.json').read_text())
    tag=str(report['factor_count']);assert tag=='125'
    output=root/'multiperiod_comparison';output.mkdir(parents=True,exist_ok=True)
    core_reference=pl.read_parquet('results/predict/dos-minute20-model-comparison-v1/daily60_minute45_dos20/csi500_predictions.parquet').select('trade_date','ts_code')
    frames={name:pl.read_parquet(root/f'predictions/{name}.parquet') for name in ['lstm125','lgbm_default125_common']}
    keys=['trade_date','ts_code','execution_date']
    assert frames['lstm125'].select(keys).sort(keys).equals(frames['lgbm_default125_common'].select(keys).sort(keys))
    env={**os.environ,'PYTHONPATH':'src','MPLCONFIGDIR':'/tmp/quant-alpha-mpl'}
    def command(args):
        print('RUN',' '.join(map(str,args)),flush=True)
        subprocess.run([sys.executable,*map(str,args)],check=True,env=env)
    for name,frame in frames.items():
        folder=output/name;folder.mkdir(exist_ok=True)
        source=root/f'predictions/{name}.parquet'
        core=folder/'csi500_predictions.parquet'
        frame.join(core_reference,on=['trade_date','ts_code'],how='semi').write_parquet(core)
        for sleeve,cap,predictions in [('max1',.01,core),('max5',.05,source)]:
            if not (folder/sleeve/'optimizer_summary.json').exists():
                command(['scripts/optimize_multiperiod_mu_turnover.py','--predictions',predictions,'--output',folder/sleeve,'--max-weight',cap])
        blend=folder/'blend'
        if not (blend/'optimizer_summary.json').exists():
            command(['scripts/blend_target_weights.py','--target-weights',folder/'max1/target_weights.parquet','--allocation',.8,'--target-weights',folder/'max5/target_weights.parquet','--allocation',.2,'--output',blend])
        weights=pl.read_parquet(blend/'target_weights.parquet')
        assert weights['target_weight'].max()<=.018+1e-10
        assert weights.group_by('trade_date').agg(pl.col('target_weight').sum()).select((pl.col('target_weight')-.98).abs().max()).item()<1e-9
        bt=blend/'backtest'
        if not (bt/'portfolio_summary.json').exists():
            command(['-m','a_share_data.predict','backtest-portfolio','--catalog',cfg['catalog'],'--target-weights',blend/'target_weights.parquet','--output',bt])
        if not (bt/'report/report_summary.json').exists():
            command(['-m','a_share_data.predict','render-backtest-report','--portfolio-daily',bt/'portfolio_daily.parquet','--target-weights',blend/'target_weights.parquet','--output',bt/'report','--title',f'{name}: matched-sample 80/20 blend'])
    # Every optimization and backtest above is intentionally sequential.
    daily={name:pl.read_parquet(output/name/'blend/backtest/portfolio_daily.parquet').sort('execution_date') for name in frames}
    summaries={name:json.loads((output/name/'blend/backtest/report/report_summary.json').read_text()) for name in frames}
    assert daily['lstm125']['execution_date'].equals(daily['lgbm_default125_common']['execution_date'])
    metrics=['gross_total_return','net_total_return','net_annualized_return','net_sharpe','max_drawdown','excess_annual_return_243','excess_sharpe_243','excess_max_drawdown','average_buy_turnover','average_fee_bps']
    comparison=pl.DataFrame([{'metric':k,**{name:s[k] for name,s in summaries.items()}} for k in metrics]);comparison.write_csv(output/'portfolio_comparison.csv')
    periods=[]
    for name,df in daily.items():
        for freq in ['year','quarter']:
            expr=pl.col('execution_date').dt.year().cast(pl.String)
            if freq=='quarter':expr=expr+pl.lit('Q')+pl.col('execution_date').dt.quarter().cast(pl.String)
            for (period,),g in df.with_columns(expr.alias('period')).partition_by('period',as_dict=True).items():
                net=float(np.prod(1+g['net_return'].to_numpy())-1);bench=float(np.prod(1+g['benchmark_return'].to_numpy())-1)
                periods.append(dict(model=name,frequency=freq,period=period,start=str(g['execution_date'].min()),end=str(g['execution_date'].max()),days=g.height,net_return=net,benchmark_return=bench,excess_difference=net-bench,buy_turnover=g['buy_turnover'].mean()))
    pf=pl.DataFrame(periods);pf.write_csv(output/'period_performance.csv')
    rows=[]
    for name,data in report['models'].items():
        for h in ['h1','h5']:
            rows.append(dict(model=name,horizon=h,**{k:data[h][k] for k in ['mean_rank_ic','mean_pearson_ic','positive_ic_ratio','days']}))
    ictable=pl.DataFrame(rows);ictable.write_csv(output/'ic_comparison.csv')
    quarterly=[]
    for name,data in report['models'].items():
        for q,hs in data['period_metrics']['quarter'].items():
            for h in ['h1','h5']:quarterly.append(dict(model=name,quarter=q,horizon=h,rank_ic=hs[h]['mean_rank_ic']))
    pl.DataFrame(quarterly).write_csv(output/'quarterly_ic.csv')
    windows=[]
    for file in sorted((root/'lstm/quarters').glob('*/training.json')):
        data=json.loads(file.read_text());assert data['architecture']['channels']==250 and data['device']=='mps' and not data['cpu_fallback']
        windows.append(dict(quarter=file.parent.name,best_epoch=data['best_epoch'],epochs_explored=len(data['selection_history']),fit_samples=data['fit_samples'],full_training_samples=data['full_training_samples'],test_samples=data['test_samples'],validation_loss=data['best_validation_loss']))
    assert len(windows)==22
    pl.DataFrame(windows).write_csv(output/'training_summary.csv')
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for name,df in daily.items():
        dates=df['execution_date'].to_list();axes[0].plot(dates,df['nav'],label=name)
        axes[1].plot(dates,np.cumprod(1+df['net_return'].to_numpy()-df['benchmark_return'].to_numpy()),label=name)
        axes[2].plot(dates,df['buy_turnover'].rolling_mean(20),label=name)
    for ax,title in zip(axes,['Net NAV / matched sample','Compounded daily excess vs CSI500','20-day average buy turnover']):ax.set_title(title);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(output/'portfolio_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    yearly=pl.read_csv(root/'reports/yearly_metrics.csv')
    for ax,h in zip(axes,['h1','h5']):
        for name in frames:
            d=yearly.filter((pl.col('model')==name)&(pl.col('horizon')==h)).sort('year');ax.plot(d['year'],d['mean_rank_ic'],marker='o',label=name)
        ax.set_title(h.upper()+' yearly Rank IC');ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(output/'yearly_ic.png',dpi=160);plt.close(fig)
    def table(df,percent=()):
        cells=[]
        for r in df.to_dicts():
            row=[]
            for k,v in r.items():
                value=f'{v:.2%}' if k in percent and isinstance(v,(int,float)) else (f'{v:.5f}' if isinstance(v,float) else str(v))
                row.append('<td>'+html.escape(value)+'</td>')
            cells.append('<tr>'+''.join(row)+'</tr>')
        return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in df.columns)+'</tr>'+''.join(cells)+'</table></div>'
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((output/name).read_bytes()).decode()+'">'
    formatted=[]
    for r in comparison.to_dicts():
        percent=any(k in r['metric'] for k in ['return','drawdown','turnover'])
        formatted.append({'metric':r['metric'],**{k:(f'{v:.2%}' if percent else f'{v:.4f}') for k,v in r.items() if k!='metric'}})
    a=summaries['lstm125'];b=summaries['lgbm_default125_common']
    verdict=f"共同样本净年化：LSTM {a['net_annualized_return']:.2%}，LGBM {b['net_annualized_return']:.2%}；超额Sharpe：LSTM {a['excess_sharpe_243']:.3f}，LGBM {b['excess_sharpe_243']:.3f}。本次为研究候选，未替换正式模型。"
    note='125个原始价格因子，20日序列＋缺失指示（250通道）；128/64残差LSTM；MSE训练与早停，学习率3e-5，最多30轮、patience=4；756日训练窗口，外层标签间隔11日对齐当前LGBM，内层687日拟合＋6日隔离＋63日验证。每季重新初始化并按最佳轮数全窗口重训。IC为沪深300＋中证500联合池共同样本；组合沿用实际旧基线：80%中证500核心（1%上限）＋20%联合池集中组合（5%上限），买卖各2bp。相较旧105因子LSTM，原始价格因子口径及训练截止日也发生变化，不能把差异全部归因于新增20因子。'
    content='<h1>125因子残差LSTM：滚动样本外对照</h1><p><strong>'+verdict+'</strong></p><p>'+note+'</p><h2>IC对照</h2>'+table(ictable)+image('yearly_ic.png')+'<h2>成本后组合表现</h2>'+table(pl.DataFrame(formatted))+image('portfolio_comparison.png')
    for freq,title in [('year','分年表现'),('quarter','分季度表现')]:content+='<h2>'+title+'</h2>'+table(pf.filter(pl.col('frequency')==freq).sort('period','model'),['net_return','benchmark_return','excess_difference','buy_turnover'])
    content+='<h2>训练记录</h2>'+table(pl.DataFrame(windows))
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Residual LSTM 125</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font-family:Arial,sans-serif;line-height:1.6;color:#18212f}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #d9e1ea;padding:8px;text-align:right;white-space:nowrap}th{background:#eef2f6}.scroll{overflow:auto}img{width:100%;max-width:1200px}h2{margin-top:30px}</style>'+content+'</html>'
    (output/'report_offline.html').write_text(page)
    lines=['# 125因子残差LSTM滚动重训',verdict,'',note,'','| Model | H1 Rank IC | H5 Rank IC |','|---|---:|---:|']
    for name,r in report['models'].items():lines.append(f"| {name} | {r['h1']['mean_rank_ic']:.6f} | {r['h5']['mean_rank_ic']:.6f} |")
    lines.extend(['','![Portfolio comparison](portfolio_comparison.png)','','[Offline report](report_offline.html)'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(ictable);print(comparison);print('REPORT',output/'report_offline.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/prediction_sequence_lstm_residual_raw125_v1.json'))
    run(parser.parse_args().config)
