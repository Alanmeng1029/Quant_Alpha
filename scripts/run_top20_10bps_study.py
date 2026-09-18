from pathlib import Path
from dataclasses import replace
import json
import polars as pl
from a_share_data.policy import LimitedReplacementConfig,run_limited_replacement_policy
from a_share_data.predict import render_backtest_report
root=Path('/Users/alanmxy/Documents/Quant_Alpha')
source=root/'results/predict/top20-study/csi500_predictions.parquet'
production=json.loads((root/'configs/production_strategy_csi500_top100_v1.json').read_text())
base=LimitedReplacementConfig(**{k:v for k,v in production['portfolio'].items() if k!='strategy'},buy_bps=10,sell_bps=10)
rows=[]
for name,cfg in [('top100',base),('top20',replace(base,target_holdings=20,entry_rank=20,exit_rank=24,max_weight=.15,rebalance_to_weight=.14))]:
 out=root/'results/predict/top20-study-10bps'/name
 result=run_limited_replacement_policy(root/'A_stock_database/lake/catalog/a_share.duckdb',source,out,cfg)
 metrics=render_backtest_report(out/'portfolio_daily.parquet',out/'report',name+' - buy/sell 10 bps')
 f=pl.read_parquet(out/'portfolio_daily.parquet')
 print(name,json.dumps(metrics),json.dumps(result['charged']),flush=True)
 rows.append({'name':name,**result['charged']})
pl.DataFrame(rows).write_csv(root/'results/predict/top20-study-10bps/summary.csv')
