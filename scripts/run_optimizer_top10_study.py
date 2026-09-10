import json
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np,polars as pl
from a_share_data.policy import LimitedReplacementConfig,run_limited_replacement_policy
root=Path('/Users/alanmxy/Documents/Quant_Alpha'); out=root/'results/predict/optimizer-top10-study';cat=root/'A_stock_database/lake/catalog/a_share.duckdb'
base=LimitedReplacementConfig()
variants={'baseline':base,'top10_daily':replace(base,target_fraction=.1,exit_rank=80,max_daily_replacements=80,daily_buy_budget=1,daily_sell_budget=1),'top10_buffer':replace(base,target_fraction=.1),'top10_slow':replace(base,target_fraction=.1,max_daily_replacements=3),'top10_tilt':replace(base,target_fraction=.1,entry_sizing='rank_tilt',rank_tilt=.15)}
rows=[]
for label,run,model in [('98','research-oos-daily60-minute38-v1-recheck-20260909','lgbm_default98recheck'),('105','research-oos-daily60-minute45-v2','lgbm_default105')]:
 for name,conf in variants.items():
  dest=out/(label+'_'+name); pred=root/'results/predict'/run/'predictions'/f'{model}.parquet'
  run_limited_replacement_policy(cat,pred,dest,conf)
  f=pl.read_parquet(dest/'portfolio_daily.parquet')
  def stats(g):
   r=g['net_return'].to_numpy();b=g['csi500_return'].to_numpy();nav=np.cumprod(1+r);v=nav[-1];bn=np.prod(1+b)
   return {'net_return':v-1,'annual_return':v**(252/len(r))-1,'annual_excess':(v/bn)**(252/len(r))-1,'drawdown':float((nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1).min()),'ir':float(np.mean(r-b)/np.std(r-b,ddof=1)*np.sqrt(252)),'buy_turnover':g['buy_turnover'].mean()}
  row={'model':label,'variant':name,'config':asdict(conf),'full':stats(f),'development_2021_2024':stats(f.filter(pl.col('execution_date')<pl.date(2025,1,1))),'later_2025_2026':stats(f.filter(pl.col('execution_date')>=pl.date(2025,1,1)))};rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2));print(label,name,json.dumps(row['full']),flush=True)
