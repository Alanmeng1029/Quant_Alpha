"""Evaluate the current 105-factor predictions under historical trading universes."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib
import duckdb,numpy as np,polars as pl
from a_share_data.policy import LimitedReplacementConfig,run_limited_replacement_policy
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/predict/universe-size-study'
CAT=ROOT/'A_stock_database/lake/catalog/a_share.duckdb'
SOURCE=ROOT/'results/predict/research-oos-daily60-minute45-v2/predictions/lgbm_default105.parquet'

def stats(f):
 r=f['net_return'].to_numpy();b=f['csi500_return'].to_numpy();n=np.cumprod(1+r);bench=np.prod(1+b)
 return {'days':len(r),'net_return':float(n[-1]-1),'annual_return':float(n[-1]**(252/len(r))-1),'annual_excess':float((n[-1]/bench)**(252/len(r))-1),'max_drawdown':float((n/np.maximum.accumulate(np.r_[1,n])[1:]-1).min()),'information_ratio':float((r-b).mean()/(r-b).std(ddof=1)*np.sqrt(252)),'buy_turnover':float(f['buy_turnover'].mean()),'cash_weight':float(f['cash_weight'].mean()),'holding_count':float(f['holding_count'].mean())}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 p=pl.read_parquet(SOURCE)
 c=duckdb.connect(str(CAT),read_only=True)
 u=pl.from_arrow(c.execute("SELECT DISTINCT trade_date,ts_code FROM index_trading_universe WHERE index_code='000905.SH'").arrow());c.close()
 filtered=p.join(u,on=['trade_date','ts_code'],how='semi').sort(['trade_date','ts_code'])
 assert filtered['trade_date'].unique().sort().equals(p['trade_date'].unique().sort())
 counts=filtered.group_by('trade_date').len();assert counts['len'].min()>=100
 filtered.write_parquet(OUT/'csi500_predictions.parquet')
 protocol={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'model':'105-factor LGBM, unchanged training universe CSI300 union CSI500','trading_universe':'CSI500 filtered at signal date using historical index_trading_universe','csi500_min_names':int(counts['len'].min()),'csi500_max_names':int(counts['len'].max()),'sizing':'equal entry notional; continuing weights drift; no alpha tilt','ranking':'H1/H5 50/50 z-scores recomputed inside each trading universe','exit_buffer':1.2,'replacements':[3,5],'cost_bps':[2.1,7.1]}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 rows=[]
 for uni,n in [('union',80),('union',160),('csi500',50),('csi500',100)]:
  for swaps in [3,5]:
   conf=LimitedReplacementConfig(target_holdings=n,entry_rank=n,exit_rank=int(n*1.2),max_daily_replacements=swaps)
   name=f'{uni}_top{n}_swap{swaps}';dest=OUT/name
   result=run_limited_replacement_policy(CAT,SOURCE if uni=='union' else OUT/'csi500_predictions.parquet',dest,conf)
   f=pl.read_parquet(dest/'portfolio_daily.parquet')
   row={'name':name,'universe':uni,'top':n,'swaps':swaps,'config':asdict(conf),'full':stats(f),'early':stats(f.filter(pl.col('execution_date')<pl.date(2025,1,1))),'later':stats(f.filter(pl.col('execution_date')>=pl.date(2025,1,1))),'zero_cost_return':result['zero_cost']['net_total_return']}
   rows.append(row);(OUT/'summary.json').write_text(json.dumps(rows,indent=2));print(name,json.dumps(row['full']),flush=True)
if __name__=='__main__':main()
