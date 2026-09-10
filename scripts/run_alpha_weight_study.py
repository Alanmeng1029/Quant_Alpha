"""Fixed membership comparison of equal, rank and clipped-score allocations."""
from pathlib import Path
import json
import numpy as np
import polars as pl
from a_share_data.policy import _score_frame
from a_share_data.predict import backtest_targets
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/predict/alpha-weight-study'

def weights(scores,mode):
    n=len(scores)
    if mode=='equal': raw=np.ones(n)
    elif mode=='rank': raw=1+.3*np.linspace(1,-1,n)
    else:
        a=np.clip(scores,*np.quantile(scores,[.05,.95]));sd=a.std()
        z=(a-a.mean())/sd if sd>1e-12 else np.zeros(n)
        raw=1+.3*np.clip(z,-1,1)
    return .98*raw/raw.sum()

def main():
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    for label,run,model in [('98','research-oos-daily60-minute38-v1-recheck-20260909','lgbm_default98recheck'),('105','research-oos-daily60-minute45-v2','lgbm_default105')]:
        root=ROOT/'results/predict'/run
        p=pl.read_parquet(root/f'predictions/{model}.parquet')
        h=pl.read_parquet(root/f'backtests/{model}/holdings.parquet').select(pl.col('signal_date').alias('trade_date'),'ts_code')
        frames=p.partition_by('trade_date',maintain_order=True); members={k[0]:set(g['ts_code']) for k,g in h.partition_by('trade_date',as_dict=True).items()}
        rows={m:[] for m in ['equal','rank','alpha']}
        for frame in frames:
            day=frame['trade_date'][0]
            if day not in members: continue
            scored=_score_frame(frame,.5)
            ranked=pl.DataFrame({'ts_code':sorted(members[day])}).join(scored.select('ts_code','score'),on='ts_code',how='left').with_columns(pl.col('score').fill_null(float(scored['score'].min())-1e-9)).sort(['score','ts_code'],descending=[True,False]).with_columns(pl.lit(frame['execution_date'][0]).alias('execution_date'))
            assert ranked.height==len(members[day])
            for mode in rows:
                w=weights(ranked['score'].to_numpy(),mode)
                for code,weight in zip(ranked['ts_code'],w):rows[mode].append({'trade_date':day,'execution_date':ranked['execution_date'][0],'ts_code':code,'target_weight':float(weight)})
        for mode,records in rows.items():
            target=pl.DataFrame(records);last=target['execution_date'].max()
            terminal=p['execution_date'].max()
            assert terminal>last
            target=pl.concat([target,target.filter(pl.col('execution_date')==last).with_columns(pl.lit(terminal).alias('execution_date'))])
            dest=OUT/f'{label}_{mode}';dest.mkdir(exist_ok=True);path=dest/'targets.parquet';target.write_parquet(path)
            assert target.group_by('execution_date').agg(pl.col('target_weight').sum()).select((pl.col('target_weight')-.98).abs().max()).item()<1e-10
            result=backtest_targets(ROOT/'A_stock_database/lake/catalog/a_share.duckdb',path,dest,rebalance_band=.0025)
            f=pl.read_parquet(dest/'portfolio_daily.parquet')
            result['later_return']=float((1+f.filter(pl.col('execution_date')>=pl.date(2025,1,1))['net_return']).product()-1)
            result['early_return']=float((1+f.filter(pl.col('execution_date')<pl.date(2025,1,1))['net_return']).product()-1)
            result['model']=label;result['mode']=mode;results.append(result)
            (OUT/'summary.json').write_text(json.dumps(results,indent=2));print(label,mode,json.dumps(result),flush=True)
    (OUT/'protocol.json').write_text(json.dumps({'membership':'frozen original Top80 limited-replacement baseline holdings per model','cash_target':.02,'rank_tilt':.3,'alpha_clip_quantiles':[.05,.95],'alpha_z_clip':[-1,1],'rebalance_band':.0025,'band_units':'fraction of portfolio NAV','terminal':'valuation only','selection_warning':'same historical sample, exploratory comparison'},indent=2))
if __name__=='__main__':main()
