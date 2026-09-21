"""Incremental BaoStock qfq snapshot refresh feeding the vendor-preferred qfq views.

BaoStock qfq is anchored at the fetch date, so appending new dates is only
consistent when no ex-dividend event happened since the last fetch.  Each stock
is checked via preclose continuity; on mismatch the stock's full history is
refetched instead of appended.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb
import polars as pl
from .cli import Paths

TZ=ZoneInfo('Asia/Shanghai')
FIELDS='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg,tradestatus,isST'
COLUMNS=['trade_date','baostock_code','open','high','low','close','preclose','volume','amount','adjustflag','turn','pct_chg','trade_status','is_st']
SELECT=['trade_date','ts_code','open','high','low','close','preclose','volume','amount','turn','pct_chg','trade_status','is_st','adjustflag']

def save(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    temp.replace(path)

def completed_day(now=None):
    now=now or datetime.now(TZ)
    return now.date() if now.hour>=18 else now.date()-timedelta(days=1)

def to_frame(ts_code,rows):
    return pl.DataFrame(rows,schema=COLUMNS,orient='row').with_columns(
        pl.lit(ts_code).alias('ts_code'),
        pl.col('trade_date').str.to_date(),
        *[pl.col(c).cast(pl.Float64,strict=False) for c in ('open','high','low','close','preclose','volume','amount','turn','pct_chg')]
    ).select(SELECT)

def query(bs,bs_code,start,end,retries,retry_delay):
    error=''
    for attempt in range(retries):
        result=bs.query_history_k_data_plus(bs_code,FIELDS,start,end,frequency='d',adjustflag='2')
        if result.error_code=='0':return result,''
        error=result.error_msg
        # BaoStock may reset long-lived sockets; reconnect before a bounded retry.
        try:bs.logout()
        finally:time.sleep(retry_delay*(attempt+1))
        relogin=bs.login()
        if relogin.error_code!='0':error=f'relogin failed: {relogin.error_msg}'
    return None,error

def universe_codes(paths: Paths):
    """Full A-share market: every stored file plus the current FTShare universe.

    Stored files keep refreshing (delisted members simply return empty windows);
    newly listed or never-fetched codes bootstrap their full history from --start.
    """
    codes={p.parent.name.split('=')[1] for p in (paths.lake/'canonical'/'baostock_qfq_csi300_csi500_v1').glob('ts_code=*/daily.parquet')}
    if paths.catalog.exists():
        with duckdb.connect(str(paths.catalog),read_only=True) as c:
            if c.execute("SELECT count(*) FROM duckdb_views() WHERE view_name='ftshare_universe'").fetchone()[0]:
                codes.update(r[0] for r in c.execute(
                    'SELECT DISTINCT ts_code FROM ftshare_universe WHERE trade_date=(SELECT max(trade_date) FROM ftshare_universe)').fetchall())
    return sorted(codes)

def run(args):
    root=Path(args.project_root).resolve();paths=Paths(root/'A_stock_database')
    output=paths.lake/'canonical'/'baostock_qfq_csi300_csi500_v1';output.mkdir(parents=True,exist_ok=True)
    status_root=root/'results'/'baostock';status_root.mkdir(parents=True,exist_ok=True)
    lock=(status_root/'sync.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another BaoStock sync is running') from None
    try:
        end=date.fromisoformat(args.end) if args.end else completed_day()
        if end>completed_day():raise ValueError('Refusing a date before the end-of-day publication cutoff (18:00 Shanghai)')
        codes=universe_codes(paths)
        plan=[]
        for ts_code in codes:
            dest=output/f'ts_code={ts_code}'/'daily.parquet'
            last=pl.read_parquet(dest,columns=['trade_date','close'])['trade_date'].max() if dest.exists() else None
            start=(last+timedelta(days=1)) if last else date.fromisoformat(args.start)
            if start<=end:plan.append({'ts_code':ts_code,'start':str(start),'end':str(end)})
        state={'started_at':datetime.now(TZ).isoformat(),'end':str(end),'universe_codes':len(codes),
               'to_fetch':plan,'completed':[],'status':'running'}
        save(status_root/'sync_status.json',state)
        if args.plan:
            state.update(status='planned',finished_at=datetime.now(TZ).isoformat());save(status_root/'sync_status.json',state);return state
        if not plan:
            state.update(status='up_to_date',finished_at=datetime.now(TZ).isoformat());save(status_root/'sync_status.json',state);return state
        import baostock as bs
        login=bs.login()
        if login.error_code!='0':raise RuntimeError(f'BaoStock login failed: {login.error_msg}')
        appended=refetched=empty=0;failures=[]
        try:
            for n,item in enumerate(plan,1):
                ts_code,start_s,end_s=item['ts_code'],item['start'],item['end']
                dest=output/f'ts_code={ts_code}'/'daily.parquet'
                number,exchange=ts_code.split('.');bs_code=f'{exchange.lower()}.{number}'
                result,error=query(bs,bs_code,start_s,end_s,args.retries,args.retry_delay)
                rows=[]
                if result is not None:
                    while result.next():rows.append(result.get_row_data())
                if result is None or not rows:
                    if result is None:failures.append({'ts_code':ts_code,'error':error or 'no rows returned'})
                    else:empty+=1
                else:
                    fresh=to_frame(ts_code,rows).sort('trade_date')
                    entry={'ts_code':ts_code,'start':start_s,'end':end_s,'rows':fresh.height}
                    old=pl.read_parquet(dest) if dest.exists() else None
                    has_old=old is not None and not old.is_empty()
                    # Anchor check: qfq preclose of the first fresh row must match the
                    # stored last close, otherwise an ex-dividend re-anchored the series.
                    consistent=True
                    if has_old:
                        old_close=old.filter(pl.col('close')>0)['close'][-1]
                        preclose=fresh['preclose'][0]
                        consistent=(preclose is not None and old_close is not None
                                    and abs(preclose-old_close)<=1e-4*max(1.0,old_close))
                    if not consistent:
                        full,full_error=query(bs,bs_code,args.start,end_s,args.retries,args.retry_delay)
                        full_rows=[]
                        if full is not None:
                            while full.next():full_rows.append(full.get_row_data())
                        if full is None or not full_rows:
                            failures.append({'ts_code':ts_code,'error':'anchor mismatch and full refetch failed: '+(full_error or 'no rows')})
                            state['completed'].append({**entry,'failed':True});save(status_root/'sync_status.json',state)
                            continue
                        fresh=to_frame(ts_code,full_rows).sort('trade_date')
                        entry.update(anchor_refetch=True,rows=fresh.height);refetched+=1
                        merged=fresh
                    elif has_old:
                        merged=pl.concat([old,fresh]).unique(subset='trade_date',keep='last').sort('trade_date');appended+=1
                    else:
                        merged=fresh;appended+=1
                    tmp=dest.with_suffix('.tmp');dest.parent.mkdir(parents=True,exist_ok=True)
                    merged.write_parquet(tmp,compression='zstd');tmp.replace(dest)
                    state['completed'].append(entry);save(status_root/'sync_status.json',state)
                if n%50==0 or n==len(plan):print(f'{n}/{len(plan)} appended={appended} refetched={refetched} failed={len(failures)}',flush=True)
                if args.request_delay:time.sleep(args.request_delay)
        finally:
            bs.logout()
        state.update(status='complete' if not failures else 'partial',appended=appended,anchor_refetched=refetched,
                     empty_windows=empty,failures=failures,finished_at=datetime.now(TZ).isoformat())
        save(status_root/'sync_status.json',state)
        manifest={'source':'BaoStock','adjustflag':'2 (qfq)','universe':'historical CSI300 union CSI500 plus already-fetched codes',
                  'end':str(end),'universe_codes':len(codes),'appended':appended,'anchor_refetched':refetched,
                  'empty_windows':empty,'failures':failures,'note':'Incremental append with preclose anchor check; anchor mismatches trigger a full-history refetch per stock.'}
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        return state
    finally:lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--start',default='2018-01-01');p.add_argument('--end');p.add_argument('--plan',action='store_true')
    p.add_argument('--retries',type=int,default=3);p.add_argument('--retry-delay',type=float,default=5)
    p.add_argument('--request-delay',type=float,default=0.05)
    args=p.parse_args()
    if args.retries<1 or args.request_delay<0:p.error('retries must be >=1 and request-delay >=0')
    print(json.dumps(run(args),ensure_ascii=False,default=str))

if __name__=='__main__':main()
