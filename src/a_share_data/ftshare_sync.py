"""Resumable FTShare end-of-day synchronization, using the local exchange calendar."""
from __future__ import annotations
import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb
from .cli import Paths, sha256_file
from .ftshare import ingest, register_views

TZ=ZoneInfo('Asia/Shanghai')
BASE='https://market.ft.tech/gateway'

def save(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    temp.replace(path)

def completed_day(now=None):
    now=now or datetime.now(TZ)
    return now.date() if now.hour>=18 else now.date()-timedelta(days=1)

def trade_days(calendar: Path, start: date, end: date):
    with calendar.open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['交易所']=='SSE']
    by_date={date.fromisoformat(r['日期']):r['是否交易']=='交易' for r in rows}
    days=[]
    for i in range((end-start).days+1):
        day=start+timedelta(days=i)
        if day not in by_date:raise ValueError(f'Exchange calendar does not cover {day}; refresh calendar before downloading')
        if by_date[day]:days.append(day)
    return days

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class Client:
    def __init__(self,key,rate=10):
        self.key=key;self.rate=rate;self.lock=threading.Lock();self.next_request=0.;self.requests=0
    def get(self,path,params):
        url=BASE+path+'?'+urllib.parse.urlencode(params,doseq=True)
        for attempt in range(5):
            with self.lock:
                delay=max(0,self.next_request-time.monotonic())
                self.next_request=max(time.monotonic(),self.next_request)+1/self.rate
                self.requests+=1
            if delay:time.sleep(delay)
            try:
                req=urllib.request.Request(url,headers={'FTSHARE_API_KEY':self.key,'Content-Type':'application/json'})
                with urllib.request.build_opener(NoRedirect).open(req,timeout=40) as r:data=json.load(r)
                if data.get('code')!=200:raise RuntimeError(str(data).replace(self.key,'[REDACTED]')[:400])
                return data
            except urllib.error.HTTPError as e:
                message=e.read().decode().replace(self.key,'[REDACTED]')[:400]
                if e.code not in (429,500,502,503,504) or attempt==4:
                    raise RuntimeError(f'HTTP {e.code}: {message}') from None
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                if attempt==4:raise
            time.sleep(min(30,2**attempt))

    def pages(self,path,params,page_size):
        first=self.get(path,{**params,'page':1,'page_size':page_size})
        payload=first['data'];rows=list(payload['records'])
        for page in range(2,payload['pages']+1):
            rows.extend(self.get(path,{**params,'page':page,'page_size':page_size})['data']['records'])
        if len(rows)!=payload['total']:raise ValueError('Pagination count mismatch')
        return rows

def ashare(symbol):
    return (symbol.endswith('.SH') and symbol.startswith('6')) or (symbol.endswith('.SZ') and symbol.startswith(('00','30')))

def download_day(client,day,universe,out,workers):
    out.mkdir(parents=True,exist_ok=True)
    # Persist the exact attempted universe, so an interrupted date never changes
    # membership underneath batch caches when tomorrow's stock list changes.
    if (out/'universe.json').exists():universe=json.loads((out/'universe.json').read_text())
    else:save(out/'universe.json',universe)
    symbols=sorted(r['stock_code'] for r in universe['stocks'])
    start=int(datetime.combine(day,datetime.min.time(),TZ).timestamp()*1000)
    end=start+86400000-1
    report={'date':str(day),'timezone':'Asia/Shanghai','adjustment':'none','universe_count':len(symbols),'source':BASE,'datasets':{}}
    for kind,path in [('daily','/api/v2/market/data/stock-candlesticks/batch'),('minute','/api/v2/market/data/stock_minutes/batch')]:
        def fetch(batch):
            # One daily request covers the pending range (capped at a month).
            # Minute requests cover <=3 calendar days, <=723 rows per stock.
            anchor=client.range_start
            if kind=='daily':
                range_start=max(anchor,day.replace(day=1))
                next_month=(day.replace(day=28)+timedelta(days=4)).replace(day=1)
                range_end=min(client.range_end,next_month-timedelta(days=1))
            else:
                range_start=anchor+timedelta(days=((day-anchor).days//3)*3)
                range_end=min(client.range_end,range_start+timedelta(days=2))
            query_start=int(datetime.combine(range_start,datetime.min.time(),TZ).timestamp()*1000)
            query_end=int(datetime.combine(range_end+timedelta(days=1),datetime.min.time(),TZ).timestamp()*1000)-1
            params=dict(symbols=batch,since_ts_millis=query_start,until_ts_millis=query_end,limit=1000)
            params.update({'interval_unit':'Day'} if kind=='daily' else {'interval_value':1})
            digest=hashlib.sha256(json.dumps([path,params],sort_keys=True).encode()).hexdigest()[:24]
            cache=out.parent/'range_cache'/kind/(digest+'.json.gz')
            if cache.exists():
                with gzip.open(cache,'rt') as f:d=json.load(f)
            else:
                d=client.get(path,params)
                cache.parent.mkdir(parents=True,exist_ok=True)
                temp=cache.with_suffix('.tmp')
                with gzip.open(temp,'wt') as f:json.dump(d,f,ensure_ascii=False)
                temp.replace(cache)
            data=d['data']
            groups=dict(data) if kind=='daily' else {r['symbol']:r['items'] for r in data}
            if len(groups)!=len(data) or set(groups)-set(batch):raise ValueError('Unexpected/duplicate batch symbols')
            if any(len(rows)>=1000 for rows in groups.values()):raise ValueError('Potentially truncated range response')
            groups={sym:[r for r in rows if start<=r['ts_millis']<=end] for sym,rows in groups.items()}
            return batch,groups
        counts={};batches=[symbols[i:i+10] for i in range(0,len(symbols),10)]
        tmp=out/(kind+'.jsonl.gz.tmp')
        with gzip.open(tmp,'wt') as f,ThreadPoolExecutor(max_workers=workers) as pool:
            for n,(batch,groups) in enumerate(pool.map(fetch,batches),1):
                for symbol in batch:
                    rows=groups.get(symbol,[]);counts[symbol]=len(rows)
                    if kind=='daily' and len(rows)>1:raise ValueError('More than one daily bar')
                    for row in rows:f.write(json.dumps({'symbol':symbol,**row},ensure_ascii=False,separators=(',',':'))+'\n')
                if n%100==0 or n==len(batches):print(f'{day} {kind} {min(n*10,len(symbols))}/{len(symbols)}',flush=True)
        tmp.replace(out/(kind+'.jsonl.gz'))
        report['datasets'][kind]={'errors':[],'symbols_attempted':len(counts),'rows':sum(counts.values()),'symbols_with_data':sum(v>0 for v in counts.values()),'counts_by_symbol':counts,'empty_symbols':[s for s,n in counts.items() if not n],'row_count_distribution':dict(Counter(counts.values()))}
    daily=report['datasets']['daily']['counts_by_symbol'];minute=report['datasets']['minute']['counts_by_symbol']
    discrepancies=[s for s in symbols if bool(daily[s])!=bool(minute[s]) or (daily[s] and minute[s]!=241)]
    if discrepancies:
        # Incomplete successful responses must not become permanent cache hits.
        for kind in ['daily','minute']:
            for p in (out.parent/'range_cache'/kind).glob('*.json.gz'):p.unlink()
        save(out/'incomplete.json',{'symbols':discrepancies,'retry_required':True})
        raise ValueError(f'{day}: {len(discrepancies)} stocks have incomplete daily/minute coverage; not ingested')
    if sum(bool(n) for n in daily.values())<len(symbols)*.95:
        raise ValueError('Less than 95% of requested universe has daily data; not ingested')
    susp=client.pages('/api/v1/market/data/suspension-list',{'trade_date':day.strftime('%Y%m%d')},200)
    save(out/'suspensions.json',{'code':200,'data':{'records':susp,'total':len(susp),'pages':1}})
    susp_symbols={r['symbol'] for r in susp}
    listing=getattr(client,'listing_dates',{})
    if listing:save(out/'listing_dates.json',{s:listing.get(s,{}) for s in symbols})
    not_listed=[s for s in symbols if not daily[s] and listing.get(s,{}).get('listing_date') and listing[s]['listing_date']>str(day)]
    unexplained=[s for s in symbols if not daily[s] and s not in susp_symbols and s not in not_listed]
    if unexplained and listing:
        save(out/'incomplete.json',{'unexplained_empty_symbols':unexplained,'retry_required':True})
        # Empty-but-successful responses may be due to delayed publication.
        for kind in ['daily','minute']:
            for p in (out.parent/'range_cache'/kind).glob('*.json.gz'):p.unlink()
        raise ValueError(f'{day}: unexplained empty symbols: {unexplained}; not ingested')
    save(out/'validation.json',{'coverage_passed':True,'empty_symbols':report['datasets']['daily']['empty_symbols'],
         'empty_not_in_suspensions':[s for s in symbols if not daily[s] and s not in susp_symbols],
         'not_yet_listed_symbols':not_listed,'unexplained_empty_symbols':unexplained,
         'note':'Current plus previously observed universe, not a point-in-time constituent reconstruction. Empty pairs may include not-yet-listed stocks. Full schema, OHLC, timestamp and duplicate checks run in ingest-ftshare.'})
    report['completed_at']=datetime.now(TZ).isoformat();save(out/'summary.json',report)
    return report

def run(args):
    root=Path(args.project_root).resolve();paths=Paths(root/'A_stock_database')
    output=root/'results'/'ftshare';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'sync.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another FTShare sync is running') from None
    try:
        with duckdb.connect(str(paths.catalog),read_only=True) as c:
            baseline=c.execute('SELECT max(trade_date) FROM observed_calendar').fetchone()[0]
            historical=c.execute('SELECT ts_code,name FROM instruments WHERE last_observed_date >= ?', [baseline]).fetchall()
        start=date.fromisoformat(args.start) if args.start else baseline+timedelta(days=1)
        end=date.fromisoformat(args.end) if args.end else completed_day()
        if end>completed_day():raise ValueError('Refusing a date before the end-of-day publication cutoff (18:00 Shanghai)')
        days=trade_days(paths.root/'交易日历.csv',start,end)
        vendor_root=paths.lake/'canonical'/'ftshare'
        existing={date.fromisoformat(p.parent.name.split('=')[1]) for p in vendor_root.glob('trade_date=*/manifest.json')}
        pending=[d for d in days if d not in existing]
        state={'started_at':datetime.now(TZ).isoformat(),'start':str(start),'end':str(end),'baseline_end':str(baseline),'expected_trade_dates':[str(d) for d in days],'pending_dates':[str(d) for d in pending],'completed':[],'status':'running','calendar_sha256':sha256_file(paths.root/'交易日历.csv')}
        save(output/'sync_status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
        if args.plan:
            state['status']='planned';save(output/'sync_status.json',state);return state
        if not pending:
            with duckdb.connect(str(paths.catalog)) as c:register_views(c,paths)
            state.update(status='up_to_date',finished_at=datetime.now(TZ).isoformat());save(output/'sync_status.json',state);return state
        credentials=json.loads((root/'api_credentials.local.json').read_text())
        client=Client(credentials['api_key'],args.rate)
        client.range_start=start;client.range_end=end
        descriptions=client.pages('/api/v1/market/data/stock-description',{},200)
        client.listing_dates={r['symbol']:{'listing_date':r.get('listing_date'),'name':r.get('name'),'status':r.get('status')} for r in descriptions}
        save(output/'listing_dates.json',client.listing_dates)
        save(output/'listing_dates.meta.json',{'retrieved_at':datetime.now(TZ).isoformat(),'source':BASE+'/api/v1/market/data/stock-description'})
        listed=client.pages('/api/v1/market/data/stock-list',{},500)
        mapping={r['stock_code']:r['stock_name'] for r in listed if ashare(r['stock_code'])}
        for symbol,name in historical:
            if ashare(symbol):mapping.setdefault(symbol,name or symbol)
        universe={'retrieved_at':datetime.now(TZ).isoformat(),'selection':'current FTShare A shares union last legacy observed universe; no historical membership assertion',
                  'stocks':[{'stock_code':s,'stock_name':mapping[s]} for s in sorted(mapping)],'selected_total':len(mapping)}
        for day in pending:
            try:
                out=output/str(day)
                # A fully downloaded date can be revalidated and imported without network.
                if not (out/'summary.json').exists():download_day(client,day,universe,out,args.workers)
                installed=ingest(out,paths)
                state['completed'].append({'date':str(day),'status':installed['status']})
                state['pending_dates'].remove(str(day));save(output/'sync_status.json',state)
                print('INGESTED',day,flush=True)
            except Exception as e:
                state.update(status='failed',failed_date=str(day),error=str(e).replace(client.key,'[REDACTED]'),requests=client.requests)
                save(output/'sync_status.json',state);raise
        from .ftshare_audit import audit
        checked=audit(root,state['expected_trade_dates'])
        if checked['status']!='passed':
            state.update(status='failed_audit',audit_path='results/ftshare/coverage_audit.json')
            save(output/'sync_status.json',state)
            raise ValueError('Installed data coverage audit failed; see coverage_audit.json')
        state.update(status='complete',finished_at=datetime.now(TZ).isoformat(),requests=client.requests)
        save(output/'sync_status.json',state)
        return state
    finally:lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--start');p.add_argument('--end');p.add_argument('--plan',action='store_true')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--rate',type=float,default=10)
    args=p.parse_args()
    if not 1<=args.workers<=8 or not 0<args.rate<=20:p.error('workers must be 1..8 and rate 0..20')
    print(json.dumps(run(args),ensure_ascii=False,default=str))

if __name__=='__main__':main()
