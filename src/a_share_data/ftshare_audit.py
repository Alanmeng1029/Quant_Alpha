"""Verify installed FTShare dates, stock coverage and minute primary keys."""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path
import duckdb
from .cli import Paths


def audit(root: Path, expected_dates: list[str]) -> dict:
    paths=Paths(root/'A_stock_database')
    with duckdb.connect(str(paths.catalog),read_only=True) as c:
        if not c.execute("SELECT count(*) FROM duckdb_views() WHERE view_name='ftshare_minute_bars'").fetchone()[0]:
            return {'status':'missing','missing_dates':expected_dates}
        rows=c.execute('''SELECT trade_date,count(*) AS rows,count(distinct ts_code) AS stocks,
                   count(*)-count(distinct(ts_code,datetime)) AS duplicates,
                   count(*) FILTER(WHERE abs(volume_lot*100-volume_share)>0.000001) AS unit_errors
                   FROM ftshare_minute_bars GROUP BY trade_date ORDER BY trade_date''').fetchall()
        daily=dict(c.execute('SELECT trade_date,count(*) FROM ftshare_daily_bars GROUP BY trade_date').fetchall())
        empties=c.execute('''SELECT u.trade_date,u.ts_code,u.name,
                    EXISTS(SELECT 1 FROM ftshare_suspensions s WHERE s.trade_date=u.trade_date AND s.ts_code=u.ts_code)
                    FROM ftshare_universe u ANTI JOIN ftshare_daily_bars d USING(trade_date,ts_code)
                    ORDER BY u.trade_date,u.ts_code''').fetchall()
        bad_counts=c.execute('''SELECT ts_code,trade_date,count(*) FROM ftshare_minute_bars
                    GROUP BY ts_code,trade_date HAVING count(*)<>241''').fetchall()
    listing_path=root/'results/ftshare/listing_dates.json'
    listing=json.loads(listing_path.read_text()) if listing_path.exists() else {}
    unexplained=[];not_listed=[];suspended=[]
    for day,symbol,name,is_suspended in empties:
        record={'date':str(day),'symbol':symbol,'name':name}
        listing_date=listing.get(symbol,{}).get('listing_date')
        if is_suspended:suspended.append(record)
        elif listing_date and date.fromisoformat(listing_date)>day:
            not_listed.append({**record,'listing_date':listing_date})
        else:unexplained.append(record)
    installed={str(r[0]) for r in rows}
    missing=sorted(set(expected_dates)-installed)
    stats=[dict(date=str(d),minute_rows=n,stocks=stocks,daily_rows=daily.get(d,0),duplicates=dup,volume_unit_errors=units) for d,n,stocks,dup,units in rows]
    issues=bool(missing or unexplained or bad_counts or any(r['duplicates'] or r['volume_unit_errors'] or r['daily_rows']!=r['stocks'] for r in stats))
    result={'status':'failed' if issues else 'passed','expected_dates':expected_dates,'missing_dates':missing,
            'dates':stats,'minute_rows':sum(r['minute_rows'] for r in stats),'daily_rows':sum(r['daily_rows'] for r in stats),
            'suspended_stock_days':len(suspended),'not_yet_listed_stock_days':len(not_listed),
            'unexplained_empty_stock_days':unexplained,'non_241_stock_days':bad_counts,
            'not_yet_listed':not_listed,'suspended':suspended}
    dest=root/'results/ftshare/coverage_audit.json'
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(dest)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2]));args=p.parse_args()
    root=Path(args.project_root);state=json.loads((root/'results/ftshare/sync_status.json').read_text())
    result=audit(root,state['expected_trade_dates'])
    print(json.dumps({k:v for k,v in result.items() if k not in ['suspended','not_yet_listed']},ensure_ascii=False,default=str))
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
