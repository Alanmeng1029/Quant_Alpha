"""Auditable FTShare snapshot ingestion; preserve the historical research baseline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date
from pathlib import Path

import duckdb

from .cli import Paths, connect, sha256_file, sql_path, utc_now


def register_views(conn: duckdb.DuckDBPyConnection, paths: Paths) -> None:
    root = paths.lake / 'canonical' / 'ftshare'
    if not list(root.glob('trade_date=*/minute.parquet')):
        return
    for view, filename in [('ftshare_minute_bars', 'minute'), ('ftshare_daily_bars', 'daily'),
                           ('ftshare_daily_aggregated', 'daily_aggregated'),
                           ('ftshare_universe', 'universe'), ('ftshare_suspensions', 'suspensions')]:
        conn.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{sql_path(root / 'trade_date=*' / (filename+'.parquet'))}', hive_partitioning=false)")
    # Historical source wins on overlapping keys. Explicit projection keeps
    # vendor-only fields separate and retains fractional lots without rounding.
    conn.execute('''CREATE OR REPLACE VIEW market_minute_bars AS
        SELECT ts_code, datetime, trade_date, minute_index, open, high, low, close,
               volume_lot::DOUBLE AS volume_lot, volume_share, amount_cny,
               'legacy_minute' AS source FROM minute_bars
        UNION ALL
        SELECT f.ts_code, f.datetime, f.trade_date, f.minute_index, f.open, f.high, f.low, f.close,
               f.volume_lot, f.volume_share, f.amount_cny, 'ftshare' AS source
        FROM ftshare_minute_bars f
        WHERE NOT EXISTS (SELECT 1 FROM minute_bars h
                          WHERE h.trade_date=f.trade_date AND h.ts_code=f.ts_code AND h.datetime=f.datetime)
    ''')
    conn.execute('''CREATE OR REPLACE VIEW market_daily_aggregated AS
        SELECT ts_code, trade_date, open, high, low, close, volume_lot::DOUBLE AS volume_lot,
               volume_share::DOUBLE AS volume_share, amount_cny, vwap, twap_close,
               bar_count, zero_volume_bar_count, first_bar_time, last_bar_time,
               is_full_session, observation_status, 'legacy_minute' AS source
        FROM daily_aggregated
        UNION ALL
        SELECT f.*, 'ftshare' AS source FROM ftshare_daily_aggregated f
        WHERE NOT EXISTS (SELECT 1 FROM daily_aggregated h
                          WHERE h.trade_date=f.trade_date AND h.ts_code=f.ts_code)
    ''')


def ingest(source: Path, paths: Paths) -> dict:
    summary = json.loads((source/'summary.json').read_text())
    day = date.fromisoformat(summary['date'])
    if summary.get('timezone') != 'Asia/Shanghai' or not summary.get('adjustment','').startswith('none'):
        raise ValueError('Only unadjusted Asia/Shanghai snapshots are supported')
    universe = json.loads((source/'universe.json').read_text())
    symbols = [r['stock_code'] for r in universe['stocks']]
    if len(set(symbols)) != len(symbols) or len(symbols) != summary['universe_count']:
        raise ValueError('Invalid source universe')
    for kind in ['daily','minute']:
        stats=summary['datasets'][kind]
        if stats['errors'] or stats['symbols_attempted'] != len(symbols):
            raise ValueError(f'Incomplete {kind} snapshot')
    required=['minute.jsonl.gz','daily.jsonl.gz','universe.json','summary.json','suspensions.json','validation.json']
    if (source/'listing_dates.json').exists():required.append('listing_dates.json')
    hashes={name:sha256_file(source/name) for name in required}
    target=paths.lake/'canonical'/'ftshare'/f'trade_date={day}'
    if target.exists():
        old=json.loads((target/'manifest.json').read_text())
        if old['input_sha256'] != hashes:
            raise FileExistsError(f'Different snapshot already installed: {target}')
        with duckdb.connect(str(paths.catalog)) as cat: register_views(cat, paths)
        return {'status':'already_ingested', 'path':str(target)}
    paths.ensure_layout()
    stage=paths.staging/f'ftshare-{day}-{os.getpid()}'
    stage.mkdir(exist_ok=False)
    conn=connect()
    try:
        for kind in ['minute','daily']:
            conn.execute(f"CREATE TABLE raw_{kind} AS SELECT * FROM read_json_auto('{sql_path(source/(kind+'.jsonl.gz'))}', format='newline_delimited', maximum_sample_files=1)")
            conn.execute(f'''CREATE TABLE normalized_{kind} AS
                SELECT symbol::VARCHAR AS ts_code,
                    timezone('Asia/Shanghai', to_timestamp(ts_millis / 1000.0)) AS datetime,
                    timezone('Asia/Shanghai', to_timestamp(ts_millis / 1000.0))::DATE AS trade_date,
                    open::DOUBLE AS open, high::DOUBLE AS high, low::DOUBLE AS low, close::DOUBLE AS close,
                    volume::BIGINT AS volume_share, volume::DOUBLE / 100.0 AS volume_lot,
                    turnover::DOUBLE AS amount_cny, turnover_rate::DOUBLE AS turnover_rate,
                    ts_millis::BIGINT AS ts_millis, ts_millis_open::BIGINT AS ts_millis_open
                FROM raw_{kind}''')
            invalid=conn.execute(f'''SELECT count(*) FROM normalized_{kind} WHERE
                ts_code IS NULL OR trade_date IS NULL OR trade_date <> DATE '{day}' OR
                open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR
                NOT isfinite(open) OR NOT isfinite(high) OR NOT isfinite(low) OR NOT isfinite(close) OR
                least(open,high,low,close)<=0 OR high<greatest(open,close) OR low>least(open,close) OR
                volume_share IS NULL OR volume_share<0 OR amount_cny IS NULL OR NOT isfinite(amount_cny) OR amount_cny<0''').fetchone()[0]
            count,unique=conn.execute(f'SELECT count(*),count(DISTINCT (ts_code,datetime)) FROM normalized_{kind}').fetchone()
            if invalid or count!=unique or count!=summary['datasets'][kind]['rows']:
                raise ValueError(f'{kind}: invalid, duplicate or mismatched rows')
        conn.execute('''CREATE TABLE minute AS SELECT ts_code, datetime, trade_date,
            CASE WHEN datetime::TIME BETWEEN TIME '09:30' AND TIME '11:30'
                 THEN date_diff('minute', date_trunc('day',datetime)+INTERVAL '9 hours 30 minutes',datetime)
                 WHEN datetime::TIME BETWEEN TIME '13:01' AND TIME '15:00'
                 THEN 121+date_diff('minute',date_trunc('day',datetime)+INTERVAL '13 hours 1 minute',datetime)
            END::UTINYINT AS minute_index,
            open,high,low,close,volume_lot,volume_share,amount_cny,turnover_rate,ts_millis,ts_millis_open
            FROM normalized_minute''')
        if conn.execute('SELECT count(*) FROM minute WHERE minute_index IS NULL OR second(datetime)<>0').fetchone()[0]:
            raise ValueError('Unexpected trading session timestamp')
        conn.execute('''CREATE TABLE daily AS SELECT ts_code,trade_date,open,high,low,close,
            volume_lot,volume_share,amount_cny,turnover_rate,ts_millis,ts_millis_open FROM normalized_daily''')
        conn.execute('''CREATE TABLE daily_aggregated AS SELECT ts_code,trade_date,
            arg_min(open,datetime) AS open,max(high) AS high,min(low) AS low,arg_max(close,datetime) AS close,
            sum(volume_share)::DOUBLE/100 AS volume_lot,sum(volume_share)::DOUBLE AS volume_share,
            sum(amount_cny) AS amount_cny,sum(amount_cny)/nullif(sum(volume_share),0) AS vwap,
            avg(close) AS twap_close,count(*)::INTEGER AS bar_count,
            count(*) FILTER(WHERE volume_share=0)::INTEGER AS zero_volume_bar_count,
            min(datetime) AS first_bar_time,max(datetime) AS last_bar_time,count(*)=241 AS is_full_session,
            CASE WHEN count(*)<>241 THEN 'partial_observation' WHEN sum(volume_share)=0
                 THEN 'complete_zero_volume' ELSE 'complete_trading' END AS observation_status
            FROM minute GROUP BY ts_code,trade_date''')
        conn.execute('CREATE TABLE universe (ts_code VARCHAR, name VARCHAR, trade_date DATE)')
        conn.executemany('INSERT INTO universe VALUES (?,?,?)',[(r['stock_code'],r['stock_name'],day) for r in universe['stocks']])
        for kind in ['minute','daily']:
            if conn.execute(f'SELECT count(*) FROM {kind} ANTI JOIN universe USING(ts_code)').fetchone()[0]:
                raise ValueError('Rows outside source universe')
            counts=dict(conn.execute(f'SELECT ts_code,count(*) FROM {kind} GROUP BY ts_code').fetchall())
            expected=summary['datasets'][kind]['counts_by_symbol']
            if any(counts.get(sym,0)!=expected.get(sym) for sym in symbols):
                raise ValueError(f'{kind}: per-symbol count mismatch')
        susp=json.loads((source/'suspensions.json').read_text())
        if susp.get('code')!=200:raise ValueError('Missing suspension evidence')
        conn.execute('CREATE TABLE suspensions (ts_code VARCHAR, trade_date DATE, suspension_type VARCHAR, suspend_time VARCHAR, resume_time VARCHAR, reason VARCHAR)')
        records=susp['data']['records']
        if len(records)!=susp['data']['total']:raise ValueError('Incomplete suspension list')
        if records:conn.executemany('INSERT INTO suspensions VALUES (?,?,?,?,?,?)',[(r['symbol'],day,r['suspension_type'],r.get('suspend_time'),r.get('resume_time'),r.get('reason')) for r in records])
        for kind in ['minute','daily','daily_aggregated','universe','suspensions']:
            conn.execute(f"COPY (SELECT * FROM {kind} ORDER BY trade_date,ts_code) TO '{sql_path(stage/(kind+'.parquet'))}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        manifest={'provider':'ftshare','trade_date':str(day),'ingested_at':utc_now(),'input_sha256':hashes,
                  'source_directory':str(source.resolve()),'minute_rows':summary['datasets']['minute']['rows'],
                  'daily_rows':summary['datasets']['daily']['rows'],'timezone':'Asia/Shanghai',
                  'adjustment':'none','volume_share_unit':'share','volume_lot_unit':'100 shares (fractional lots retained)',
                  'historical_baseline_modified':False,'coverage_note':'Single-day source partition; use sync_status.json and the exchange calendar to assess multi-day continuity.'}
        (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        shutil.copy2(source/'validation.json',stage/'source_validation.json')
        if (source/'listing_dates.json').exists():shutil.copy2(source/'listing_dates.json',stage/'listing_dates.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        os.rename(stage,target)
        try:
            with duckdb.connect(str(paths.catalog)) as cat:
                cat.execute('BEGIN')
                register_views(cat, paths)
                cat.execute('COMMIT')
        except Exception:
            os.rename(target,stage)
            raise
        return {'status':'ingested','path':str(target),**manifest}
    finally:
        conn.close()
        shutil.rmtree(stage,ignore_errors=True)


def cmd_ingest_ftshare(args: argparse.Namespace) -> None:
    print(json.dumps(ingest(Path(args.source_dir),Paths(Path(args.data_root))),ensure_ascii=False))
