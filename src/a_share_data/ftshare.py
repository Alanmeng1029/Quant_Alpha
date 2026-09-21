"""Auditable FTShare snapshot ingestion; preserve the historical research baseline."""
from __future__ import annotations

import argparse
import json
import math
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
    # Adjustment factors live in their own tree so installed bar partitions stay
    # immutable; qfq/hfq are derived views, never stored prices.
    adjust_root = paths.lake / 'canonical' / 'ftshare_adjust'
    if not list(adjust_root.glob('trade_date=*/adjust_factors.parquet')):
        return
    conn.execute(f"CREATE OR REPLACE VIEW ftshare_adjust_factors AS SELECT * FROM read_parquet('{sql_path(adjust_root / 'trade_date=*' / 'adjust_factors.parquet')}', hive_partitioning=false)")
    # Each day's vendor factor is anchored at that download date; re-anchor to the
    # latest stored factor per stock so the whole series stays return-consistent.
    conn.execute('''CREATE OR REPLACE VIEW ftshare_adjust_ratio AS
        SELECT ts_code, trade_date, adj_factor,
               adj_factor / first_value(adj_factor) OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS qfq_ratio
        FROM ftshare_adjust_factors''')
    conn.execute('''CREATE OR REPLACE VIEW ftshare_daily_qfq AS
        SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
               d.volume_lot, d.volume_share, d.amount_cny, d.turnover_rate,
               d.ts_millis, d.ts_millis_open,
               a.trade_date AS adjustment_trade_date, a.adj_factor, a.qfq_ratio,
               d.open*a.qfq_ratio AS qfq_open, d.high*a.qfq_ratio AS qfq_high,
               d.low*a.qfq_ratio AS qfq_low, d.close*a.qfq_ratio AS qfq_close,
               d.amount_cny/nullif(d.volume_share,0)*a.qfq_ratio AS qfq_vwap,
               d.close*a.qfq_ratio
                 / nullif(lag(d.close*a.qfq_ratio) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date),0) - 1 AS qfq_return
        FROM ftshare_daily_bars d JOIN ftshare_adjust_ratio a USING (ts_code, trade_date)''')
    conn.execute('''CREATE OR REPLACE VIEW ftshare_daily_hfq AS
        SELECT d.*, f.ex_adj_factor,
               d.open*f.ex_adj_factor AS hfq_open, d.high*f.ex_adj_factor AS hfq_high,
               d.low*f.ex_adj_factor AS hfq_low, d.close*f.ex_adj_factor AS hfq_close
        FROM ftshare_daily_bars d JOIN ftshare_adjust_factors f USING (ts_code, trade_date)''')
    if conn.execute("SELECT count(*) FROM duckdb_views() WHERE view_name='daily_qfq'").fetchone()[0]:
        # Full-market qfq is FTShare-factor derived; BaoStock stays available as a
        # manual cross-validation dataset but is not preferred here.
        conn.execute('''CREATE OR REPLACE VIEW market_daily_qfq AS
            SELECT ts_code, trade_date, qfq_open, qfq_high, qfq_low, qfq_close, qfq_vwap,
                   volume_share, amount_cny, 'legacy_minute' AS source FROM daily_qfq
            UNION ALL
            SELECT f.ts_code, f.trade_date, f.qfq_open, f.qfq_high, f.qfq_low, f.qfq_close, f.qfq_vwap,
                   f.volume_share, f.amount_cny, 'ftshare' AS source
            FROM ftshare_daily_qfq f
            WHERE NOT EXISTS (SELECT 1 FROM daily_qfq h
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


def ingest_adjust(source: Path, paths: Paths) -> dict:
    """Install one day's vendor adjustment factors beside the immutable bar partitions."""
    day = date.fromisoformat(json.loads((source/'summary.json').read_text())['date'] if (source/'summary.json').exists() else source.name)
    envelope = json.loads((source/'adjust_factors.json').read_text())
    if envelope.get('code') != 200:
        raise ValueError('Missing adjustment-factor evidence')
    records = envelope['data']['records']
    if len(records) != envelope['data']['total']:
        raise ValueError('Incomplete adjustment-factor list')
    factors: dict[str, tuple[float, float]] = {}
    for r in records:
        symbol, trade_date = r['symbol'], str(r['trade_date'])
        adj, ex = float(r['adj_factor']), float(r['ex_adj_factor'])
        if trade_date != day.strftime('%Y%m%d'):
            raise ValueError(f'Adjustment factor trade_date {trade_date} does not match partition {day}')
        if symbol in factors or not (math.isfinite(adj) and math.isfinite(ex)) or adj <= 0 or ex <= 0:
            raise ValueError('Invalid or duplicate adjustment factor')
        factors[symbol] = (adj, ex)
    vendor = paths.lake/'canonical'/'ftshare'/f'trade_date={day}'
    without_factor: list[str] = []
    if (vendor/'daily.parquet').exists():
        conn = duckdb.connect()
        try:
            conn.execute(f"CREATE VIEW bars AS SELECT ts_code FROM read_parquet('{sql_path(vendor/'daily.parquet')}')")
            conn.execute('CREATE TABLE factor_symbols(ts_code VARCHAR)')
            conn.executemany('INSERT INTO factor_symbols VALUES (?)', [(s,) for s in factors])
            without_factor = [r[0] for r in conn.execute(
                'SELECT ts_code FROM bars ANTI JOIN factor_symbols USING (ts_code) ORDER BY ts_code').fetchall()]
        finally:
            conn.close()
        if len(without_factor) > len(factors) * 0.001:
            raise ValueError(f'{day}: {len(without_factor)} bar symbols lack adjustment factors; not ingesting')
    digest = sha256_file(source/'adjust_factors.json')
    target = paths.lake/'canonical'/'ftshare_adjust'/f'trade_date={day}'
    if target.exists():
        old = json.loads((target/'manifest.json').read_text())
        if old['input_sha256'] != digest:
            raise FileExistsError(f'Different adjustment factors already installed: {target}')
        with duckdb.connect(str(paths.catalog)) as cat:
            register_views(cat, paths)
        return {'status': 'already_ingested', 'path': str(target)}
    paths.ensure_layout()
    stage = paths.staging/f'ftshare-adjust-{day}-{os.getpid()}'
    stage.mkdir(exist_ok=False)
    conn = connect()
    try:
        conn.execute('CREATE TABLE factors(ts_code VARCHAR, trade_date DATE, adj_factor DOUBLE, ex_adj_factor DOUBLE)')
        conn.executemany('INSERT INTO factors VALUES (?,?,?,?)', [(s, day, a, e) for s, (a, e) in factors.items()])
        count, unique = conn.execute('SELECT count(*), count(DISTINCT ts_code) FROM factors').fetchone()
        if count != unique:
            raise ValueError('Duplicate adjustment-factor symbol')
        conn.execute(f"COPY (SELECT * FROM factors ORDER BY ts_code) TO '{sql_path(stage/'adjust_factors.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        manifest = {'provider': 'ftshare', 'trade_date': str(day), 'ingested_at': utc_now(), 'input_sha256': digest,
                    'source_directory': str(source.resolve()), 'symbols': len(factors),
                    'bars_without_factor': without_factor,
                    'anchor_note': 'qfq_ratio is derived at query time as adj_factor / max(adj_factor) per stock over stored dates; vendor factors are point-in-time per trade_date.'}
        (stage/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, target)
        try:
            with duckdb.connect(str(paths.catalog)) as cat:
                cat.execute('BEGIN')
                register_views(cat, paths)
                cat.execute('COMMIT')
        except Exception:
            os.rename(target, stage)
            raise
        return {'status': 'ingested', 'path': str(target), **manifest}
    finally:
        conn.close()
        shutil.rmtree(stage, ignore_errors=True)


def cmd_ingest_ftshare(args: argparse.Namespace) -> None:
    print(json.dumps(ingest(Path(args.source_dir), Paths(Path(args.data_root))), ensure_ascii=False))


def cmd_ingest_ftshare_adjust(args: argparse.Namespace) -> None:
    print(json.dumps(ingest_adjust(Path(args.source_dir), Paths(Path(args.data_root))), ensure_ascii=False))
