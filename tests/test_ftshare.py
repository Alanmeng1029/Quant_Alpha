import gzip
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb
from a_share_data.cli import Paths
from a_share_data.ftshare import ingest, ingest_adjust, register_views

class FTShareTest(unittest.TestCase):
    def fixture(self, base, day='2026-09-17'):
        paths=Paths(base/'db');paths.ensure_layout();source=base/f'input-{day}';source.mkdir()
        d=date.fromisoformat(day)
        stamp=int(datetime(d.year,d.month,d.day,9,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()*1000)
        row=dict(symbol='000001.SZ',open='10',high='11',low='9',close='10.5',volume=101,turnover='1010',turnover_rate=.01,ts_millis=stamp,ts_millis_open=stamp-60000)
        for kind in ['daily','minute']:
            with gzip.open(source/(kind+'.jsonl.gz'),'wt') as f:f.write(json.dumps(row)+'\n')
        stats=dict(errors=[],symbols_attempted=1,rows=1,counts_by_symbol={'000001.SZ':1})
        for name,data in {'summary':dict(date=day,timezone='Asia/Shanghai',adjustment='none',universe_count=1,datasets={'daily':stats,'minute':stats}),
                          'universe':{'stocks':[{'stock_code':'000001.SZ','stock_name':'sample'}]},
                          'suspensions':{'code':200,'data':{'total':0,'records':[]}},'validation':{}}.items():
            (source/(name+'.json')).write_text(json.dumps(data))
        with duckdb.connect(str(paths.catalog)) as c:
            c.execute('''CREATE TABLE IF NOT EXISTS minute_bars(ts_code VARCHAR,datetime TIMESTAMP,trade_date DATE,minute_index UTINYINT,
                open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,volume_lot BIGINT,volume_share BIGINT,amount_cny DOUBLE)''')
            c.execute('''CREATE TABLE IF NOT EXISTS daily_aggregated(ts_code VARCHAR,trade_date DATE,open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,
                volume_lot DOUBLE,volume_share DOUBLE,amount_cny DOUBLE,vwap DOUBLE,twap_close DOUBLE,bar_count INTEGER,
                zero_volume_bar_count INTEGER,first_bar_time TIMESTAMP,last_bar_time TIMESTAMP,is_full_session BOOLEAN,observation_status VARCHAR)''')
        return paths,source

    def write_factors(self, base, day, rows):
        target=base/f'input-{day}'
        (target/'adjust_factors.json').write_text(json.dumps({'code':200,'data':{'records':rows,'total':len(rows),'pages':1}}))
        return target

    def test_units_idempotency_and_legacy_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            paths,source=self.fixture(Path(temp))
            self.assertEqual(ingest(source,paths)['status'],'ingested')
            self.assertEqual(ingest(source,paths)['status'],'already_ingested')
            with duckdb.connect(str(paths.catalog)) as c:
                self.assertEqual(c.execute('SELECT volume_share,volume_lot,minute_index,datetime FROM market_minute_bars').fetchone(),(101,1.01,0,datetime(2026,9,17,9,30)))
                self.assertEqual(c.execute('SELECT count(*) FROM market_daily_aggregated').fetchone()[0],1)
                c.execute("INSERT INTO minute_bars VALUES ('000001.SZ','2026-09-17 09:30','2026-09-17',0,10,11,9,10.5,1,100,1000)")
                register_views(c,paths)
                self.assertEqual(c.execute('SELECT volume_share,source FROM market_minute_bars').fetchall(),[(100,'legacy_minute')])
            (source/'validation.json').write_text('{"changed":true}')
            with self.assertRaises(FileExistsError):ingest(source,paths)

    def test_reject_duplicate_without_installing(self):
        with tempfile.TemporaryDirectory() as temp:
            paths,source=self.fixture(Path(temp))
            p=source/'minute.jsonl.gz'
            with gzip.open(p,'rt') as f:line=f.read()
            with gzip.open(p,'wt') as f:f.write(line*2)
            with self.assertRaises(ValueError):ingest(source,paths)
            self.assertFalse(list((paths.lake/'canonical'/'ftshare').glob('trade_date=*')))

    def test_adjust_factor_ingest_anchoring_and_views(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            paths,d1=self.fixture(base,'2026-09-17')
            _,d2=self.fixture(base,'2026-09-18')
            ingest(d1,paths);ingest(d2,paths)
            # factor follows the vendor convention (non-decreasing, anchored at 1 on
            # the latest date): a 2:1 split between the days halves day 1's factor
            f1=self.write_factors(base,'2026-09-17',[{'symbol':'000001.SZ','trade_date':'20260917','adj_factor':0.5,'ex_adj_factor':100.0}])
            f2=self.write_factors(base,'2026-09-18',[{'symbol':'000001.SZ','trade_date':'20260918','adj_factor':1.0,'ex_adj_factor':200.0}])
            self.assertEqual(ingest_adjust(f1,paths)['status'],'ingested')
            self.assertEqual(ingest_adjust(f1,paths)['status'],'already_ingested')
            self.assertEqual(ingest_adjust(f2,paths)['status'],'ingested')
            with duckdb.connect(str(paths.catalog)) as c:
                register_views(c,paths)
                rows=c.execute('''SELECT q.trade_date,q.qfq_close,h.hfq_close,q.qfq_ratio
                    FROM ftshare_daily_qfq q JOIN ftshare_daily_hfq h USING (ts_code,trade_date) ORDER BY q.trade_date''').fetchall()
                self.assertEqual([(r[0],r[1],r[2],r[3]) for r in rows],
                                 [(date(2026,9,17),5.25,1050.0,0.5),(date(2026,9,18),10.5,2100.0,1.0)])
                self.assertAlmostEqual(c.execute('''SELECT qfq_return FROM ftshare_daily_qfq WHERE trade_date=DATE '2026-09-18' ''').fetchone()[0],1.0)
            # market_daily_qfq is pure FTShare-factor on the ftshare leg (BaoStock
            # is a manual cross-check dataset, not preferred), legacy leg first
            with duckdb.connect(str(paths.catalog)) as c:
                c.execute('''CREATE VIEW daily_qfq AS SELECT '000001.SZ' AS ts_code, DATE '2026-09-17' AS trade_date,
                    1.0 AS qfq_open,1.0 AS qfq_high,1.0 AS qfq_low,1.0 AS qfq_close,1.0 AS qfq_vwap,1 AS volume_share,1.0 AS amount_cny''')
                c.execute('''CREATE VIEW baostock_qfq_daily AS SELECT '000001.SZ' AS ts_code, DATE '2026-09-18' AS trade_date,
                    99.0 AS open,99.0 AS high,99.0 AS low,99.0 AS close,99.0 AS volume,99.0 AS amount''')
                register_views(c,paths)
                merged=c.execute('''SELECT trade_date,qfq_open,source FROM market_daily_qfq ORDER BY trade_date''').fetchall()
                self.assertEqual(merged,[(date(2026,9,17),1.0,'legacy_minute'),(date(2026,9,18),10.0,'ftshare')])
            self.write_factors(base,'2026-09-18',[{'symbol':'000001.SZ','trade_date':'20260918','adj_factor':0.7,'ex_adj_factor':100.0}])
            with self.assertRaises(FileExistsError):ingest_adjust(f2,paths)
            wrong=self.write_factors(base,'2026-09-17',[{'symbol':'000001.SZ','trade_date':'20260919','adj_factor':1.0,'ex_adj_factor':100.0}])
            with self.assertRaises(ValueError):ingest_adjust(wrong,paths)

if __name__=='__main__':unittest.main()
