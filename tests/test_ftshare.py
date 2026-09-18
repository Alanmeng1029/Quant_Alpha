import gzip
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb
from a_share_data.cli import Paths
from a_share_data.ftshare import ingest, register_views

class FTShareTest(unittest.TestCase):
    def fixture(self, base):
        paths=Paths(base/'db');paths.ensure_layout();source=base/'input';source.mkdir()
        stamp=int(datetime(2026,9,17,9,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()*1000)
        row=dict(symbol='000001.SZ',open='10',high='11',low='9',close='10.5',volume=101,turnover='1010',turnover_rate=.01,ts_millis=stamp,ts_millis_open=stamp-60000)
        for kind in ['daily','minute']:
            with gzip.open(source/(kind+'.jsonl.gz'),'wt') as f:f.write(json.dumps(row)+'\n')
        stats=dict(errors=[],symbols_attempted=1,rows=1,counts_by_symbol={'000001.SZ':1})
        for name,data in {'summary':dict(date='2026-09-17',timezone='Asia/Shanghai',adjustment='none',universe_count=1,datasets={'daily':stats,'minute':stats}),
                          'universe':{'stocks':[{'stock_code':'000001.SZ','stock_name':'sample'}]},
                          'suspensions':{'code':200,'data':{'total':0,'records':[]}},'validation':{}}.items():
            (source/(name+'.json')).write_text(json.dumps(data))
        with duckdb.connect(str(paths.catalog)) as c:
            c.execute('''CREATE TABLE minute_bars(ts_code VARCHAR,datetime TIMESTAMP,trade_date DATE,minute_index UTINYINT,
                open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,volume_lot BIGINT,volume_share BIGINT,amount_cny DOUBLE)''')
            c.execute('''CREATE TABLE daily_aggregated(ts_code VARCHAR,trade_date DATE,open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,
                volume_lot DOUBLE,volume_share DOUBLE,amount_cny DOUBLE,vwap DOUBLE,twap_close DOUBLE,bar_count INTEGER,
                zero_volume_bar_count INTEGER,first_bar_time TIMESTAMP,last_bar_time TIMESTAMP,is_full_session BOOLEAN,observation_status VARCHAR)''')
        return paths,source

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

if __name__=='__main__':unittest.main()
