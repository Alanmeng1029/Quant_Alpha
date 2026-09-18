import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from a_share_data.ftshare_sync import completed_day,trade_days,ashare

class SyncTest(unittest.TestCase):
    def test_cutoff(self):
        tz=ZoneInfo('Asia/Shanghai')
        self.assertEqual(completed_day(datetime(2026,9,17,17,59,tzinfo=tz)),date(2026,9,16))
        self.assertEqual(completed_day(datetime(2026,9,17,18,0,tzinfo=tz)),date(2026,9,17))
    def test_calendar_fails_closed_and_skips_holiday(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'calendar.csv'
            p.write_text('交易所,日期,是否交易\nSSE,2026-09-18,交易\nSSE,2026-09-19,休市\n')
            self.assertEqual(trade_days(p,date(2026,9,18),date(2026,9,19)),[date(2026,9,18)])
            with self.assertRaises(ValueError):trade_days(p,date(2026,9,18),date(2026,9,20))
    def test_universe(self):
        for s in ['000001.SZ','300001.SZ','600519.SH','688001.SH']:self.assertTrue(ashare(s))
        for s in ['900901.SH','200001.SZ','430001.BJ','510300.SH']:self.assertFalse(ashare(s))
if __name__=='__main__':unittest.main()

class RangeCacheTest(unittest.TestCase):
    def test_reuses_range_without_leaking_other_dates(self):
        import gzip,json
        from datetime import timedelta
        from a_share_data.ftshare_sync import download_day
        class FakeClient:
            range_start=date(2026,9,1)
            range_end=date(2026,9,3)
            calls=0
            def pages(self,*args):return []
            def get(self,path,params):
                self.calls+=1;rows=[]
                for day in [date(2026,9,1),date(2026,9,2),date(2026,9,3)]:
                    start=datetime.combine(day,datetime.min.time(),ZoneInfo('Asia/Shanghai'))
                    stamps=[start+timedelta(hours=15)] if 'candlesticks' in path else [start+timedelta(minutes=570+i if i<=120 else 780+i-120) for i in range(241)]
                    for stamp in stamps:rows.append({'ts_millis':int(stamp.timestamp()*1000),'volume':101})
                return {'code':200,'data':[['000001.SZ',rows]] if 'candlesticks' in path else [{'symbol':'000001.SZ','items':rows}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);client=FakeClient();u={'stocks':[{'stock_code':'000001.SZ','stock_name':'sample'}]}
            for day in [date(2026,9,1),date(2026,9,2)]:
                r=download_day(client,day,u,root/str(day),2)
                self.assertEqual(r['datasets']['daily']['rows'],1)
                self.assertEqual(r['datasets']['minute']['rows'],241)
                with gzip.open(root/str(day)/'minute.jsonl.gz','rt') as f:
                    dates={datetime.fromtimestamp(json.loads(line)['ts_millis']/1000,ZoneInfo('Asia/Shanghai')).date() for line in f}
                self.assertEqual(dates,{day})
            self.assertEqual(client.calls,2)
