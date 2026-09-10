from datetime import date
import duckdb
import polars as pl
from a_share_data.predict import backtest_targets


def test_band_preserves_small_drift_but_allows_exit(tmp_path):
    db=tmp_path/'db.duckdb';c=duckdb.connect(str(db))
    c.execute('create table daily_qfq(trade_date DATE, ts_code VARCHAR, open DOUBLE, qfq_open DOUBLE)')
    c.execute('create table index_daily(trade_date DATE, index_code VARCHAR, open DOUBLE)')
    days=[date(2024,1,i) for i in (2,3,4,5)]
    for day in days:
        c.execute('insert into daily_qfq values (?, ?, 10, 10)',[day,'A'])
        c.execute('insert into daily_qfq values (?, ?, 10, 10)',[day,'B'])
        c.execute('insert into index_daily values (?, ?, 100)',[day,'000905.SH'])
    c.close()
    pl.DataFrame({'execution_date':days,'ts_code':['A','A','B','B'],'target_weight':[.5,.501,.5,.5]}).write_parquet(tmp_path/'target.parquet')
    backtest_targets(db,tmp_path/'target.parquet',tmp_path/'out',initial_capital=100000,lot_size=1,rebalance_band=.0025)
    trades=pl.read_parquet(tmp_path/'out/executions.parquet')
    assert trades.filter(pl.col('execution_date')==days[1]).is_empty()
    assert trades.filter((pl.col('execution_date')==days[2]) & (pl.col('ts_code')=='A') & (pl.col('side')=='sell')).height==1
