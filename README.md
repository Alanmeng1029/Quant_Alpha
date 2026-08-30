# A 股本地数据湖

本项目将 `A_stock_database/minute` 中的分钟 CSV 转换为按交易日分区的
Parquet，并从该层聚合日频 OHLCV。原始 CSV 和供应商复权文件永不修改。

运行环境已经存在于 `ml311` Conda 环境：

```bash
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data --help
```

典型顺序：

```bash
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data inventory
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data backfill-minute
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data build-daily
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data ingest-adjustment --snapshot-date 2026-08-28
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data build-universe
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data validate
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data build-catalog
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data status
```

可将可信的指数事件历史转换为月末成分快照；当前导入器支持
`index-constitution` 的 `history/csi300.csv` 与 `history/csi500.csv`：

```bash
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data ingest-index-events \
  --source-dir /path/to/index-constitution/history --replace
```

DuckDB 的 `index_monthly_constituents` 是月末成分快照；
`index_trading_universe` 是其与 `trading_universe` 的逐日交集。该公开源
不含中证1000或历史 ST，二者不会被伪造为已覆盖数据。

`backfill-minute` 默认跳过已存在的交易日分区。历史 CSV 删除后，后续收到
单日或月度 CSV 包时，显式传入该包所在目录：

```bash
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data update-day \
  --trade-date YYYY-MM-DD --source-dir /path/to/received_csvs
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data refresh-month \
  --month YYYY-MM --source-dir /path/to/received_csvs
```

两条命令会将被替换的 Parquet 分区移入 `lake/backups/`，不删除它们。
它们也会同步重建对应年度的 `trading_universe`。当导入了新的复权因子
快照后，再运行一次 `build-universe --replace` 即可按最新快照刷新股票池。

`daily_aggregated` 是从分钟数据计算出的日线，不是交易所官方日线；其中
`missing`、`complete_zero_volume` 都不能解释为停牌。
