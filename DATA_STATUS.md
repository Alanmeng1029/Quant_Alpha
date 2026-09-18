# A 股研究数据层：当前状态与缺口

更新时间：2026-08-30。本文记录本地 `A_stock_database` 的真实建设状态；
`daily_aggregated` 均由分钟数据聚合，不应称作交易所官方日线。

## 2026-09-17 增量接入：FTShare

已在同一 DuckDB 目录接入 FTShare 单日快照：5,208 只股票、1,255,128 条分钟线、
5,208 条供应商日线；查询的 5,220 只股票中，12 只无行情且均有停牌记录。
新数据在 `canonical/ftshare/trade_date=2026-09-17/`，统一查询使用
`market_minute_bars` / `market_daily_aggregated`；供应商日线使用 `ftshare_daily_bars`。
原研究基线、观察日历和回测股票池未变更。8 月 28 日至 9 月 17 日之间尚有数据缺口，
新日期缺少复权与基准配套数据。详情和复现命令见 [FTShare 接入说明](docs/ftshare_ingestion.md)。

## 已完成

| 数据集 / DuckDB 视图 | 覆盖与规模 | 主要来源 | 用途与备注 |
| --- | --- | --- | --- |
| `minute_bars` | 2018-01-02 至 2026-08-28；2,205,570,298 行 | 用户提供的分钟 CSV，已转为按交易日分区 Zstd Parquet | 原始 OHLCV；每个完整交易日预期 241 根 bar。原 CSV 已删除，Parquet 是当前唯一历史行情副本。 |
| `daily_aggregated` | 同期；9,190,478 行 | 从 `minute_bars` 聚合 | 原始日频 OHLCV、VWAP、bar 数、观察状态、原始收益。不是官方日线。 |
| `adjustment_snapshots` / `daily_qfq` | 2018-01-02 至 2026-08-28；9,368,405 行 | 用户已有“前复权因子” CSV，快照日 2026-08-28 | 动态关联复权价格及 `qfq_return`；不另存复权分钟行情。 |
| `trading_universe` | 2,101 个观察交易日；9,145,628 个日期×股票成员 | `daily_aggregated` + 有效复权因子 | 条件为 `complete_trading` 且前复权因子有限、正数、有效。暂未加入 ST、上市天数、涨跌停或流动性门槛。 |
| `index_daily` | CSI300、CSI500 各 2,101 日；2018-01-02 至 2026-08-28 | BaoStock：`sh.000300`、`sh.000905`，不复权指数日线 | 官方价格指数的 OHLC、前收、成交量、成交额、日收益。 |
| `daily_excess_returns` | 与日频股票表按交易日连接 | `daily_qfq` + `index_daily` | 提供 `excess_return_vs_csi300`、`excess_return_vs_csi500`。定义为股票前复权收益减指数价格收益。 |
| `index_monthly_constituents` | 2018-01 至 2026-08；104 个月度快照 | GitHub `unliftedq/index-constitution` 的 CSI300/CSI500 纳入/剔除事件 | 由事件在月末重建。CSI300 每月 299–300 只，CSI500 每月 500 只。 |
| `index_trading_universe` | 每日动态交集 | `index_monthly_constituents` ∩ `trading_universe` | 可直接取指数成分内、当日可交易且复权有效的股票。2026-08-28 为 CSI300 300 只、CSI500 500 只。 |

## 存储与查询

```text
A_stock_database/lake/
├── canonical/minute/                 # year/month/trade_date 分区 Parquet
├── canonical/daily_aggregated/       # 按年 Parquet
├── canonical/adjustment/             # 按快照日、按年 Parquet
├── canonical/index_daily/            # CSI300 / CSI500 日线
├── canonical/reference/
│   ├── observed_calendar.parquet
│   ├── instruments.parquet
│   ├── trading_universe/
│   └── index_constituents/
├── quality/                           # 源盘点、覆盖、隔离与校验结果
└── catalog/a_share.duckdb             # DuckDB 查询目录
```

主要查询目录为 `A_stock_database/lake/catalog/a_share.duckdb`。其视图包括：
`minute_bars`、`daily_aggregated`、`daily_qfq`、`trading_universe`、
`index_daily`、`daily_excess_returns`、`index_monthly_constituents`、
`index_trading_universe`、`observed_calendar`、`instrument_day_coverage` 与质量视图。

## 数据来源与可复现性

| 数据 | 来源 | 版本 / 取得方式 | 注意事项 |
| --- | --- | --- | --- |
| 股票分钟行情 | 用户本地原始 CSV | 已转换；原 CSV 已按用户要求删除 | 不能再用原 CSV 重跑历史转换；应备份 `lake/`。 |
| 股票列表 | `A_stock_database/股票列表_沪深.csv` | 用户提供 | 是当前证券主数据，不是历史名称/ST 状态。 |
| 前复权因子 | `A_stock_database/复权因子/复权因子_前复权/` | 快照 `2026-08-28` | 原因子文件保留；无效记录不参与 `daily_qfq`。 |
| 沪深300/中证500日线 | [BaoStock](https://baostock.com/) | `fetch-index-daily` 下载 | 使用不复权价格指数日线；每次刷新应保留下载日期和质量检查。 |
| 沪深300/中证500历史成分 | [unliftedq/index-constitution](https://github.com/unliftedq/index-constitution)，提交 `dd1604c5176176f6cf6418c2a4bab0277b51d4ff` | 公开事件重建；月末快照 | 非官方完整日度成分档，CSI300 存在少量 299 只月份。 |

## 已知质量问题与限制

1. 前复权因子有 4,362 条非正值，日期集中于 2018-01-02 至 2021-07-14；另有 6,110 个已有分钟观察日没有有效因子。因此 `daily_qfq.qfq_return` 和两列超额收益会在这些股票日为 `NULL`。
2. 有 6 条源分钟记录因 OHLC 非法而被隔离，见 `quality/quarantined_records/`；canonical 分钟与日频层不包含它们。
3. `missing` 与 `complete_zero_volume` 只表示观察结果，不能自动解释为停牌。
4. 当前没有历史 ST 状态。股票列表中的当前名称含 ST 的股票约 230 只，但把该状态回填全历史会造成未来函数，因此没有使用。
5. 指数成分只覆盖 CSI300 与 CSI500；中证1000 (`000852.SH`) 尚无可信的 2018 至今历史成员数据，不能以当前名单回填历史。
6. 当前没有官方昨收、涨跌停价、停复牌、除权除息事件、退市状态或历史上市日期状态。
7. 指数基准为价格指数收益，而非全收益指数；股票 `qfq_return` 与价格指数相减的口径应在回测报告中明确说明。

## 进入 Alpha 研究 / 回测前仍需补齐

按优先级：

1. **替换或修复前复权因子。** 这是最优先的数据缺口；应覆盖全部观察证券日且因子为有限正数，并保留供应商、快照日和文件校验和。
2. **中证1000历史成分。** 需要点时成员数据（建议 JQData、Wind、Choice 或可信导出）；导入后按现有月末快照结构落库。
3. **历史 ST / 名称变更。** 回测需要按当日状态排除 ST，而不是用当前名称静态筛选。
4. **停牌、涨跌停和可交易约束。** 至少需要官方停牌标记、昨收及涨跌停价格，才能实现更现实的日频调仓成交约束。
5. **公司行为与退市证券。** 对长期样本、幸存者偏差和复权完整性重要；需要历史证券主数据、上市/退市日期及名称代码变更。
6. **基准收益口径。** 若目标是严格的投资组合超额收益，应补齐 CSI300/CSI500 全收益指数，或明确“股票前复权收益相对价格指数”的近似口径。
7. **交易成本与容量字段。** 日均成交额、自由流通市值、行业、市值、借券/融资信息可在进入正式回测时逐步加入。

## 后续操作

```bash
# 刷新两条指数日线；完成后重建目录
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data fetch-index-daily \
  --start 2018-01-01 --end YYYY-MM-DD
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data build-catalog

# 用新版复权因子建立新快照，并刷新动态股票池
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data ingest-adjustment \
  --snapshot-date YYYY-MM-DD --source-dir /path/to/factors
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data build-universe --replace
```

在每次重要更新后运行 `validate`。前复权因子问题未解决前，该命令会以非零状态退出；这应被视为数据质量提醒，而不是可以忽略的成功状态。
