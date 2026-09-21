# FTShare 数据接入

2026-09-17 的沪深 A 股快照已经接入现有 `A_stock_database/lake/catalog/a_share.duckdb`。
数据存储在 `lake/canonical/ftshare/trade_date=2026-09-17/`，使用 Zstd Parquet。

## 查询入口

| DuckDB 视图 | 含义 |
| --- | --- |
| `market_minute_bars` | 旧库分钟行情 + FTShare 分钟行情；相同股票、日期、时间戳以旧库为准；带 `source` |
| `market_daily_aggregated` | 旧库分钟聚合日线 + FTShare 分钟聚合日线；相同股票日以旧库为准；带 `source` |
| `ftshare_minute_bars` | FTShare 不复权分钟行情，含源时间戳、换手率 |
| `ftshare_daily_bars` | FTShare 直接返回的不复权日线 |
| `ftshare_daily_aggregated` | 仅从 FTShare 分钟行情重新聚合的日线 |
| `ftshare_universe` | 供应商下载时点的沪深 A 股列表，不用于回填历史成员 |
| `ftshare_suspensions` | 对应交易日的停牌记录 |

```sql
SELECT ts_code, datetime, open, high, low, close,
       volume_share, amount_cny, source
FROM market_minute_bars
WHERE trade_date = DATE '2026-09-17' AND ts_code = '000001.SZ'
ORDER BY datetime;

SELECT * FROM ftshare_daily_bars
WHERE trade_date = DATE '2026-09-17';
```

原 `minute_bars`、`daily_aggregated`、`daily_qfq`、`observed_calendar`、`trading_universe`
以及直接扫描旧 canonical 目录的因子脚本仍使用原研究基线。需要读取新数据时使用上面的
`market_*` 或 `ftshare_*` 入口。`build-catalog` 会保留、重建这些新视图。

## 字段与质量

- `datetime` 是上海本地无时区时间，取供应商 K 线结束时间；`minute_index` 为 0–240。
- `volume_share` 保留原始股数；`volume_lot = volume_share / 100.0` 保留小数，不能强转整数。
- OHLC、成交额统一为 DOUBLE；原始 JSON 字符串、时间戳及文件校验和仍有留档。
- 5,220 只股票全部查询；5,208 只各 241 根分钟线，共 1,255,128 行；12 只无行情，均有停牌记录。
- 分钟主键无重复，价格和单位检查通过；159,868 行存在非整百股成交量。
- 供应商日线与分钟聚合日线在 42 只开盘价、3 只最高价上不同；分别保留，不相互覆盖。
- 旧研究基线截至 2026-08-28；本次只增加 2026-09-17，中间日期尚未补齐。
- 本次未下载复权因子、基准日线或历史成员，未计算跨缺口收益，未刷新回测股票池。

## 可复现入库

```bash
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data ingest-ftshare \
  --source-dir results/ftshare/2026-09-17
```

输入包含 `minute.jsonl.gz`、`daily.jsonl.gz`、`universe.json`、`summary.json`、
`suspensions.json`、`validation.json`。适配器验证日期、单位、OHLC、交易时段、主键、
逐股记录数与下载完整性，在 staging 写完后安装日期分区，并事务注册目录视图。
相同校验和重复入库为 no-op；同一天内容不同则拒绝覆盖。

入库清单位于日期分区的 `manifest.json`，源质量差异保留在 `source_validation.json`。
查询验证结果在 `results/ftshare/2026-09-17/database_validation.json`。
本次修改前 DuckDB 目录备份在 `lake/backups/ftshare_catalog/20260917T215700/`。

## 每日增量同步与缺口补齐

每日入库口径为“FTShare 裸行情 + FTShare 复权因子”，qfq 只在视图层派生，不物化、不改写已入库分区。完整流程按以下顺序执行：

1. 按交易所日历下载 FTShare 当日不复权日线与 1 分钟行情，并完成现有的完整性、
   主键、交易时段、OHLC、成交量和停牌校验（原有第 1 步，不变）。
2. 对每个已入库日期调用 FTShare `stock-adjust-factor` 下载全市场复权因子
   （`adj_factor` 当日锚定为 1，`ex_adj_factor` 为后复权累计因子），保存
   `adjust_factors.json` 后原子安装到独立目录 `lake/canonical/ftshare_adjust/`，
   已有 bar 分区保持不可变。历史缺因子日期在每次运行时自动补齐。
3. `ftshare_adjust_ratio` 视图按每只股票**最新已存因子**重新锚定：
   `qfq_ratio = adj_factor / first_value(adj_factor ORDER BY trade_date DESC)`，
   因此 `ftshare_daily_qfq` 的 qfq OHLC/VWAP 与 `qfq_return` 在序列内自洽；
   新增除权事件后无需手工重算，视图自动重锚。`ftshare_daily_hfq` 用
   `ex_adj_factor` 直接给出后复权价格。
4. 合并视图 `market_daily_qfq`：旧 `daily_qfq`（≤基线，`source='legacy_minute'`）
   优先，其后全部日期用 FTShare 因子推算值（`source='ftshare'`）。
5. 交叉验证结论（2026-09-18，CSI300/500 重叠 20,776 股票日）：无除权股票两源
   qfq 中位差异为 0；差异仅集中于窗口内除权股（0.3%），且为事件日口径差
   ——FTShare 因子将除权调整作用于事件日前一日（close-to-close 收益干净），
   BaoStock 从事件日起调整（事件日 K 线含跳空）。因子数值两家一致。
6. `baostock_sync` 为**手动交叉验证工具**（不在每日流程）：增量刷新 BaoStock
   `adjustflag=2` qfq 日线，股票池=已存文件∪FTShare 全市场名单；追加前校验
   `preclose` 连续性，除权自动全历史重拉。BaoStock 逐股接口单次 1–7 秒，
   全市场耗时小时级，故仅按需运行。

统一入口：

```bash
PYTHONPATH=src /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data.ftshare_sync
# 手动交叉验证（可选，不在自动任务内）：
# python -m a_share_data.baostock_sync
```

- 从旧研究基线末日次日起，按本地 SSE 交易日历枚举到最新可下载日期，逐个检查已入库 manifest，补齐中间遗漏日期，而不是只取最大日期的下一天。
- 北京时间 18:00 前不下载当天；每日自动任务在 18:30 运行。周末/节假日跳过；停机后下一次运行会补缺口。
- 本地日历目前覆盖至 2026-12-31；覆盖不足时明确失败，不将普通工作日直接当交易日。
- 每批 10 只，默认 4 个下载线程、全局每秒最多 10 次请求，遇到限流/服务错误指数退避重试。日线复用月内范围请求，分钟线复用最多 3 个自然日的范围请求（每股上限 1,000 条，遇到触顶即拒绝），随后按时间戳拆分每日；多日缓存位于 results/ftshare/range_cache/。
- 当前 FTShare 股票列表与旧库最后观察日证券取并集；按下载时点名单查询，不声称重建了点时历史股票池。
- 日期内固定保存股票列表；批次缓存使用接口、股票列表和日期参数的哈希命名，避免名单变化后复用错批次。
- 锁文件防止两个同步进程同时写入。已完成原始下载可直接入库；中断批次从缓存恢复；已有分区不覆盖。
- 日线与分钟必须匹配，有行情股票须有 241 根 bar，且日线覆盖不少于尝试股票的 95%；否则不入库并报告。此门槛可能阻止存在合法部分时段行情的日期，届时需根据供应商/交易所证据处理，不得静默放宽。
- 运行状态：`results/ftshare/sync_status.json`；本次补数日志：`results/ftshare/sync_run.log`。
  每日全流程仅 FTShare，单日约 5–10 分钟；BaoStock 手动运行时约 0.8 秒/只（增量）/ 4 秒/只（全历史）。
- 自动任务（ZCode）：`沪深A股每日下载入库（每天18:30）`，每天北京时间 18:30 运行
  ftshare_sync；codex 中同名任务（ID `a`）保持暂停。
- 本地任务依赖电脑开机、应用运行及网络可用；完成新增入库、失败或需要操作时通知，状态未变化时保持安静。

只查看需要补齐的日期：`python -m a_share_data.ftshare_sync --plan`。
可用 `--start YYYY-MM-DD --end YYYY-MM-DD` 限定范围；不允许尚未达到当天下载截止时间的日期。
