# 收盘型分钟因子生产框架

## 目标与边界

本框架用于从分钟 OHLCV 与成交额数据构建**收盘后可得的日频股票因子**。
分钟数据是构建输入，而不是最终因子频率；每个因子最终仍输出：

```text
trade_date, ts_code, factor_value
```

信号在交易日收盘后才完整可得，评估与执行必须采用下一交易日的可成交价格口径（当前默认研究口径为 T+1 VWAP）。本框架不产生盘中持仓、盘中信号或分钟级因子输出。

## 数据契约

分钟行情的主键为：

```text
(trade_date, minute_index, ts_code)
```

其中 `minute_index` 覆盖每个完整交易日的 241 根 bar。原始输入字段为：

```text
open, high, low, close, volume_share, amount_cny
```

`bar_vwap` 为派生字段，定义为 `amount_cny / volume_share`；当成交量为零时为缺失值。日内 VWAP 必须定义为全日成交额除以全日成交量，而不是逐分钟 VWAP 的简单均值。

分钟原始价格可用于同一交易日内的收益、波动与路径特征。涉及跨交易日价格比较时，必须使用前复权日频价格或等价的按日复权转换，不能直接串联原始分钟价格。

## 存储原则

- 保持分钟数据为按交易日分区的长表 Parquet。
- 不持久化全历史的 `time × code` 宽矩阵：它规模大、缺失多，并会放大内存与临时副本成本。
- 计算时可将单个交易日临时视为 `241 × code` 的矩阵；这只是执行视图，不是数据存储格式。
- 生产结果与现有日频因子保持同一长表 schema，以便接入既有研究、评估和正式因子库。

## 计算分层

```text
minute_bars
  -> 日内特征层
  -> 历史标准化层
  -> 每日截面层
  -> 日频 factor.parquet
  -> T+1 评估与执行
```

### 1. 日内特征层

对每个 `(trade_date, ts_code)`，按 `minute_index` 排序，使用当天完整分钟路径构建原始特征，例如日内 VWAP、收盘相对 VWAP、已实现波动率、趋势效率、早午盘收益、尾盘收益、成交额集中度与价格冲击代理。

该层只依赖当日数据，因此可以按交易日并行，也可以在单日内按股票并行。

### 2. 历史标准化层

若特征需要与历史比较，例如“当日尾盘成交额占比相对过去 20 个交易日的异常程度”，先完成所有当日原始特征，再按 `ts_code` 和交易日历执行历史窗口。

所有窗口必须先滞后一日：

```text
baseline[d] = rolling_stat(raw_feature[d-N : d-1])
```

缺失股票日不得压缩时间窗口；应按交易日历对齐。若按日期块并行，块开始处必须带上足够的历史 warm-up 数据，计算后仅保留目标日期的输出。

### 3. 每日截面层

对同一 `trade_date` 的可交易股票执行去极值、rank 或 z-score 等截面变换。股票池必须使用信号日可见的动态股票池，不能以未来成分或当前静态名单回填历史。

## 并行与因果性

“按天并行”与时间因果性不矛盾：

- 纯当日路径特征：按天完全独立，可直接并行。
- 历史标准化特征：逻辑上只引用过去日期；实现上可以用窗口算子、按股票分区，或按带 warm-up 的日期块并行。
- 每日截面变换：不同日期之间独立，可并行。

禁止使用未来分钟 bar、未来交易日数据、全样本确定因子方向或全样本确定参数后再宣称样本外结果。

## 生产检查清单

- [ ] 因子只使用收盘前或收盘时已经完整可见的分钟数据。
- [ ] 跨日价格使用前复权或等价的复权转换。
- [ ] 历史统计均通过 `shift(1)` 排除当日与未来信息。
- [ ] 窗口按交易日历而不是股票有效行数计算。
- [ ] 输出主键为 `(trade_date, ts_code)`，且唯一。
- [ ] 因子方向、参数与筛选规则在训练期确定；报告样本外表现。
- [ ] 收盘信号使用 T+1 的可成交标签和成本假设评估。

## OHLCV 候选集生产

`ohlcv_candidates_v1` 是与 `core24` 隔离的数据集，只写入新增的 45 个分钟 OHLCV 候选列；日内全市场计算完成后，输出仅保留信号日 CSI300∪CSI500 成分。它不会改写既有 `core24` 目录。

```bash
cargo build --release -p quant-minute-factor
target/release/quant-minute-factor build \
  --factor-set ohlcv_candidates_v1 \
  --catalog A_stock_database/lake/catalog/a_share.duckdb \
  --minute-root A_stock_database/lake/canonical/minute \
  --output A_stock_database/lake/derived/minute_factors/ohlcv_candidates_v1 \
  --start 2018-01-02 --end 2026-08-28 \
  --jobs 4 --threads-per-job 2 --memory-limit-mb 12000
```

每日文件为 `year=YYYY/YYYY-MM-DD.parquet`，schema 为 `trade_date, ts_code` 加 45 个候选因子列；根目录 `manifest.json` 记录输入、窗口和逐日完成状态。重跑已完成日期会自动跳过，使用 `--replace` 才重算请求区间。

频谱候选在去除过去 20 日同分钟金额份额季节性后，只评估预注册的 1–30 个非零低中频点；峰值占比和中频带占比均在此网格内定义。
