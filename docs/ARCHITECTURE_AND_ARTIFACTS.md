# Quant Alpha：四模块架构与已有产物

本项目按以下四个模块组织。它们是有严格输入输出边界的一条研究流水线，而不是四个独立系统：

```text
Data → Factor research + factor production → Model + prediction → Optimizer + backtest
```

本文件描述的是截至 2026-09-02 的代码能力和本机已落盘产物。`A_stock_database/` 和 `results/` 都是本地数据，已被 Git 忽略；它们可复现或审阅，但不会随代码提交传播。外部参考仓库位于 `external/`，同样被忽略。

## 1. Data

### 职责

将本地 A 股分钟 CSV 和复权因子整理成可查询、可验证的历史数据湖，并提供日频研究所需的标准目录与动态股票池。

### 技术栈

- Python 3.11：导入、转换、质量检查与命令行编排。
- Polars / PyArrow / Parquet：列式读取、按交易日或年度分区、Zstd 压缩。
- DuckDB：将 Parquet 注册为研究查询目录和视图。

### 如何实现

1. 分钟 CSV 被规范化为 `canonical/minute/` 下按年月和交易日分区的 Parquet。
2. 由分钟 bar 聚合出 `daily_aggregated`：日频 OHLCV、VWAP、bar 数、观察状态和原始收益。
3. 复权因子按快照写入并与日频表动态关联，形成 `daily_qfq`；不复制一份前复权分钟行情。
4. 结合完整交易日、有效复权因子与股票日频观察，形成 `trading_universe`。
5. CSI300、CSI500 的指数日线与月度成分事件写入后，和可交易股票池取逐日交集，得到 `index_trading_universe`。
6. `validate` 对主键、OHLC、分钟 session、复权和覆盖率做质量校验；`build-catalog` 刷新 DuckDB 视图。

### 当前数据产物

- 数据说明：[DATA_STATUS.md](../DATA_STATUS.md)。
- `A_stock_database/lake/catalog/a_share.duckdb`：研究查询目录。
- 2018-01-02 至 2026-08-28：约 22.06 亿分钟 bar、919 万日频聚合记录、937 万前复权日频记录。
- `index_trading_universe`：CSI300/CSI500 的逐日可交易成分；2026-08-28 分别为 300 和 500 只。

### 关键限制

`daily_aggregated` 来源于分钟数据聚合，不是交易所官方日线。复权因子存在缺失或非正值；没有完整历史 ST、停复牌、涨跌停、上市退市和中证1000 历史成分数据。因此它是可靠的研究底座，但尚不是完整的实盘交易可行性数据集。

## 2. Factor research + factor production

### 职责

研究侧批量计算和评估候选日频因子；生产侧只接收人工批准的因子，保存其审批与来源证据，并安全地持续更新每日完整截面。

### 技术栈

- Python、Polars、Numba、NumPy：日历对齐的长面板因子计算与滚动算子。
- Pandas：作为现有公式和参考实现的兼容路径。
- DuckDB：正式因子库和元数据/审计记录。
- Rust、DuckDB、Parquet、HTML/PDF 报告：批量单因子预测能力评估。

### 如何实现

**研究侧**

- `quant-factor` 读取 `daily_qfq` 和交易日历，先补齐“交易日 × 证券”网格，再在每只证券内执行滚动窗口，避免缺失股票日压缩时间窗口。
- 因子算子包括截面 rank/scale、lag/delta、滚动统计、相关/协方差、线性衰减、时序 rank 等；默认使用 Polars 列式执行，Pandas 路径用于兼容与核验。
- 因子先在完整有效 `daily_qfq` 截面上计算（保留公式中的截面 rank 语义），再仅将逐日 CSI300 ∪ CSI500 成分写入 `lake/derived/factors/<factor>/v1/factor.parquet`。不保存全市场因子值；`manifest.json` 必须记录计算截面、存储股票池、公式、输入字段、引擎、时间区间、行数与 SHA-256。
- `quant-backtest batch-factor-eval` 对每个 Universe 只生成一次市场标签缓存，再顺序评估因子；Python 报告模块生成 HTML/PDF 与汇总表。

**生产侧**

- `quant-factor-store` 首先注册研究 manifest，记录审批人、审批说明、manifest 哈希与来源文件哈希。
- 只有已注册且已启用的因子可以写入。历史导入拒绝区间重叠；日更要求输入仅含目标交易日，使用“整日替换”防止重复行和半截面更新。
- 每个正式因子拥有独立 DuckDB 值表，并维护完成交易日水位线。因此研究因子不会自动进入生产库。

### 当前因子与评估产物

- 因子存量清单：[results/factor_inventory.md](../results/factor_inventory.md)。已构建 WorldQuant Alpha101 的 82 个和 GTJA191 的 190 个；GTJA30 因缺 MKT/SMB/HML 输入未构建。
- Core40 批评估：[results/factor_batches/core40-csi800-v1/batch_report.html](../results/factor_batches/core40-csi800-v1/batch_report.html)，40 项任务成功完成。
- 全量日频 Polars 批评估：[results/factor_batches/daily-polars-272-csi300-csi500-v1/batch_report.html](../results/factor_batches/daily-polars-272-csi300-csi500-v1/batch_report.html)，272 项任务成功完成；结构化汇总见同目录的 `batch_summary.parquet`、`batch_metrics.parquet` 与 `task_status.parquet`。
- 这些评估是单因子的预测诊断，不等同于经过交易约束、成本和组合构建后的可投资策略。

## 3. Model + prediction

### 职责

把候选日频因子变为严格样本外的个股超额收益预测，并留下模型、特征、配置与评估证据。

### 技术栈

- Python、LightGBM：梯度提升树回归。
- Polars、DuckDB、NumPy：宽表特征构建、截面预处理、标签构造和指标计算。

### 如何实现

1. `quant-predict build-features` 从候选因子长表中按每日动态 CSI300 ∪ CSI500 股票池 pivot 出宽表，并用源文件大小和修改时间生成缓存指纹。
2. 每日每个因子先做 1%/99% 截面去极值，再做截面 Z-score；这一步仅使用信号日可见信息。
3. 可执行标签为开盘到开盘：信号日 T 收盘后产生信号，T+1 开盘进入；h1 在 T+2 开盘退出，h5 在 T+6 开盘退出。两者均减去同期 CSI500 开盘到开盘收益，形成超额收益。
4. 每月首个信号日训练两个 LightGBM：严格使用此前 756 个交易日，且与标签之间留 6 个交易日空档；随后预测该月全体信号日。这避免标签尚未实现时进入训练集。
5. 输出独立的 `pred_h1` 与 `pred_h5`；组合层再按可调比例合成，不把持有期假设硬编码进模型。

### 当前模型产物

- 目录：[results/predict/open-open-icir33-excluding-000937-v1](../results/predict/open-open-icir33-excluding-000937-v1)。
- OOS 区间：2021-04-01 至 2026-08-28；`predictions.parquet` 含个股预测和执行日。
- 训练证据：`models/month=*/h1.txt`、`h5.txt` 和 manifest；另有 `feature_importance.parquet`、`model_timings.parquet`。
- 指标：[summary.json](../results/predict/open-open-icir33-excluding-000937-v1/summary.json)。h1 Rank IC 为 0.02865、h5 Rank IC 为 0.02284；这是预测诊断，不能直接推出可投资收益。
- `000937.SZ` 已列入不可执行黑名单，因为供应商复权因子导致非经济性跳变；其他复权异常尚未做系统性清洗。

## 4. Optimizer + backtest

### 职责

将 h1/h5 预测转换为带个股上限和换手约束的目标权重，并用一致的开盘执行价格和成本假设评价组合净收益。

### 技术栈

- Python、NumPy：截面标准化、softmax 目标权重和换手投影。
- DuckDB、Polars：读取行情、动态股票池并写入日度账本。
- Matplotlib、HTML：生成净值、成本、换手、持仓和相对 CSI500 超额报告。

### 如何实现

1. `quant-predict optimize-dual-alpha` 在每个截面分别标准化 `pred_h1` 与 `pred_h5`，按可调比例（当前 50/50）合成，再经 softmax 形成全投资多头目标。
2. 目标权重投影到单票最大 10%，再从昨日持仓向目标线性移动；实际单边换手不超过设定上限（当前 30%）。没有行业、中证500权重或预设 Top-N 约束。
3. `quant-predict backtest-portfolio` 用 T+1 开盘到下一执行日开盘逐日计算毛收益；成交按相邻目标权重之差计算，买入 2.1bp、卖出 7.1bp。
4. `quant-predict render-backtest-report` 输出 Gross、Fee、Net、CSI500、回撤、换手及持仓集中度；CSI500 是 alpha/收益比较基准，Top10 等权仅是已移除的研究对照。

### 当前组合产物与解读

- 目录：[dual-alpha-30pct](../results/predict/open-open-icir33-excluding-000937-v1/dual-alpha-30pct)。
- [目标权重](../results/predict/open-open-icir33-excluding-000937-v1/dual-alpha-30pct/target_weights.parquet)、[日度账本](../results/predict/open-open-icir33-excluding-000937-v1/dual-alpha-30pct/weight-backtest/portfolio_daily.parquet)、[回测报告](../results/predict/open-open-icir33-excluding-000937-v1/dual-alpha-30pct/report/report.html)。
- 1,311 个持有期：净总收益 +151.1%，CSI500 +25.7%，相对财富超额 +99.8%，信息比率 1.13，最大回撤 -31.3%，平均单边换手 17.4%。持股数中位数 796，但有效持股数中位数约 636，因此这是广泛分散的组合而非集中选股策略。

该结果仅是研究回测：当前 `daily_qfq` 的复权质量尚未全面审计，且未模拟停牌、涨跌停、冲击成本、订单容量与一手整数执行。不得直接用于实盘。

## 模块接口总表

| 上游模块 | 下游模块 | 核心交付物 |
| --- | --- | --- |
| Data | Factor research | `a_share.duckdb` 的 `daily_qfq`、日历、动态股票池和行业信息 |
| Factor research | Factor production / Model | 带 manifest 的 `factor.parquet`；通过评估后才可申请注册 |
| Factor production | Model | 已审批、已启用且按日完整更新的正式因子值 |
| Model + prediction | Optimizer | 含 `trade_date`、`execution_date`、`ts_code`、`alpha_daily` 的 `predictions.parquet` |
| Optimizer | Backtest | 含执行日和目标权重的 `target_weights.parquet` |
| Backtest | 决策 | 日度净值、成本、换手、基准对比和风险统计 |

## 参考仓库的边界

`external/minute-factor-references/` 下的三个仓库仅作为公式、建模思路和实验协议参考。它们是独立 Git 仓库，不参与本项目的代码历史，也不应在未明确记录来源、许可和版本的情况下直接复制到生产路径。
