# Quant Alpha：四模块架构与已有产物

本项目按以下四个模块组织。它们是有严格输入输出边界的一条研究流水线，而不是四个独立系统：

```text
Data → Factor research + factor production → Model + prediction → Optimizer + backtest
```

本文件描述的是截至 2026-09-01 的代码能力和本机已落盘产物。`A_stock_database/` 和 `results/` 都是本地数据，已被 Git 忽略；它们可复现或审阅，但不会随代码提交传播。外部参考仓库位于 `external/`，同样被忽略。

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
- 因子原始值保存为 `lake/derived/factors/<factor>/v1/factor.parquet`，并随附 `manifest.json`：公式、输入字段、引擎、时间区间、行数与 SHA-256。
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

把经过筛选的 Core40 日频因子变为严格样本外的个股收益预测，并留下模型、特征、配置与评估证据。

### 技术栈

- Python、LightGBM：梯度提升树回归。
- Polars、DuckDB、NumPy：宽表特征构建、截面预处理、标签构造和指标计算。

### 如何实现

1. `quant-predict build-features` 从 40 个长表因子中按每日动态 CSI300 ∪ CSI500 股票池 pivot 出宽表，并用源文件大小和修改时间生成缓存指纹。
2. 每日每个因子先做 1%/99% 截面去极值，再做截面 Z-score；这一步仅使用信号日可见信息。
3. 标签以 T+1 VWAP 进入，分别计算 1 日和 5 日持有期收益，再减去当日等权截面收益，形成横截面超额收益。
4. 每月首个信号日训练两个 LightGBM：严格使用此前 756 个交易日，且与标签之间留 6 个交易日空档；随后预测该月全体信号日。这避免标签尚未实现时进入训练集。
5. 最终 alpha 为 `0.25 × pred_h1 + 0.75 × pred_h5 / 5`，并输出逐月模型文件、训练 manifest、预测 Parquet、特征重要度和 OOS 指标。

### 当前模型产物

- 目录：[results/predict/core40-lgbm-cs-zscore-v1](../results/predict/core40-lgbm-cs-zscore-v1)。
- OOS 区间：2021-04-01 至 2026-08-28；`predictions.parquet` 含个股预测和执行日。
- 训练证据：`models/month=*/h1.txt`、`h5.txt` 和 manifest；另有 `feature_importance.parquet`、`model_timings.parquet`。
- 指标：[summary.json](../results/predict/core40-lgbm-cs-zscore-v1/summary.json)。融合 alpha 的 1 日 Rank IC 为 0.01984，日化 5 日 Rank IC 为 0.02056；这说明存在弱预测信号，不能直接推出投资收益。

## 4. Optimizer + backtest

### 职责

将模型预测转换为满足风险/交易约束的目标权重，并用一致的执行价格和成本假设评价组合净收益。

### 技术栈

- Rust：性能敏感的组合优化和批量评估命令。
- OSQP：二次规划求解器。
- DuckDB：读取动态股票池、行业和行情。
- Polars / Python：读取目标权重并执行 T+1 VWAP 投组合回测、产出汇总。

### 如何实现

1. `quant-backtest optimize-portfolio` 从预测 Parquet、CSI300/CSI500 动态股票池和行业信息加载每个信号日截面。
2. OSQP 解带全投资、个股最大权重、行业相对基准偏离容忍度和单边换手上限的二次规划；目标使用预测 alpha 并扣减交易成本。
3. 当求解器失败时，使用确定性的行业内高 alpha/低 alpha 权重转移作为可行回退，维持预算与行业合计。
4. `quant-predict backtest-portfolio` 用 T+1 VWAP 目标权重逐日计算毛收益、相邻调仓日的 L1 换手成本、等权动态股票池基准、净值和回撤。

### 当前组合产物与解读

- 目录：[results/predict/core40-lgbm-cs-zscore-v1/optimizer](../results/predict/core40-lgbm-cs-zscore-v1/optimizer)。
- `target_weights.parquet`：优化目标权重；`optimizer_daily.parquet`：求解状态、耗时、换手和约束诊断。
- [portfolio_daily.parquet](../results/predict/core40-lgbm-cs-zscore-v1/optimizer/portfolio_daily.parquet)：逐日收益与净值；[portfolio_summary.json](../results/predict/core40-lgbm-cs-zscore-v1/optimizer/portfolio_summary.json)：汇总指标。
- 当前设置为 10% 个股上限、5% 行业容忍度、30% 单边换手上限、10bp 成本。回测 1,311 天后净总收益为 -26.7%，等权基准为 +33.5%，信息比率为 -3.20，最大回撤 -52.0%。

结论：优化与回测链路已经可运行、可复现，并清楚显示当前参数与预测信号在成本后没有可投资性。下一步应先基于 OOS 和成本敏感性筛选/重训模型，再讨论放宽或改变组合约束；不应把当前权重用于实盘。

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
