# Quant_Alpha 架构与产物

更新：2026-09-20。

项目是一条有明确数据契约的研究流水线：

```text
Data → Factor research / production → Model / prediction → Portfolio policy / backtest
```

`A_stock_database/`、`results/` 和 `external/` 都是本地目录并被 Git 忽略。仓库只发布可复现代码、版本化配置、研究结论和轻量回测快照。

## 当前版本指针

| 层 | 当前正式口径 | 版本来源 |
| --- | --- | --- |
| 数据 | 2018-01-02 至 2026-08-28 的本地分钟湖与聚合日频 | `DATA_STATUS.md` |
| 因子 | `o2o_raw_daily60_minute45_dos20_v1`，125 因子，状态 `active` | `configs/formal_factor_sets/o2o_raw_daily60_minute45_dos20_v1.json` |
| 模型 | 默认 LightGBM 回归 H1，季度滚动样本外 | `results/predict/research-oos-raw-daily60-minute45-dos20-csi300-csi500-h1-h5-h10-v1/` |
| 组合 | 历史 CSI500 内，H1/H5 五期滚动 optimizer；80%单票1%核心与20%单票5%增强目标净额合并，无风险项 | `configs/production_strategy_csi500_h1h5_blend_80_20_2bps_v5.json` |
| 回测 | 2021-04-02 至 2026-08-27，已计买卖费用 | `docs/PRODUCTION_BACKTEST.md` |

旧105因子和98因子版本均已标记为 `superseded`。它们仍是有效的历史基线，但不应在新文档中称为“当前正式版本”。

## 1. Data

### 职责

把供应商分钟 CSV 和复权因子转换为可查询、可验证的历史数据湖，向下游提供交易日历、复权日频行情和点时股票池。

### 实现与产物

1. 分钟 CSV 规范化为按年月和交易日分区的 Zstd Parquet。
2. 分钟 bar 聚合为 `daily_aggregated`，包含 OHLCV、VWAP、bar 数和观察状态。
3. 复权快照与日频表动态关联为 `daily_qfq`，不复制复权分钟行情。
4. `trading_universe` 由完整交易观察和有效复权因子生成。
5. CSI300/CSI500 的历史成分事件与可交易股票池相交为 `index_trading_universe`。
6. DuckDB 目录 `A_stock_database/lake/catalog/a_share.duckdb` 暴露统一查询视图。

数据质量、规模、来源版本和缺口统一记录在 [`DATA_STATUS.md`](../DATA_STATUS.md)。

### 下游契约

- 因子计算只能读取信号日可见数据。
- 股票池必须使用点时成分，不能用当前名单回填历史。
- `daily_aggregated` 不是交易所官方日线。
- 复权缺失、非正值和不可执行代码必须显式暴露，不能静默填补。

## 2. Factor research / production

### 研究侧

Python/Polars 和 Rust 引擎在完整历史截面计算候选因子；批量评估使用统一市场标签缓存，输出 IC、ICIR、覆盖率和稳定性诊断。分钟候选先从分钟 OHLCV 生产为日频宽表，再与日频因子合并。

### 生产侧

正式因子集由版本化 JSON 冻结。研究因子不会自动晋级：正式因子必须保留公式、输入、区间、行数、哈希和审批证据，并通过 `quant-factor-store` 注册和启用。

当前正式集 [`o2o_raw_daily60_minute45_dos20_v1`](../configs/formal_factor_sets/o2o_raw_daily60_minute45_dos20_v1.json) 包含：

| 组件 | 数量 | 来源 |
| --- | ---: | --- |
| 日频候选 | 60 | `candidate_factors_daily_o2o_candidate60.txt` |
| Minute OHLCV v1 | 28 | `candidate_factors_minute_ohlcv_open_to_open_abs_icir_gt2_h1_or_h5.txt` |
| Minute Core24 | 10 | `candidate_factors_minute_core24_o2o_selected10.txt` |
| Minute OHLCV v3 | 7 | `candidate_factors_minute_ohlcv_v3_o2o_abs_annual_icir_gt2_h1_or_h5.txt` |
| DOS Minute v1 | 20 | `candidate_factors_minute_dos_v1_selected20.txt` |

共125个特征。详细分钟因子公式、筛选结果和生产约束见 [`MINUTE_FACTOR_FRAMEWORK.md`](MINUTE_FACTOR_FRAMEWORK.md) 与 `research/minute_factor_sources/`。

## 3. Model / prediction

### 标签与时点

- T 日收盘后生成信号，T+1 开盘执行。
- H1 标签在 T+2 开盘退出，H5 标签在 T+6 开盘退出。
- 个股开盘到开盘收益减同期 CSI500 价格指数收益，形成超额收益标签。
- `execution_date` 必须是下一观察到的交易日，并且当天存在 CSI500 开盘基准。

### 训练协议

`quant-predict research-oos` 每季度重训 H1/H5 两个 LightGBM。训练只使用此前 756 个交易日，并与测试段留出 6 个交易日的标签成熟间隔。每个信号日的特征做截面去极值和标准化，不使用未来截面。

当前模型为125因子默认 LightGBM 回归的 H1 输出。预测和模型文件保存在本地 `results/predict/research-oos-raw-daily60-minute45-dos20-csi300-csi500-h1-h5-h10-v1/`。

## 4. Portfolio policy / backtest

### 正式规则

当前组合分别以1%和5%单票上限运行同一个五期滚动线性 optimizer，再按80%/20%资本权重合并目标、对重合股票净额化并统一执行。H1用于第1期，`(H5-H1)/4`用于第2至5期，每期均计买卖成本；每天只执行规划的第一期，次日用新预测重新求解。回测从真实成交后的现金和持仓继续下一日。

| 参数 | 值 |
| --- | ---: |
| 选股股票池 | 信号日历史 CSI500 成分 |
| 目标投资权重 | 98% |
| 袖套内单票上限 | 核心1%；增强5% |
| 合并后单票上限 | 1.8% |
| 目标持股数 | 平均106.46只；有效持股数100.09只 |
| 信号 | H1/H5 增量期限结构 |
| 风险项 | 无 |
| 换手硬限制 | 无；通过目标函数中的成本惩罚控制 |
| 整手 | 100 股 |
| 买入 / 卖出成本 | 2bp / 2bp |

回测按 T+1 开盘成交并逐日记录订单、未成交原因、成交、现金、复权等价股数、费用和 NAV。独立零费率账户只用于诊断成本拖累，不能与含费账户拼接。

正式结果为累计净收益177.94%、年化21.71%、最大回撤-27.20%、信息比率1.466。完整说明见 [`PRODUCTION_BACKTEST.md`](PRODUCTION_BACKTEST.md)。

## 模块接口

| 上游 | 下游 | 稳定交付物 |
| --- | --- | --- |
| Data | Factor | DuckDB 中的行情、日历、点时股票池和质量视图 |
| Factor research | Formal factor set | 因子 Parquet、manifest、指标与审批记录 |
| Formal factor set | Model | 版本化因子列表和宽表特征缓存 |
| Model | Portfolio policy | `pred_h1`、`pred_h5`、信号日与执行日 |
| Portfolio policy | Backtest | 订单、成交、持仓、现金和日度账本 |
| Backtest | Research decision | 含费指标、基准比较、回撤、换手和版本结论 |

## Git 与本地存储边界

应提交：源代码、测试、配置、因子定义、轻量指标表、复现脚本和文档。

不应提交：

- `A_stock_database/`：供应商数据、Parquet、DuckDB、特征缓存和潜在授权内容；
- `results/`：模型、预测、订单、持仓、账本和大体积 HTML/PDF；
- `external/`：独立第三方 Git 仓库；
- `target/`、`__pycache__/`、`.pytest_cache/`、`.DS_Store`、编辑器状态；
- 本地下载的论文、研报 PDF 和网页快照，除非许可明确且有必要随仓库发布。

HTML、PNG 等回测展示由现有 `quant-predict render-backtest-report` 从本地账本生成。`docs/reports/csi500_top100_swap3/` 只保留旧基线的历史快照；当前 optimizer 报告位于本地 `results/predict/regression-105-125-execution-2bps-v1/`，不能替代 Parquet 审计账本。

## 仍未覆盖的生产风险

这套系统已经具备研究生产化的版本、时点和账本约束，但尚不是实盘执行系统。主要缺口包括历史 ST、停复牌、涨跌停可成交性、完整公司行为、全收益指数、盘口冲击、订单容量和实时监控。所有回测结论都必须保留这一边界。
