# Quant_Alpha

面向 A 股的本地多频 Alpha 研究系统：把分钟行情整理为可审计的数据湖，生产日频与分钟衍生因子，做无泄漏滚动样本外预测，并在真实现金、整手和交易成本约束下回测组合。

> 本仓库是研究与生产化计算框架，不是实盘交易系统。行情、模型和大体积回测账本只保存在本机；Git 只保存代码、配置、可复现说明和轻量结果快照。

## 当前正式研究版本

截至 2026-09-21，当前正式因子集是 [`o2o_raw_daily60_minute45_dos20_v1`](configs/formal_factor_sets/o2o_raw_daily60_minute45_dos20_v1.json)：

- 60 个 raw 日频因子；
- 45 个既有分钟因子；
- 20 个 DolphinDB 复现分钟因子；
- 默认 LightGBM，分别预测 H1/H5 开盘到开盘的相对 CSI500 超额收益；
- 历史中证500成分内运行五期成本感知 optimizer；80%单票1%核心袖套与20%单票5%增强袖套先合并目标、再净额执行。

旧因子集和旧执行配置保留用于复现和版本比较，不再代表当前正式版本。

## 正式回测快照

样本外区间为 2021-04-02 至 2026-08-27，共 1,311 个持有期；买入、卖出成本各2bp。

| 指标 | 正式策略 | CSI500 |
| --- | ---: | ---: |
| 累计收益 | 177.94% | 25.69% |
| 年化收益 | 21.71% | 4.49% |
| 最大回撤 | -27.20% | — |
| 信息比率 / 超额 Sharpe | 1.466 | — |
| 平均单边买入换手 | 43.14% / 日 | — |
| 平均持股数 | 106.69 | — |

完整指标、年度拆分和生成方式见 [当前生产版本回测](docs/PRODUCTION_BACKTEST.md)。大体积标准报告和逐日账本保留在本地 `results/`；上述结果是历史研究模拟，不代表未来收益。

## 系统边界

```text
本地供应商数据
    ↓
Data lake / DuckDB catalog
    ↓
Factor research → approved factor set
    ↓
Rolling OOS LightGBM H1/H5 predictions
    ↓
Five-period 1% / 5% sleeves → 80% / 20% netted target blend
    ↓
Charged backtest + audit artifacts
```

四个模块的职责、输入输出和当前产物见 [架构与产物](docs/ARCHITECTURE_AND_ARTIFACTS.md)。数据覆盖与已知缺口见 [数据状态](DATA_STATUS.md)。

## 环境与验证

现有开发环境位于 `ml311`：

```bash
export PYTHONPATH=src
PYTHON=/Users/alanmxy/anaconda3/envs/ml311/bin/python

$PYTHON -m a_share_data --help
$PYTHON -m a_share_data.predict --help
$PYTHON -m pytest -q tests
cargo test --workspace
```

如果尚未安装项目，也可在 Python 3.11 环境中执行：

```bash
python -m pip install -e .
```

## 典型工作流

### 1. 建设和校验数据湖

```bash
PYTHONPATH=src python -m a_share_data inventory
PYTHONPATH=src python -m a_share_data backfill-minute
PYTHONPATH=src python -m a_share_data build-daily
PYTHONPATH=src python -m a_share_data build-universe
PYTHONPATH=src python -m a_share_data validate
PYTHONPATH=src python -m a_share_data build-catalog
```

原始 CSV 和供应商复权文件永不原地修改。增量更新、指数成分和复权快照命令见 [数据状态](DATA_STATUS.md)。

### 2. 生产和评估因子

```bash
cargo run --release -p quant-minute-factor -- --help
cargo run --release -p quant-backtest -- --help
PYTHONPATH=src python -m a_share_data.factors --help
PYTHONPATH=src python -m a_share_data.factor_store --help
```

研究因子不会自动进入正式库。正式因子必须带来源 manifest，经人工批准、注册和启用后才能写入生产因子库。

### 3. 复现正式滚动 OOS 与组合回测

```bash
PYTHONPATH=src python -m a_share_data.predict research-oos \
  --config configs/prediction_research_105_v2.json
PYTHONPATH=src python scripts/run_universe_size_study.py
```

第一条命令使用固定的 105 因子配置重新训练季度滚动 LightGBM；第二条命令按历史中证500成分过滤预测并运行 Top100/换仓3正式组合。两者依赖本机数据湖和特征缓存，运行前应核对配置中的绝对路径。

### 4. Rust 缓存与 MPS LSTM 研究

LSTM 使用过去 20 个连续交易日的 105 因子序列。Python 负责沿用既有截面标准化和标签口径，`quant-sequence-cache` 将面板转换为紧凑只读 mmap；PyTorch 在独立进程中通过 MPS 训练，禁止静默回退 CPU。

```bash
# 只生产并校验紧凑缓存
PYTHONPATH=src python -m a_share_data.predict sequence-oos \
  --config configs/prediction_sequence_lstm_105_v1.json --prepare-only

# 一个季度、两轮训练的资源与正确性试跑
PYTHONPATH=src python -m a_share_data.predict sequence-oos \
  --config configs/prediction_sequence_lstm_105_v1.json --pilot

# 完整季度滚动 OOS、共同样本 LightGBM 对照和正式 Top100/换仓3回测
PYTHONPATH=src python -m a_share_data.predict sequence-oos \
  --config configs/prediction_sequence_lstm_105_v1.json
```

每个季度先用 687 日拟合、6 日隔离、63 日验证选 epoch，再以该 epoch 数在完整 756 日窗口重新训练。缓存位于配置输出目录的 `sequence_cache/`，试跑与正式结果分别位于 `pilot/` 和 `full/`；模型、窗口审计、预测、CSV 指标和图表都保存在对应目录。

## 仓库地图

| 路径 | 内容 | 是否进入 Git |
| --- | --- | --- |
| `src/a_share_data/` | 数据、因子、预测、策略与报告 Python 实现 | 是 |
| `crates/` | Rust 分钟因子生产与批量回测引擎 | 是 |
| `configs/` | 因子集、实验和回测配置 | 是 |
| `docs/` | 架构、实验记录和轻量结果快照 | 是 |
| `research/` | 研究结论、公式索引与小型指标表 | 是 |
| `scripts/` | 可复现实验和报告生成脚本 | 是 |
| `A_stock_database/` | 原始数据、Parquet、DuckDB、特征缓存 | 否 |
| `results/` | 模型、预测、订单、持仓、账本、HTML 报告 | 否 |
| `external/` | 独立克隆的第三方参考仓库 | 否 |
| `target/`、缓存、编辑器状态 | 构建与本机临时文件 | 否 |

## 研究限制

- 日线由分钟行情聚合，不是交易所官方日线。
- 当前缺少完整历史 ST、停复牌、涨跌停、上市退市和全收益指数数据。
- 前复权质量仍需系统审计；异常代码通过显式黑名单处理，但这不是完整解决方案。
- 回测未建模盘口冲击、订单容量和成交概率。
- 2021-04 至 2026-08 已用于多轮研究比较，存在研究选择偏差；正式升级应锁定规则并等待新样本验证。

更细的策略实验记录见 [Optimizer 实验日志](research/OPTIMIZER_EXPERIMENT_LOG.md)。
