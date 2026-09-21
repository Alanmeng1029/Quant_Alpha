# 当前生产版本回测

快照日期：2026-09-21。

当前正式基线是125因子 `o2o_raw_daily60_minute45_dos20_v1`、默认 LightGBM 回归模型的 H1/H5 预测，以及不含风险项的五期滚动成本感知 optimizer。策略分别运行单票上限1%的分散核心袖套和单票上限5%的增强袖套，再按80%/20%资本比例合并目标权重、对重合股票净额化并统一执行。两个袖套的目标函数均为：

```text
maximize sum_t alpha_t'w_t - 2bp * buys_t - 2bp * sells_t
```

期限结构定义为 `alpha_1=H1`、`alpha_2..5=(H5-H1)/4`。每天规划五期权重，只执行第一期，次日用新预测滚动重算。两个袖套均保持98%投资；合并后的理论单票上限为 `80%×1% + 20%×5% = 1.8%`，平均目标持股106.46只、有效持股100.09只。版本化配置见 [`production_strategy_csi500_h1h5_blend_80_20_2bps_v5.json`](../configs/production_strategy_csi500_h1h5_blend_80_20_2bps_v5.json)。

## 核心结果

| 指标 | 数值 |
| --- | ---: |
| 样本外持有期 | 1,311 |
| 执行区间 | 2021-04-02 至 2026-08-27 |
| 含费累计净收益 | 177.94% |
| 含费年化收益 | 21.71% |
| CSI500 累计收益 | 25.69% |
| 相对 CSI500 累计净超额 | 121.13% |
| 相对 CSI500 年化超额 | 16.48% |
| 最大回撤 | -27.20% |
| 净 Sharpe | 0.964 |
| 信息比率 / 超额 Sharpe | 1.466 |
| 平均单边买入 / 卖出换手 | 43.14% / 43.08% |
| 平均持股数 | 106.69 |
| 平均费用 | 1.724 bp / 日 |

## 近期拆分与执行方式对照

全部结果均使用原回归预测、H1 信号、点时 CSI500 成分和买卖各2bp。

| 模型 / 执行 | 全段净收益 | 全段净超额 | 2024--2026 净超额 | 2025--2026 净超额 |
| --- | ---: | ---: | ---: | ---: |
| 新125 / 80%单票1% + 20%单票5%净额组合（当前生产） | **177.94%** | **121.13%** | **21.57%** | **11.64%** |
| 新125 / H1-H5五期 optimizer / 单票1% | 167.23% | 112.60% | 17.40% | 8.00% |
| 新125 / H1单期 optimizer | 147.46% | 96.87% | 16.01% | 5.38% |
| 新125 / H1-H5-H10十期 optimizer | 151.81% | 100.34% | 10.33% | 0.89% |
| 新125 / 无换手控制 Top100 | 134.57% | 86.62% | 14.67% | 3.31% |
| 新125 / 每日最多替换3只 | 85.85% | 47.86% | -4.94% | -13.29% |
| 旧105 / optimizer | 110.80% | 67.71% | 5.67% | -4.78% |
| 旧105 / 无换手控制 Top100 | 120.34% | 75.30% | 8.01% | 1.95% |
| 旧105 / 每日最多替换3只 | 111.01% | 67.88% | 4.36% | -6.82% |

进一步比较显示，80%/20%净额组合的全段信息比率1.466，高于单票1%版本的1.432和单票5%版本的1.119；同时年化收益由单票1%的20.80%提高至21.71%，最大回撤由-27.32%小幅改善为-27.20%。因此按“超额收益的 Sharpe”为主要目标升级为生产基线。该提升来自同一账户先合并目标再下单，不等于两条既有净值曲线的事后加权。

## 固定口径

- 因子集：60个 raw 日频 + 45个既有分钟 + 20个新 DOS 分钟因子，共125个。
- 模型：默认 LightGBM 回归；CSI300 ∪ CSI500 训练；756交易日窗口、11日标签隔离、季度滚动重训。
- 信号：H1与H5构成不重叠增量期限结构；H1用于第1期，`(H5-H1)/4`用于第2至5期。
- 交易池：信号日历史 CSI500 成分，来自 `index_trading_universe`。
- optimizer：两个袖套分别每日规划五期、逐期计交易成本，只输出第一期目标；无协方差或其他风险项。
- 组合：1%袖套占80%、5%袖套占20%；目标权重先净额合并后统一执行；合并单票上限1.8%，不存在“每日最多换几只”的硬限制。
- 执行：T日收盘后产生信号，T+1开盘按100股整手交易，真实现金和持仓滚动。
- 成本：买入2bp、卖出2bp。
- 基准：CSI500价格指数，不是全收益指数。

## 结果来源与复现

从已生成的125因子回归预测复现实验：

```bash
PYTHONPATH=src python scripts/optimize_multiperiod_mu_turnover.py \
  --predictions results/predict/research-oos-raw-daily60-minute45-dos20-csi300-csi500-h1-h5-h10-v1/predictions.parquet \
  --output results/predict/production-max1 --max-weight 0.01

PYTHONPATH=src python scripts/optimize_multiperiod_mu_turnover.py \
  --predictions results/predict/research-oos-raw-daily60-minute45-dos20-csi300-csi500-h1-h5-h10-v1/predictions.parquet \
  --output results/predict/production-max5 --max-weight 0.05

PYTHONPATH=src python scripts/blend_target_weights.py \
  --target-weights results/predict/production-max1/target_weights.parquet --allocation 0.8 \
  --target-weights results/predict/production-max5/target_weights.parquet --allocation 0.2 \
  --output results/predict/production-blend-80-20
```

本地审计证据位于 `results/predict/no-risk-multiperiod-blend-80pct-max1-20pct-max5/`。`results/` 被 Git 忽略，因此模型、预测、目标权重、订单和 Parquet 账本不上传；仓库提交配置、代码和本文中的轻量指标快照。

## 生产风险

- 平均单边换手约43%，远高于旧 Swap3 基线；2bp是研究假设，不包含完整冲击成本与容量退化。
- optimizer 没有风险模型、行业约束或风格约束，可能形成不可见的集中风险。
- 2021-04 至 2026-08 已参与多轮模型与执行方式比较，存在选择偏差。
- 当前缺少完整历史 ST、停复牌、涨跌停成交概率、全收益指数和盘口级成交模拟。
- 正式监控必须同时记录预测尺度、实际换手、成交偏离和费用；不能只监控净值。

这些结果只说明当前历史样本和数据口径下的表现，不构成收益承诺。
