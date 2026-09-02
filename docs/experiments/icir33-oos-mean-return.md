# ICIR33 候选因子：完整 OOS 预测与优化回测

日期：2026-09-02。该实验将单因子批评估中有限 `|close_h1_rank_icir| > 3` 的 33 个因子作为候选特征。候选清单是 [`configs/candidate_factors_abs_icir_gt3.txt`](../../configs/candidate_factors_abs_icir_gt3.txt)。

## 协议

- 股票池：每日 CSI300 ∪ CSI500 的可交易交集。
- 特征：33 个候选因子的日频原始值，按信号日横截面 1%/99% 去极值并做 Z-score。
- 标签：`(excess_h1 + excess_h5) / 2`，其中收益均以 T+1 VWAP 进入，且相对当日等权截面收益计算超额收益。
- 样本外：每月滚动训练；每次仅用此前 756 个交易日，并保留 6 个交易日标签空档。OOS 为 2021-04-01 至 2026-08-28。
- 模型：一个 LightGBM 回归模型（100 棵树、31 个叶子、学习率 0.1、固定种子 20260831）。
- 优化：全动态股票池进入 OSQP，没有预设 50 只或其它持仓数量。参数为单票上限 15%、行业偏离 10%、单边换手上限 50%、交易成本 10bp。

## 预测结果

完整预测产物在 [`results/predict/icir33-lgbm-mean-h1-h5-v1`](../../results/predict/icir33-lgbm-mean-h1-h5-v1)。

| 指标 | 数值 |
| --- | ---: |
| 平均标签 Rank IC | 0.00412 |
| 标签 ICIR | 0.0897 |
| 1 日超额收益 Rank IC | 0.00419 |
| 5 日超额收益 Rank IC | 0.00449 |

单因子筛选后的 33 个特征在多因子、严格 OOS 的模型中没有形成强预测信号。

## 组合回测结果

| 指标 | 数值 |
| --- | ---: |
| 回测日数 | 1,311 |
| 毛总收益 | 45.3% |
| 10bp 成本后净总收益 | -33.3% |
| 动态等权基准总收益 | 33.5% |
| 年化收益 | -7.49% |
| Sharpe | -0.30 |
| 信息比率 | -0.68 |
| 最大回撤 | -63.0% |
| 平均单边换手 | 29.7% |
| 累计交易成本 | 77.8% |
| 每期持仓数（最小 / 中位 / 最大） | 15 / 317 / 800 |

结果文件：[`portfolio_summary.json`](../../results/predict/icir33-lgbm-mean-h1-h5-v1/optimizer/portfolio_summary.json)、[`portfolio_daily.parquet`](../../results/predict/icir33-lgbm-mean-h1-h5-v1/optimizer/portfolio_daily.parquet)、[`target_weights.parquet`](../../results/predict/icir33-lgbm-mean-h1-h5-v1/optimizer/target_weights.parquet)。

## 求解器审计与结论

OSQP 在每个交易日达到 1,000 次迭代上限后给出近似解；代码保留该迭代，而没有退回启发式组合。所有行的 `solver_status` 都是 `max_iterations`，无 `heuristic_fallback`。实际观测到的最大单票权重为 15.07%、最大行业偏离为 10.31%、最大单边换手为 56.29%，略高于目标约束，因此本次应视为近似优化实验，不是严格可行的生产组合。

本实验不支持进入正式因子库或实盘。下一步应先做两件事：

1. 在训练期内完成候选因子去重、相关性控制和滚动特征筛选，避免直接把全样本单因子 ICIR 当作特征选择依据。
2. 调整或替换优化器的数值方案，使约束严格收敛；随后再做成本敏感性与持仓约束实验。
