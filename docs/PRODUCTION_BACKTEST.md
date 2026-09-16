# 当前生产版本回测

快照日期：2026-09-16。

当前正式版本是 105 因子 `o2o_daily60_minute45_v2`、默认 LightGBM，并且只在信号日的历史中证500成分内选股。组合采用 `limited_replacement_v2`：Top100、退出 Top120、每日最多主动替换 3 只。版本化配置见 [`production_strategy_csi500_top100_v1.json`](../configs/production_strategy_csi500_top100_v1.json)。[标准 HTML 报告](reports/csi500_top100_swap3/report.html)由项目现有报告器直接生成。

## 核心结果

| 指标 | 数值 |
| --- | ---: |
| 样本外持有期 | 1,311 |
| 执行区间 | 2021-04-02 至 2026-08-27 |
| 含费累计净收益 | 106.14% |
| 含费年化收益 | 14.92% |
| CSI500 累计收益 | 25.69% |
| CSI500 年化收益 | 4.49% |
| 相对 CSI500 年化超额 | 9.98% |
| 最大回撤 | -23.36% |
| 净 Sharpe | 0.728 |
| 信息比率 | 0.974 |
| 年化跟踪误差 | 9.13% |
| 平均主动收益 | 3.53 bp / 日 |
| 平均单边买入 / 卖出换手 | 2.98% / 2.92% |
| 平均持股数 | 99.85 |
| 平均现金权重 | 5.63% |
| 平均费用 | 0.270 bp / 日 |

零费率独立账户的累计收益为 113.02%，只用于衡量交易成本影响；正式结果始终引用含费账户的 106.14%。

## 年度拆分

2021 和 2026 都是不完整自然年。

| 年份 | 持有期 | 策略净收益 | CSI500 | 平均买入换手 |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 184 | 27.80% | 16.96% | 3.40% |
| 2022 | 242 | -1.70% | -20.66% | 2.91% |
| 2023 | 242 | 6.60% | -7.34% | 2.92% |
| 2024 | 242 | 12.71% | 5.27% | 2.93% |
| 2025 | 243 | 19.17% | 31.58% | 2.92% |
| 2026 | 158 | 14.61% | 5.52% | 2.89% |

这不是“每年都跑赢”的策略：2025 年落后 CSI500，2022 年策略自身也录得负收益。全段最大回撤约 23%，不能只看累计收益。

## 固定口径

- 因子集：60 日频 + 28 Minute v1 + 10 Core24 + 7 Minute v3，共 105 个。
- 模型训练池：CSI300 ∪ CSI500；H1/H5 两个默认 LightGBM，季度滚动重训。
- 正式交易池：信号日历史 CSI500 成分，来自 `index_trading_universe`，不是当前成分回填。
- 排名：在当日 CSI500 交易池内重新标准化 H1/H5 分数，各占 50%。
- 信号：T 日收盘后生成，T+1 开盘执行。
- 组合：Top100 入选、跌出 Top120 才正常退出，每日最多主动替换 3 只。
- 账户：初始 1,000 万元，真实现金与实际成交持仓滚动，100 股整手。
- 成本：买入 2.1bp、卖出 7.1bp。
- 基准：CSI500 价格指数，不是全收益指数。

## 结果来源与复现

完整复现分为预测和正式组合两步：

```bash
PYTHONPATH=src python -m a_share_data.predict research-oos \
  --config configs/prediction_research_105_v2.json
PYTHONPATH=src python scripts/run_universe_size_study.py
```

从已有正式账本生成标准 HTML/PNG 报告：

```bash
PYTHONPATH=src python -m a_share_data.predict render-backtest-report \
  --portfolio-daily results/predict/universe-size-study/csi500_top100_swap3/portfolio_daily.parquet \
  --output results/predict/universe-size-study/csi500_top100_swap3/report \
  --title "105-factor LGBM — CSI500 Top100 / max 3 replacements — charged"
```

本地审计源位于：

```text
results/predict/universe-size-study/
├── csi500_predictions.parquet
└── csi500_top100_swap3/
    ├── manifest.json
    ├── orders.parquet
    ├── executions.parquet
    ├── holdings.parquet
    ├── portfolio_daily.parquet
    ├── annual_metrics.parquet
    ├── report/report.html
    └── zero_cost/
```

`results/` 被 Git 忽略，不上传模型、个股预测、订单、持仓明细或 Parquet 账本。仅将标准报告器生成的 HTML、三张 PNG 和汇总 JSON 复制到 `docs/reports/csi500_top100_swap3/` 作为远端轻量快照；它们不能替代本地 Parquet 审计证据。

## 解读限制

- 当前日线由分钟数据聚合，复权质量仍需继续审计。
- 没有完整模拟历史 ST、停复牌、涨跌停成交概率、盘口冲击和容量。
- 模型仍在 CSI300 ∪ CSI500 上训练；正式组合只在 CSI500 内重新标准化和选股，并非 CSI500 专属重训模型。
- 2021-04 至 2026-08 已参与多轮因子和组合研究，存在选择偏差。
- 结果只说明该历史样本和当前数据口径下的表现，不构成收益承诺或投资建议。
