# Quant Alpha 实验汇总与当前基线

更新时间：2026-09-22。

本文是当前研究结论的入口。具体实现、逐年/逐季度指标和复现命令仍以链接的专题文档为准；出现口径冲突时，以 [`PRODUCTION_BACKTEST.md`](PRODUCTION_BACKTEST.md) 的正式基线定义为准。

## 当前正式基线

当前正式方案是 **125因子默认 LightGBM + 无风险项五期滚动 optimizer + 80%/20%双袖套净额组合**。

| 组件 | 当前定义 |
|---|---|
| 因子 | 60个 raw 日频 + 45个既有分钟 + 20个 DOS 分钟，共125个 |
| 模型 | 默认 LightGBM 回归，H1/H5/H10分别训练 |
| 训练池 | CSI300 ∪ CSI500 |
| 滚动训练 | 756交易日等权窗口，11交易日标签隔离，季度重训 |
| 标签 | T日信号；T+1开盘进入；T+2/T+6/T+11开盘退出；减CSI500同期收益 |
| optimizer | H1/H5五期 receding horizon；每期显式扣买卖各2bp；无协方差风险项 |
| 组合 | 80%单票上限1%的CSI500核心袖套 + 20%单票上限5%的增强袖套 |
| 投资与执行 | 98%目标投资；袖套先合并净额；T+1开盘、100股整手执行 |
| 理论单票上限 | 1.8% |

正式样本外区间为2021-04-02至2026-08-27，共1,311个持有期：

| 指标 | 当前基线 |
|---|---:|
| 含费累计净收益 | 177.94% |
| 含费年化收益 | 21.71% |
| CSI500累计收益 | 25.69% |
| 相对CSI500累计净超额 | 121.13% |
| 相对CSI500年化超额 | 16.48% |
| 最大回撤 | -27.20% |
| 净Sharpe | 0.964 |
| 信息比率 | 1.466 |
| 平均单边买入/卖出换手 | 43.14% / 43.08% |
| 平均持股数 | 106.69 |
| 平均费用 | 1.724 bp/日 |

正式配置见 `configs/production_strategy_csi500_h1h5_blend_80_20_2bps_v5.json`，本地审计产物位于 `results/predict/no-risk-multiperiod-blend-80pct-max1-20pct-max5/`。`results/` 不进入Git。

## 已完成实验与决策

| 方向 | 关键结果 | 决策 |
|---|---|---|
| 98因子默认LGBM vs 年度Rank-IC调参 | 调参版IC略高但不显著；Optimizer V2净收益57.90%，低于默认版71.38% | 保留默认LGBM；该结论延续至后续因子版本 |
| 日频因子稳定性筛选 | Top20和无泄漏119因子方案净收益仅23.27%/18.44%；均明显不足 | 不晋级，删除大型产物 |
| 全272因子Ridge/Elastic Net | 净收益35.52%/50.78%；Elastic Net存在242日无有效排序 | 不晋级，保留通用实现 |
| 新raw Daily60 | 新Top60替换17个旧因子，在当时Top100/Swap3口径下优于旧Daily60 | 晋级为后续正式日频60因子 |
| 新Daily60 + Minute45 | H1-only净收益100.91%，优于旧组合；H1/H5等权会稀释增益 | 形成105因子阶段主要候选，随后继续加入DOS因子 |
| 125因子（增加20个DOS分钟因子） | 配合五期optimizer和80/20净额组合，含费累计净收益177.94%、IR 1.466 | **当前正式基线** |
| H1/H5五期 optimizer | 单票1%版本净收益167.23%，优于H1单期147.46%和H1/H5/H10十期151.81% | 五期H1/H5期限结构晋级 |
| 80%/20%双袖套 | 年化21.71%、IR 1.466，高于单独1%袖套的20.80%和1.432 | 净额组合晋级 |
| 风险项 optimizer | 温和风险项版本净收益169.46%，低于无风险基线177.94%；其他更集中设置增加换手与回撤 | 当前不加入风险项 |
| 125因子残差LSTM | H1/H5 IC 0.029379/0.035793；年化17.76%，低于共同样本LGBM的21.16% | 保留研究候选，不替换LGBM |
| LSTM行业+流动性embedding | 完整滚动H1 +0.000461，但H5 -0.002776；单季度正结果无法推广 | 不晋级，完整产物已删除 |
| LGBM月度重训 | H1/H5/H10 IC均下降；净收益142.99%，低于季度重训177.94% | 不晋级 |
| LGBM 504日半衰期 | H5下降0.004886；净收益132.13%，换手上升6.46pp | 不晋级 |
| 排名倾斜开仓V3 | 净收益68.32%、IR 0.464，低于当时V2的73.27%和0.532 | 不晋级 |

## 当前模型判断

默认LGBM仍是正式预测模型。残差LSTM在H5上与LGBM接近，部分季度明显领先，但全期H1显著落后，且没有转化为更好的含费组合收益。加入行业和流动性embedding后，H1只有很小改善，H5反而下降，说明增加静态实体信息不能稳定解决当前问题。

提高LGBM重训频率和引入时间衰减也没有改善泛化。月度重训改变了大量选股却降低H10并增加换手；504日半衰期降低H5质量。这些结果说明当前主要约束并非模型“更新不够快”。

现阶段模型侧不应继续盲目增加embedding、缩短重训周期或扫描更多衰减参数。若继续改进，优先要求：

1. 新信息必须在多个不相邻季度和完整滚动区间同时验证。
2. 预测评估必须覆盖H1、H5及其组合实现，不能只看单一季度或验证MSE。
3. 新模型必须使用相同股票—日期键、相同标签和相同optimizer进行含费比较。
4. 若训练目标改为排序或IC相关目标，应先做严格配对pilot，再决定是否跑完整22窗口。

## 当前 optimizer 判断

当前无风险项五期optimizer仍是最强正式版本。它利用H1和H5构造不重叠的五期预期收益路径，并在规划时显式计入每次买卖成本。80%/20%袖套组合在提高收益的同时没有恶化最大回撤，因此优于单一持仓上限。

已有风险项实验没有证明协方差惩罚能改善当前目标；部分风险设置导致更集中持仓、更高换手和更大回撤。后续若重启风险模型，应先明确风险暴露、行业约束和组合基准，并使用真正的点时风险输入，而不是只增加一个惩罚系数。

## 仍需警惕的限制

- 平均单边换手约43%，2bp成本假设没有覆盖完整冲击成本和容量退化。
- 当前optimizer没有行业、风格或协方差风险约束。
- CSI500使用价格指数作为基准，并非全收益指数。
- 历史ST、停复牌、涨跌停成交概率和盘口级成交模拟仍不完整。
- 2021--2026样本已用于多轮方案比较，存在研究选择偏差。
- LSTM embedding实验使用静态行业字段，不是历史时点行业。

## 专题文档

- 当前正式回测：[`PRODUCTION_BACKTEST.md`](PRODUCTION_BACKTEST.md)
- 125因子残差LSTM：[`RESIDUAL_LSTM_RAW125.md`](RESIDUAL_LSTM_RAW125.md)
- LGBM重训频率与时间衰减：[`experiments/lgbm-refit-frequency-and-time-decay-20260922.md`](experiments/lgbm-refit-frequency-and-time-decay-20260922.md)
- LSTM embedding失败实验：[`research/failed-lstm-industry-liquidity-embedding-20260922.md`](research/failed-lstm-industry-liquidity-embedding-20260922.md)
- 日频筛选与线性模型失败实验：[`research/failed-factor-selection-and-linear-models-20260919.md`](research/failed-factor-selection-and-linear-models-20260919.md)
- 新Daily60 + Minute45：[`research/raw-new-daily60-minute45-20260919.md`](research/raw-new-daily60-minute45-20260919.md)
- 98因子LGBM历史实验：[`experiments/prediction-lgbm-rolling-oos-v1.md`](experiments/prediction-lgbm-rolling-oos-v1.md)
- Optimizer V2历史实验：[`experiments/optimizer-v2-limited-replacement.md`](experiments/optimizer-v2-limited-replacement.md)
