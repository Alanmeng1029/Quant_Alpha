# 股票多因子优化器：通用方法与 Quant_Alpha 建议

日期：2026-09-17。

## 结论

股票多因子领域最常见、也最适合当前项目的方案，是将预测和组合构造分离：模型只产生股票预期收益或 alpha，风险模型估计股票协方差，成本模型描述换仓摩擦，最后通过带约束的凸优化得到目标权重。实际账户、整手交易、停牌和缓冲区规则放在执行层处理。

第一版不应使用端到端神经网络直接输出权重，也不应使用未经收缩的样本协方差。当前样本期和模型数量不足以稳定估计大量自由参数。

## 通用架构

### 1. Alpha 层

对每个交易日产生股票 alpha 向量 `a_t`。因子或模型输出先做截面去极值、标准化和必要的行业/市值中性化，再进行组合。常见方法包括：

- 等权 z-score；
- 按滚动 IC/ICIR 加权，但向等权强收缩；
- ridge/elastic-net 或简单 stacking，训练输入必须是基础模型的历史 OOS 预测；
- 先整合股票层信号再构建一个组合，而不是分别构建多个组合后进行资金混合。

当前项目可继续使用 LGBM/LSTM 的每日截面 z-score 50/50 融合。随后利用过去 OOS 中的截面回归斜率，将无量纲分数校准成预期收益；也可以保留无量纲 alpha，并通过风险厌恶参数控制整体强度。

### 2. 风险层

使用因子风险模型：

`Sigma_t = X_t F_t X_t' + D_t`

- `X_t`：行业及风格暴露，如规模、beta、波动率、流动性、动量；
- `F_t`：因子收益协方差；
- `D_t`：个股特异风险。

因子协方差和特异风险应做 EWMA 或 shrinkage。约 500 只股票而历史窗口约 252 日时，原始样本协方差不稳定甚至奇异。Ledoit-Wolf 的研究直接针对这一估计误差；行业实践也主要通过 Barra 类因子风险模型管理非预期暴露。

### 3. 成本与优化层

建议的单期目标函数为：

```text
maximize    a_t' w
          - lambda_risk * (w - b)' Sigma_t (w - b)
          - lambda_linear * sum_i c_i * |w_i - w_prev_i|
          - lambda_impact * sum_i q_i * (w_i - w_prev_i)^2
          - lambda_anchor * ||w - w_rank||^2
```

其中：

- `w` 是目标权重，`b` 是 CSI500 基准权重；
- 第一项购买 alpha；
- 第二项控制跟踪误差和集中风险；
- 第三、第四项分别表示线性交易成本和非线性冲击；
- 第五项可选，用于向简单排名权重收缩，降低参数误差。

典型约束：

```text
sum(w) = 0.98                 # 2% 现金
0 <= w_i <= 0.03             # long-only 与单票上限
turnover(w, w_prev) <= cap    # 换手上限
sector_active within bounds   # 行业主动偏离
style_active within bounds    # beta/规模/波动/流动性偏离
trade_i <= ADV_i * limit      # 容量约束
```

持仓数量不必硬编码。可以先限制候选集，再将非常小的目标权重删除并重新求解。硬性持仓数属于 cardinality 约束，通常会把凸问题变成混合整数问题，第一版没有必要。

### 4. 执行层

优化器输出目标权重后，执行层继续使用当前项目已具备的能力：真实现金和股数账本、100 股整手、不可交易处理、成分退池、单票上限、订单和成交审计。为减少微小交易，可加入 no-trade band：目标权重与实际权重之差小于阈值时不交易。

## 方法比较

| 方法 | 优点 | 主要问题 | 用途 |
|---|---|---|---|
| TopN 等权 | 稳定、透明、估计误差小 | 不利用分数强弱和风险差异 | 必须保留的基准 |
| 排名倾斜 | 简单利用 alpha 强度 | 倾斜过强会集中和减少持仓 | 当前最容易落地的增强 |
| 资金分仓 | 模型独立、风险分散 | 无法在股票层净额交易 | 风险基准 |
| 信号整合 | 可净额交易，充分利用低相关信号 | 需要稳健的信号尺度 | 推荐 alpha 输入 |
| 最小方差/风险平价 | 不依赖预期收益 | 可能忽略有效 alpha | 风险基准 |
| 约束均值方差/QP | 同时处理 alpha、风险、成本和约束 | 输入估计和参数选择敏感 | 推荐主优化器 |
| Black-Litterman/贝叶斯收缩 | 稳定预期收益 | 需要合理先验 | 后续增强 |
| 端到端 Portfolio-ML | 可直接优化净效用 | 数据需求高、验证复杂 | 研究后期 |

## 与当前 Quant_Alpha 的差距

`LimitedReplacementConfig` 已经解决真实账户、排名缓冲、换手数量、现金、单票权重、成本和整手问题，但没有风险模型，也没有在统一目标函数内权衡 alpha、跟踪误差和成本。

旧 `optimize_dual_alpha_targets` 使用 softmax/等权生成目标，再按换手上限从旧目标向新目标做线性投影。它没有使用真实账户状态、协方差、行业/风格暴露或流动性冲击，因此不应作为最终优化器。

近期排名倾斜实验说明组合构造确实重要：日频 10% 倾斜改善了收益和 IR；但它仍是单参数启发式规则，而且参数是在完整研究 OOS 上观察后选出，需要滚动验证。

## 推荐实施顺序

1. 保留等权、10% 排名倾斜和当前有限换仓作为三条基准。
2. 建立 CSI500 风险模型与每日风险快照，先实现行业、beta、规模、波动率、流动性暴露。
3. 实现单期凸优化：alpha、主动风险、线性成本、换手上限、long-only、3% 单票上限。
4. 使用当前真实账户状态 `w_prev`，不能使用昨日理论目标替代实际持仓。
5. 加入 no-trade band 和整手后约束复核。
6. 只用过去窗口选择少量 `lambda_risk`、`lambda_cost`；季度冻结参数，下一季度 OOS 使用。
7. 记录预测跟踪误差与实现跟踪误差、预测成本与实现成本、约束影子价格和 alpha transfer coefficient。
8. 风险模型和单期优化稳定后，再研究多期优化或端到端 Portfolio-ML。

## 验收对照

统一使用同一预测、股票池和费用，对比：

- Top100 等权、每日最多换 3 只；
- Top100、10% 排名倾斜；
- 约束 QP，不设硬持仓数；
- QP 的零成本版本；
- 去掉风险项、去掉成本项、去掉行业约束的消融版本。

除收益、IR和回撤外，还应报告跟踪误差、行业和风格最大偏离、单票集中度、有效持股数、换手、容量、成本预测误差以及约束导致的 alpha 损失。

## 主要资料

- Boyd et al., *Multi-Period Trading via Convex Optimization*: https://web.stanford.edu/~boyd/papers/cvx_portfolio.html
- Ledoit and Wolf, *Honey, I Shrunk the Sample Covariance Matrix*: https://ledoit.net/honey.pdf
- Jagannathan and Ma, *Risk Reduction in Large Portfolios*: https://onlinelibrary.wiley.com/doi/10.1111/1540-6261.00580
- DeMiguel et al., *A Transaction-Cost Perspective on the Multitude of Firm Characteristics*: https://academic.oup.com/rfs/article-abstract/33/5/2180/5821387
- Kelly et al., *Machine Learning and the Implementable Efficient Frontier*: https://academic.oup.com/rfs/advance-article/doi/10.1093/rfs/hhag022/8524346
- Daniel et al., *The Cross-Section of Risk and Returns*: https://academic.oup.com/rfs/article/33/5/1927/5803086
- Grinold, *The Fundamental Law of Active Management*: https://doi.org/10.3905/jpm.1989.409211
- Clarke, de Silva and Thorley, *Portfolio Constraints and the Fundamental Law*: https://people.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf
- FTSE Russell, *Comprehensive Factor Methodology Overview*: https://research.ftserussell.com/products/downloads/Comprehensive_Factor_Methodology_Overview.pdf
- MSCI, *Factor Advanced Indexes Methodology*: https://www.msci.com/indexes/documents/methodology/3_MSCI_Factor_Advanced_Indexes_Methodology_20250203.pdf
