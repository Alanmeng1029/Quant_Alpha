# CSI500 因子风险模型设计

日期：2026-09-17。

## 目标与边界

该模型用于预测股票组合未来短期协方差、主动风险和风险贡献，不用于产生 alpha。第一版服务于每日收盘决策、下一交易日开盘执行的 CSI500 long-only 组合。

风险模型形式：

```text
r_(t+1) = X_t f_(t+1) + u_(t+1)
Sigma_t = X_t F_t X_t' + D_t
```

`X_t` 是 T 日已知的股票风险暴露，`f_(t+1)` 是下一持有期实现的因子收益，`u_(t+1)` 是特异收益，`F_t` 是因子收益协方差，`D_t` 是特异方差对角阵。

## 当前数据可用性

- `daily_qfq`：2018 年以来日频复权价格、收益、成交额和成交量；
- `index_trading_universe`：历史 CSI300/CSI500 成分；
- `index_daily`：CSI500 指数收益；
- `risk_inputs/csi500_akshare_monthly_float_mv`：公告日约束的流通股本、总股本和每日流通市值，覆盖率 99.49%；
- `instruments.industry`：存在行业字段，但当前表没有历史生效日期，正式 OOS 使用前必须确认或补充点时行业历史。

估计股票池建议使用历史 CSI300∪CSI500，可提高行业和风格因子收益的截面稳定性；优化股票池仍为当日 CSI500。

## 第一版风险因子

### 分类因子

- 市场截距；
- 点时行业 one-hot。

市场截距与全部行业虚拟变量共线，需要使用市值加权行业收益和为零的约束，或者删除一个基准行业。推荐显式约束，避免基准行业变化影响结果。

### 风格因子

第一版只使用当前数据能够可靠构造的因子：

1. `size`：`log(float_mv)`；
2. `beta`：股票相对 CSI500 的滚动 EWLS beta；
3. `residual_volatility`：市场模型残差的滚动波动率；
4. `liquidity`：20/60 日成交额或换手率的对数均值；
5. `momentum`：跳过最近约 1 个月的中期累计收益；
6. `short_term_reversal`：最近约 20 日收益；
7. `nonlinear_size`：size 的非线性分量，并对线性 size 正交化。

不要把 105 个 alpha 因子全部放进风险模型。风险因子需要解释共同波动且保持低维稳定；将 alpha 因子全部风险中性化会消除希望持有的预测暴露。

基本面质量、价值、成长、杠杆和中国市场的国企因子放到第二版，前提是获得严格点时的财务与所有权数据。MSCI 的中国模型也特别强调点时基本面、国企因子、动态行业暴露和更完整的估计股票池。

## 暴露处理

每个交易日对连续风格暴露执行：

1. 只使用 T 日收盘时已知的数据；
2. 极值裁剪，优先使用 median/MAD；
3. 缺失值使用行业中位数，行业样本不足时使用全市场中位数；
4. 截面标准化为均值 0、标准差 1；
5. 对明显冗余因子做加权正交化，例如 residual volatility 对 size 和 beta 正交化；
6. 保存原值、处理值、缺失标记和处理参数。

标准化权重可以先用流通市值平方根，使大股票对风险估计更重要，同时避免最大市值股票完全控制截面。

## 每日因子收益

在每个实现日做加权截面回归：

```text
min_f sum_i omega_(i,t) * (r_(i,t+1) - X_(i,t) f_(t+1))^2
```

- `omega` 使用经过上限处理的流通市值平方根；
- `X_(i,t)` 必须来自收益发生前；
- `r_(i,t+1)` 使用与策略一致的 open-to-open 收益；
- 不可交易、异常价格和数据不完整股票剔除；
- 第一版使用 WLS，并对回归残差做稳健裁剪；后续再评估 robust regression；
- 保存每日因子收益、股票残差、有效样本数、条件数和加权 R²。

## 因子协方差

对历史因子收益使用 EWMA：

```text
F_t = EWMA_cov(f_(t-L+1), ..., f_t)
```

第一版比较 60、90、120 日半衰期，窗口至少覆盖 252 个交易日。最终参数只能用历史预测误差选择。随后执行：

- 向对角阵或长期协方差收缩；
- 小特征值设置正数下限；
- 强制矩阵对称及半正定；
- 对短期波动状态可加入整体波动缩放，但不在第一版同时引入多个复杂校正。

## 特异风险

对每只股票历史特异收益 `u` 估计 EWMA 方差：

```text
sigma2_specific_(i,t) = EWMA(u_i^2)
```

随后按行业和规模分组向组内中位数收缩。历史短、停牌多或残差样本不足的股票增加收缩强度；设置合理的方差上下限。`D_t` 第一版使用对角阵，不急于估计特异收益之间的相关性。

优化时不需要显式生成完整的 500×500 矩阵，可以直接使用因子形式计算：

```text
portfolio_variance = (X' w)' F (X' w) + sum_i sigma2_specific_i * w_i^2
```

## 点时和时间口径

- 暴露日期 T 只能解释 T 之后的收益；
- 股本以 `available_date <= T` 为准；
- 行业需要生效日期，不能将当前行业回填到历史；
- 风险协方差只使用截至 T 已实现的因子收益；
- 新上市股票使用行业/规模组特异风险先验；
- 缓存指纹绑定输入数据、因子定义、处理参数和估计窗口。

## 验证

### 数据与代数检查

- 暴露不存在未来日期；
- 行业约束和风格加权均值满足定义；
- `F` 和最终风险矩阵对称、半正定；
- 组合风险分解之和等于总预测风险；
- 扩大单一股票权重时，特异风险贡献合理上升。

### 风险预测验证

对当前策略、等权基准、行业集中组合和随机可行组合分别比较：

```text
predicted_variance_t = w_t' Sigma_t w_t
realized_variance = future portfolio return squared or multi-day realized variance
```

至少报告：

- 预测/实现波动率比率；
- 20 日和 63 日滚动风险偏差；
- 不同波动状态、年份和组合类型下的偏差；
- 预测 tracking error 与实现 tracking error；
- 因子和特异风险贡献；
- 风险超限频率。

风险模型的第一目标是校准准确，而不是让回测收益最高。若一个风险模型提高收益但持续低估实现风险，不应进入优化器。

## 建议交付物

```text
risk_model/
  exposures/year=YYYY/part.parquet
  factor_returns.parquet
  factor_covariance/date=YYYY-MM-DD.parquet
  specific_variance/year=YYYY/part.parquet
  diagnostics/daily.csv
  diagnostics/forecast_calibration.csv
  manifest.json
  _SUCCESS
```

## 实施顺序

1. 审计并补齐点时行业数据；
2. 构建 size、beta、residual volatility、liquidity、momentum 和 reversal 暴露；
3. 实现逐日受约束 WLS 和特异收益；
4. 实现 EWMA + shrinkage 的 `F` 与收缩特异方差 `D`；
5. 完成风险预测校准报告；
6. 接入组合风险分析，但暂不改变持仓；
7. 校准通过后才接入风险成本 QP。

## 参考

- MSCI China Equity Factor Model: https://www.msci.com/downloads/web/msci-com/data-and-analytics/factor-investing/equity-factor-models/China%20Equity%20Factor%20Model-cfs-en.pdf
- MSCI Equity Factor Models: https://www.msci.com/data-and-analytics/factor-investing/equity-factor-models
- Ledoit and Wolf, covariance shrinkage: https://ledoit.net/honey.pdf
- Boyd et al., convex portfolio optimization: https://web.stanford.edu/~boyd/papers/cvx_portfolio.html
