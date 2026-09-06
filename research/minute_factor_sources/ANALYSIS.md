# 分钟 OHLCV 到日频因子：资料解析与候选池

## 结论

下一批应优先测试的不是再堆叠全天收益、尾盘收益或总成交额集中度，而是三条与现有 core24 明显不同的机制：**差分后的价量交互（尤其量先价行）**、**相对日内季节性的放量条件分布**、**按价格状态分层的分钟波动率**。它们全部可以在“全市场计算、仅落盘 CSI300+CSI500”的既定日度 parquet 流程中完成。

资料、访问状态见 [SOURCES.md](SOURCES.md)。本文件只摘要定义，不复制研报内容。

## 已解析的可复现机制

| 机制 | 日频原子量 | 平滑/输出 | 当前数据可行性 | 与 core24 的关系 |
| --- | --- | --- | --- | --- |
| CPV 基准 | 当日 `corr(close_t, volume_t)` | 过去 20 个交易日均值 | 可行 | `mf_return_amount_corr` 是 `corr(r, amount)`，不是这个量。 |
| 差分 CPV | `corr(delta close_t, delta volume_t)`；并按四个正负象限拆分 | 过去 20 日均值 | 可行 | 新机制。整体相关会抵消方向相反的象限，故应保留原子项。 |
| 量先价行 | `corr(delta volume_t, delta close_{t+1})`，同样按象限拆分 | 过去 20 日均值 | 可行 | 新机制；使用同日 `t+1`，不跨越收盘。 |
| 放量条件波动率 | 仅取 `volume_t > mean(volume)+std(volume)` 的分钟；对其收益算标准差 | 当日；可选过去 5/20 日均值 | 可行 | 与全样本 `mf_realized_volatility` 不同，条件在异常放量时段。 |
| 尾盘成交额/流通市值 | 14:30--15:00 成交额除以流通市值 | 当日或周频 | 缺少流通市值 | 现有 `mf_tail30_amount_share` 是日内份额，不能替代市值归一化版本。 |
| 分时 CPV | 8 个 30 分钟区间各自计算价量相关；再取均值、标准差、末段值 | 过去 20 日均值 | 可行 | 新的时段异质性机制。 |
| 高/低位条件波动 | 分钟滚动波动率；在当日价格最高/最低的 20%分钟里求均值并除全天均值 | 过去 20 日均值 | 可行 | 新机制，使用路径状态而不是总 RV。 |
| 聪明钱（原始 pooled 版） | 合并过去 10 日分钟，将 `|r|/volume^0.25` 排序，取累计成交量前 20%，算 smart VWAP / all VWAP | 10 日池直接形成一个值 | 可行 | 当前 smart-Q 是“每日 smart VWAP 比率再做 10 日归一化”，不等价于 pooled 选择。 |
| APM | 上午/下午股票收益分别对相应市场指数收益回归，使用残差构造预期利润动量 | 过去 20 日回归 | 需可靠指数分钟线 | 新机制；暂不应以个股全市场平均替代正式指数序列。 |

### 资料对关键设计的约束

- 东吴 CPV 资料的关键启发是：总体相关系数会混合方向相反的子状态。因此首轮应测试**有经济含义的原子象限**，不直接照搬研报在旧样本上拟合的复合权重。
- “量先价行”里的 `delta close_{t+1}` 只取 `t=1..239`；最后一分钟没有同日未来值，必须排除，不能回填或跨日连接。
- 信达放量定义是**个股、当日**分钟成交量均值加一标准差，不是跨股票阈值。没有正收益且放量的分钟时，该日因子为 null；不能把空集合编码为零。
- 所有 20 日平滑必须沿用现有交易日历语义：只取严格过去交易日，不让停牌日压缩窗口，也不在信号日使用未来数据。

## 候选名单（仅 OHLCV，按首轮优先级）

| 优先级 | factor_id 建议 | 定义摘要 | 窗口 | 原因 |
| --- | --- | --- | --- | --- |
| P0 | `mf_cpv_price_volume_level_w20` | `mean_20(corr(close_t, volume_t))` | 20 日 | CPV 基线，与现有 return-amount corr 非同一对象。 |
| P0 | `mf_cpv_dprice_dvolume_pp_w20` | `mean_20(corr(deltaP, deltaV \| deltaP>0,deltaV>0))` | 20 日 | 拆解总体抵消；保留方向，后续统一检验符号。 |
| P0 | `mf_cpv_dprice_dvolume_pm_w20` | 同上，`deltaP>0, deltaV<0` | 20 日 | 同上。 |
| P0 | `mf_cpv_dprice_dvolume_mp_w20` | 同上，`deltaP<0, deltaV>0` | 20 日 | 同上。 |
| P0 | `mf_cpv_dprice_dvolume_mm_w20` | 同上，`deltaP<0, deltaV<0` | 20 日 | 同上。 |
| P0 | `mf_cpv_dvolume_lead_dprice_pp_w20` | `mean_20(corr(deltaV_t,deltaP_{t+1} \| +,+))` | 20 日 | 量先价行，直接检验可预测的日内传播。 |
| P0 | `mf_cpv_dvolume_lead_dprice_pm_w20` | 同上，`+,-` | 20 日 | 同上。 |
| P0 | `mf_cpv_dvolume_lead_dprice_mp_w20` | 同上，`-,+` | 20 日 | 同上。 |
| P0 | `mf_cpv_dvolume_lead_dprice_mm_w20` | 同上，`-,-` | 20 日 | 同上。 |
| P0 | `mf_volume_up_return_std` | 放量分钟的收益标准差，按信达定义取负号；仅在存在放量正收益分钟时有效 | 当日 | 条件分布；不是重复 RV。 |
| P0 | `mf_volume_up_positive_return_std` | 只在放量且收益为正的分钟计算收益标准差 | 当日 | 这是对研报条件的显式变体；空集留 null。 |
| P1 | `mf_cpv_segment_std_w20` | 8 个 30 分钟 CPV 的当日标准差，再平滑 | 20 日 | 捕捉分时异质性；避免全天聚合的辛普森效应。 |
| P1 | `mf_cpv_segment_last30_w20` | 14:30--15:00 区间 CPV，再平滑 | 20 日 | 尾盘信息的价格-成交量交互。 |
| P1 | `mf_high_price_vol_ratio_w20` | 最高价位 20%分钟的 5 分钟滚动收益波动率均值/全天均值 | 20 日 | 高位放量/波动的路径状态。 |
| P1 | `mf_low_price_vol_ratio_w20` | 最低价位 20%分钟的同类比值 | 20 日 | 与高位项构成对照，不预设线性组合。 |
| P1 | `mf_smart_q_pooled_b025_w10_p20` | 10 日分钟池统一选 smart bars 后算 VWAP 比值 | 10 日池 | 正确复刻经典 pooled 语义，与当前版本不同。 |
| P2 | `mf_minute_amount_seasonality_return_corr_w20` | 当日 `corr(r_t, amount_share_t / past20_mean_share_t)`，再平滑 | 20 日 | 用历史同时刻季节性定义“异常量”，而非绝对量。 |
| P2 | `mf_tail31_amount_to_float_mv` | 14:30--15:00 成交额/流通市值 | 当日 | 信达来源支持，但先接好无前视流通市值。 |
| P2 | `mf_apm_am_pm_residual_w20` | 上午/下午股票收益对指数收益回归后的残差动量 | 20 日 | 需先明确指数分钟线与回归规格。 |

## 不进入当前 OHLCV 批次

`MPC/MPB`、有效价差、盘口深度、VOI/OFI/OIR、撤单、主动买卖、真实大单/机构成交占比都需要 L2 或逐笔成交。保留为数据升级后的第二阶段，不用分钟 K 线的价格位置或单分钟成交额去伪造。

## 复旦《HFFactor》课件补充的分钟 OHLCV 候选

课件已逐页读取，关键的“周期性成交占比”和“筹码分布”页也进行了视觉核验。下面只新增当前字段能严格支持、且不与现有 core24 完全重复的项目；`amount` 优先于 `volume_share`，因为前者天然按成交金额衡量。

表中后缀 `w20` 一律指在输出日使用 `[d-20,d-1]` 的交易日历窗口，严格排除信号日；无 `w20` 的是使用当日完整分钟线、收盘后可得的日频原子量。

| 优先级 | factor_id 建议 | 可复现定义 | 为什么值得补充 |
| --- | --- | --- | --- |
| P0 | `mf_amount_share_skewness` | 当日 241 个 `amount_t / sum(amount)` 的样本偏度 | 现有有 HHI、熵，但没有分布不对称性。 |
| P0 | `mf_amount_share_kurtosis` | 同一金额份额序列的 Pearson 峰度 | 区分“一次极端爆量”和一般集中度。 |
| P0 | `mf_return_volume_corr_w20` | `mean_20(corr(r_t, volume_share_t))` | 课件的 `Corr(R, V)`；不能用成交额相关性替代。 |
| P0 | `mf_return_amount_corr_w20` | `mean_20(corr(r_t, amount_t))` | 现有 `mf_return_amount_corr` 的严格过去 20 日平滑版；先做增量价值检验。 |
| P0 | `mf_cpv_price_volume_level_w20` | 已在上方候选池：`mean_20(corr(close_t, volume_share_t))` | 课件的 `Corr(P, V)`，因此不另起重复的 factor_id。 |
| P0 | `mf_price_dvolume_corr_w20` | `mean_20(corr(close_t, delta volume_share_t))` | 课件的 `Corr(P, ΔV)`，捕捉价格位置与边际增量资金的配合。 |
| P0 | `mf_return_damount_corr_w20` | `mean_20(corr(r_t, delta amount_t))` | 课件的 `Corr(R, ΔV)`；与现有 `corr(r, amount)` 不同。 |
| P0 | `mf_amihud_intraday_w20` | `mean_20(mean_t(abs(r_t) / max(amount_t, eps)))`，按固定金额单位缩放后落盘 | 是与现有 `sum(abs(r))/sqrt(volume)` 不同的经典单位金额冲击度量。 |
| P1 | `mf_amount_periodicity_peak_share_w20` | 先以严格过去 20 日同分钟均值去除 U 型季节性；对当日残差金额份额做 DFT，取非零频率最大功率/总非零功率，再平滑 | 课件的机构拆单“周期性成交”思想；去季节性是必要条件，不能直接对原始 U 型量做 FFT。 |
| P1 | `mf_amount_periodicity_band_power_w20` | 同一残差频谱中，预注册的一组中频 bin 功率占总功率 | 比单一最高峰更稳健，避免把噪声峰当作信号。 |
| P1 | `mf_chip_return_bin_std` | 按分钟收益率分箱；每箱金额占全天金额的比例，再对各箱份额计算标准差 | 与时间维度的 amount HHI 不同，衡量成交筹码在收益状态上的集中。 |
| P1 | `mf_chip_return_bin_skewness` | 同一“收益率箱-金额份额”向量的偏度 | 衡量筹码分布不对称性。 |
| P1 | `mf_chip_return_bin_kurtosis` | 同一向量的 Pearson 峰度 | 衡量是否集中于极少数收益状态。 |
| P1 | `mf_chip_top3_return_bin_share` | 分箱后金额份额最大的 3 个箱之和 | 课件“大筹码占比”的可审计版本。 |
| P1 | `mf_chip_return_q80` | 以分钟累计收益率为横轴、金额为权重的当日加权 80%分位数 | 课件“筹码占比分位”；先固定收益率口径为 `ln(close_t/open_0)`。 |
| P2 | `mf_segment_return_dispersion` | 8 个等长 30 分钟对数收益的标准差 | 补足现有仅尾盘、上午下午差的路径信息。 |
| P2 | `mf_segment_return_reversal` | `r_last30 - r_first30` 或 8 段收益的一阶负自相关 | 明确检验日内反转，不能直接推定符号。 |

## 日内时段因子族：段内值、差、比与相对异常

把连续交易时段固定切为 8 段，每段 30 分钟：`0930,1000,1030,1100,1300,1330,1400,1430`；午间休市不跨段。对每一个原子量先逐段计算，再生成有限、预注册的比较量。这样“开盘 30 分钟/收盘 30 分钟”不是两个孤立因子，而是同一语义下可系统筛选的一族候选。

| 原子量 | 段内日频值 | 推荐比较候选 | factor_id 示例 |
| --- | --- | --- | --- |
| 成交额 | `A_s / A_day` | 开盘/尾盘比、上午/下午比、尾盘/日内中位段比、相对历史同段 z-score | `mf_open30_tail30_amount_ratio`, `mf_am_pm_amount_ratio`, `mf_tail30_amount_to_segment_median`, `mf_open30_amount_z20` |
| 成交量 | `V_s / V_day` | 与成交额同构的比值；另测金额份额与股数份额之差 | `mf_open30_tail30_volume_ratio`, `mf_tail30_amount_minus_volume_share` |
| 收益 | `r_s=ln(P_end/P_start)` | 首末段差、上午下午差、尾盘相对前 7 段均值、相邻段反转 | `mf_tail30_minus_open30_return`, `mf_tail30_minus_prior7mean_return`, `mf_segment_return_lag1_autocorr` |
| 波动率 | `sqrt(sum(r_t^2), t in s)` | 尾盘/开盘比、下午/上午比、尾盘相对历史同段异常值 | `mf_tail30_open30_rv_ratio`, `mf_pm_am_rv_ratio`, `mf_tail30_rv_z20` |
| 价格冲击/Amihud | `mean(|r_t|/amount_t, t in s)` | 尾盘/开盘比，尾盘相对全天比 | `mf_tail30_open30_amihud_ratio`, `mf_tail30_day_amihud_ratio` |
| 价量相关 | `corr(r_t, amount_t)` 或 `corr(r_t, volume_t)`，要求段内有效点数足够 | 尾盘减开盘、尾盘减前 7 段均值、8 段标准差 | `mf_tail30_minus_open30_ra_corr`, `mf_segment_ra_corr_std` |

### 比值的统一防呆规则

- 对正的、尺度型原子量（成交额、成交量、RV、Amihud）落盘 `log((x_a+eps)/(x_b+eps))`，而不是裸比值；`eps` 固定为该指标全样本单位下的极小正数并写进 manifest。
- 对可为负的收益率、相关系数，使用**差**或 Fisher-z 变换后的差，绝不取 ratio。
- 分母段无成交、有效分钟不足或数据缺失时返回 null；不把无交易股票的分母替换成零。
- 第一轮只预注册相邻的开盘/尾盘、上午/下午、尾盘/全天三类比较；8×8 的全排列放到第二轮，防止参数挖掘。

### 建议追加的 P1 时段批次

`mf_open30_amount_share`、`mf_open30_tail30_amount_ratio`、`mf_am_pm_amount_ratio`、`mf_open30_amount_z20`、`mf_tail30_amount_to_segment_median`、`mf_tail30_open30_rv_ratio`、`mf_pm_am_rv_ratio`、`mf_tail30_open30_amihud_ratio`、`mf_tail30_minus_open30_return`、`mf_tail30_minus_prior7mean_return`、`mf_tail30_minus_open30_ra_corr`、`mf_segment_ra_corr_std`。

### 实现口径必须冻结的细节

- **分钟收益率**：`r_0=ln(close_0/open_0)`，其后 `r_t=ln(close_t/close_{t-1})`；筹码横轴用累计日内收益 `ln(close_t/open_0)`，不要混用两种口径。
- **筹码分箱**：首轮固定为 `[-5%,-4%),..., [4%,5%]` 加两个尾箱（`<-5%`、`>=5%`），再做一个等宽 0.5% 网格的稳健性版本；箱数和边界要写入 manifest。
- **频谱因子**：使用过去 20 个交易日的同分钟平均金额份额作基准，完全排除信号日以避免泄漏；DFT 前先减基准并去均值，排除零频。
- **成交额为零**：对应的 `abs(r)/amount` 不参与均值；若有效分钟少于预先规定阈值，返回 null，绝不以零补齐。

课件中的大资金涨幅 MB、PTOR/BNI/BAM/SAM、机构成交占比等都要求“笔数、单笔成交额或主动买卖方向”，当前分钟 OHLCV 没有这些字段，因此未列为可生产候选。

## 推荐的第一个生产批次

首轮可以扩为 18 个 P0 因子：原始 CPV/放量批 11 个，加上课件补充的金额份额偏度、金额份额峰度、三种尚未覆盖的基础价量相关和 Amihud 7 个。`Corr(P,V)` 已是上方 CPV 基线，故不重复生产。每个交易日读取当日 parquet，外加 20 个**严格过去**交易日的所需逐股日级中间量；全市场计算中间量和因子，最后按该日 CSI300∪CSI500 成分过滤写入一个宽表 daily parquet。这样既与当前生产语义一致，也使回测只读取一个日期文件。

后续评价不以单一全样本 ICIR 入选：先做方向冻结、子样本稳定性、行业/市值中性化后 IC、与 core24 的 Spearman 相关、缺失率和换手敏感性，再挑选低相关的进入下一批。
