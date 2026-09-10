# 分钟 OHLCV 日频因子候选清单 V2（区别于已实施的 core24 + ohlcv_candidates_v1）

编制日期：2026-09-06（2026-09-09 重建，内容未变）。依据本轮搜集的 GitHub 复现仓库（中金手册 CICC、方正多因子 FZ、Kysec、QuantsPlaybook、Quantitative-analysis）与研报 PDF（东北因子选股系列、东方九、国君八十四、开源微观结构系列 1/7、方正分钟系列、光大七、财通特质偏度、国盛量价淘金七、海通四十六等）整理。来源登记见 [SOURCES.md](SOURCES.md)。

## 0. 口径与边界（沿用现有生产框架）

- 每个完整交易日 241 根 1 分钟 bar；`r_0 = ln(close_0/open_0)`，其后 `r_t = ln(close_t/close_{t-1})`；`amount` 优先于 `volume_share` 衡量资金。
- 所有 `wN` 窗口一律 `[d-N, d-1]` 严格过去交易日，按交易日历对齐，`shift(1)` 排除当日；无 `wN` 的为当日收盘后可得的日频原子量。
- 空集/有效分钟不足返回 null，不以零补齐；比值型原子量落盘 `log((a+eps)/(b+eps))`，可负量用差或 Fisher-z。
- "市场项"（APM、协同、羊群等）默认以**信号日可见的全市场横截面**构造：等权或成交额加权均值。不引入外部指数分钟线也可先行；接通指数分钟线后再切换正式口径。
- 跨日价格比较必须用前复权日线换算；隔夜缺口用日线 `open/prev_close`。

**与现有 69 个已实施因子的去重原则**：凡与 core24（realized_vol/skew/kurt、downside_semivariance_ratio、trend_efficiency、amount_hhi/entropy、smart_q 族、tail30 族、amihud 等）或 45 候选（CPV 差分象限族、放量条件族、筹码分箱族、时段比值族、频谱族等）语义相同或仅量纲替换（amount↔volume）的，一律不列为新候选，只进入文末"近似重复记录"。

候选总量：**63 个新 factor_id**（P0 23 个、P1 29 个、P2 11 个），按 9 个机制族组织。

---

## A 族：收益分布高阶结构（已实现矩的扩展）

机制：现有 `mf_realized_skewness / mf_realized_kurtosis` 只用到三、四阶矩；研报证据表明更高阶矩、跳跃拆分、尾部条件量与"矩的不确定性"包含增量信息。

| # | factor_id | 定义 | 来源 | 与现有因子的区别 / 实现注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| A1 | `mf_realized_hyperskew_w20` | 五阶已实现超偏度：`m5 = Σr_t^5`，标准化 `m5 / RV^2.5`，20 日均值 | 东北《高频数据下的已实现高阶矩因子及改进》 | 新阶数；对重尾+不对称同时敏感。原文提示高阶矩噪声大，**必须配合 5min 降采样**版本 | P1 |
| A2 | `mf_realized_hyperkurt_w20` | 六阶已实现超尾度：`m6 / RV^3`，20 日均值 | 同上 | 同上 | P1 |
| A3 | `mf_realized_skew_5m_w20` | 先把 1min bar 无重叠聚合为 5min bar 再算偏度（10min 同理做稳健版），20 日均值 | 同上（原文核心改进：下采样平均去噪） | 不是新机制，是对现有 realized_skew 的**口径变体**；用于检验 1min 噪声对现有因子的影响 | P0（作为校准项） |
| A4 | `mf_rv_jump_share_w20` | 跳跃占比：`max(RV − BPV, 0)/RV`，其中 `BPV = (π/2)·Σ|r_t·r_{t−1}|`（bipower variation），20 日均值 | 标准高频计量；方正（6）《股价跳跃及其对振幅因子的改进》同方向 | 现有只有总 RV；跳跃-连续分解是新对象。首分钟无 `r_{t-1}` 时从 t=1 起算 | P0 |
| A5 | `mf_realized_skew_pos_w20` | 暴涨因子：只取 `r_t > 0` 的三阶贡献 `Σ_{r>0} r_t^3 / Σ r_t^3`（占比），20 日均值 | 财通《博彩偏好还是风险补偿？高频特质偏度因子全解析》 | 把偏度拆成正/负尾贡献；原文结论"暴涨侧主导"。分母为零返回 null | P0 |
| A6 | `mf_realized_skew_neg_w20` | 暴跌因子：`Σ_{r<0} r_t^3 / Σ r_t^3`，20 日均值 | 同上 | 与 A5 成对出现，方向分别检验 | P0 |
| A7 | `mf_left_tail_mean_w20` | 左尾强度：当日最差 5% 分钟（约 12 根）收益的均值，20 日均值 | 东北 18《基于高频数据的风险不确定性因子》左尾风险组件 | 尾部条件期望（cVaR 型），与偏度不同（偏度被中间区域稀释） | P0 |
| A8 | `mf_left_tail_cvar_ratio_w20` | 左尾集中度：最差 5% 分钟的 `Σr_t^2 / RV` | 同上 | 尾部方差占比；与 A4 跳跃占比互补（负向跳跃 vs 全部跳跃） | P1 |
| A9 | `mf_id_vov_w20` | 特质已实现波动率的不确定性：`std_20(IRV_d)`，其中 `IRV_d` 为对市场分钟收益回归残差的 RV（市场项用横截面等权分钟收益代理） | 东北 18 ID_VOV（Rank IC 5.77%，ICIR 1.26） | "波动率的波动率"特异性版；纯时序 VoV 见 E 族。回归可用单因子简化 | P1 |
| A10 | `mf_left_tail_uncertainty_w20` | 左尾风险的不确定性：`std_20(left_tail_mean_d)` | 东北 18 ID_VOcVaR95 组件 | 二阶套娃：尾部风险本身的稳定性 | P1 |

> UOIDR 最终合成（已实现波动率、左尾、右尾三分类各选绩优再等权）暂不预注册，等 A7–A10 单因子检验后再定权重，避免照搬旧样本拟合。

---

## B 族：反转/动量的时间切割

机制：开源《A 股反转之力的微观来源》证明传统 20 日反转可按日内时间与每日状态切割出更稳健的成分；方正《黄金律》与 Lou-Polk-Skouras 证明隔夜与日内成分方向相反。

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| B1 | `mf_reversal_hour_weighted_d20` | 过去 20 日分钟收益加权和：`Σ_d Σ_t w(t)·r_{d,t}`，`w(t) = t/240`（线性递增，临近收盘权重高），再除以 `Σ w` 归一 | 开源《市场微观结构研究系列（1）》引 2017-9-14 日内加权反转 | 现有 reversal 均为日收益级别；这是分钟级时间加权反转。权重形状预注册为线性，禁止网格搜权重 | P0 |
| B2 | `mf_overnight_gap_sum_w5` | 过去 5 日隔夜缺口之和：`Σ ln(open_d/prev_close_d)`（前复权） | 方正《行业轮动的黄金律》2017；Lou-Polk-Skouras NBER w24465 | 日线即可算，但属于该机制族最小完整成员；隔夜反转+日内动量成对检验 | P0 |
| B3 | `mf_intraday_momentum_sum_w5` | 过去 5 日日内收益之和：`Σ ln(close_d/open_d)` | 同上 | 与 B2 成对；两者相关性应显著为负，正交后各留残差 | P0 |
| B4 | `mf_apm_tstat_w20` | APM：过去 20 日，每股日上午/下午收益对市场（横截面等权代理）上下午收益做混合 OLS，`stat = mean(ε_am − ε_pm) / (std(ε_am − ε_pm)/√20)` | 开源《市场微观结构研究系列（5）：APM 进阶版》；Kysec `factors/apm.py` 逐行复现 | 新机制（上下午残差差）。市场代理口径要在 manifest 冻结；接通指数分钟线后重跑 | P0 |
| B5 | `mf_tail30_sharpe_ratio` | 尾盘 30 分钟"夏普"：`r_last30 / sqrt(Σ_{t∈尾盘} r_t^2)`，当日值 | 国君《数量化专题之八十四：神秘的尾盘 30 分钟》（DailyR/DailyV/DailyMI 三因子体系） | 现有 tail30_return 与 tail30_rv_z20 是分离的；比值是条件信息效率 | P1 |

> 国君 84 的 DailyMI（尾盘 Amihud）已被 `mf_tail30_open30_amihud_ratio / mf_tail30_day_amihud_ratio` 覆盖；DailyV 的金额版被 `mf_tail30_amount_z20` 覆盖，volume 版记为近似重复（见文末）。

---

## C 族：聪明钱与单笔金额切割

现有 `mf_smart_q_*` 族（b=0.10/0.25/0.50、logv、pooled）已完整覆盖方正 2016《跟踪聪明钱》与开源 2.0 的 pooled 语义。本族仅登记**不可实施**项与一个增量变体。

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| C1 | `mf_smart_q_5m_b025_w10_p20` | 聪明钱 Q 的 5min bar 版（把 1min 聚成 5min 再按 `|r|/V^0.25` 选前 20% 量） | 东北 15 的降采样思想应用到现有因子 | 检验 bar 粒度对现有 smart_q 的噪声影响；非新机制 | P2 |
| — | ~~W 式切割反转、平均单笔成交金额族~~ | 需要每日成交**笔数**（`W = amount/笔数`） | 开源系列（1）、海通 46 | **数据边界外**（逐笔/L2），不实施 | 排除 |

---

## D 族：成交量分布结构（时间、收益、价格三维）

机制：东北 20 把日内成交量分布切成四个维度（对数分布、时间 U 型、收益条件、价格区间），其中价格区间维度（POC/筹码公允区）与现有"按累计收益分箱"的筹码族不同——**按价格水平分箱**是全新对象。

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| D1 | `mf_logvol_intraday_std_w20` | 日内对数成交量离散：`std_t(ln(amount_t + 1))` 先减过去 20 日同分钟均值（去 U 型季节性），再取当日 std，20 日均值 | 东北 20 维度一 | 现有频谱族只取功率比；这是残差量的整体离散度。**必须先去季节性** | P0 |
| D2 | `mf_open30_volshare_stability_w20` | 早盘量占比稳定性：`−std_20(open30_amount_share_d)` | 东北 20"早盘成交量波动稳定性因子" | 负号：稳定=高因子值。与 `mf_open30_amount_z20`（当日异常）不同，这是**跨日稳定性** | P1 |
| D3 | `mf_pm_open30_to_am_open30_vol_ratio` | 下午开盘 30 分钟金额 / 上午开盘 30 分钟金额（13:00–13:30 vs 09:30–10:00），log 比值，20 日均值 | 东北 20"早午开盘成交量比例因子" | 现有时段族没有下午开盘段专项比较 | P1 |
| D4 | `mf_upvol_share_stability_w20` | 价格显著上行段的量占比稳定性：上行分钟（`r_t > 过去20日同分钟 r 的 75% 分位`）金额占比的 `−std_20` | 东北 20"价格显著上行价稳量比稳定性" | 条件集合 + 跨日稳定性的组合，现有族无此对象 | P1 |
| D5 | `mf_pricebin_volume_skew` | 成交量在价格上的分布偏移：当日价格按分位十等分，每箱金额占比 `v_k`，位置标号 `k/10` 的偏度 `skew(k, v_k)` | 东北 20 `v_p_skewness` | 与 `mf_chip_return_bin_skewness`（按累计收益分箱）不同：这里按**价格水平**分箱，衡量公允区偏移 | P0 |
| D6 | `mf_close_vs_poc` | 收盘相对 POC：`(close − POC) / (day_high − day_low)`，POC = 金额最大价格箱的中点价 | 东北 20 POC 反转改进 | 新对象（公允价格锚）；当日值，可再 20 日平滑 | P0 |
| D7 | `mf_logsig_v_l2_1` | 成交量路径签名第 1 项：`X^1 = Σ Δln(amount_norm)`（level-1 log-signature） | 东北 20 Logsig-Alpha；仅预注册 level-2 | `amount_norm` 为去季节性后标准化序列 | P2 |
| D8 | `mf_logsig_v_l2_21` | 签名第 2 项：`X^{2,1} = (1/2)·Σ(ΔX^2·X^1 − ΔX^1·X^2)` 的 (2,1) 分量（收益-量二维路径） | 同上 | 阶数冻结在 2，防止特征爆炸；两项足够检验路径信息增量 | P2 |
| D9 | `mf_tail_volume_weighted_return` | 尾盘量加权收益：`Σ_{t∈last30}(v_t·r_t) / Σ v_t`，当日 | 中金手册 CICC 仓 `cal_trade_bottom20retRatio/bottom50retRatio` | 现有尾盘因子均为金额份额或收益差；量加权收益是新聚合方式 | P0 |
| D10 | `mf_topvol_ret_w20` | 顶量分钟动量：当日金额 Top-50 分钟的 `∏(1+r_t) − 1`，20 日均值 | 中金手册 CICC `cal_mmt_top50VolumeRet` | 条件子集的收益方向（vs 现有 `mf_volume_up_return_std` 是条件波动） | P0 |
| D11 | `mf_bottomvol_ret_w20` | 底量分钟动量：金额 Bottom-50 分钟同构造 | CICC `cal_mmt_bottom50VolumeRet` | 与 D10 成对（缩量时的漂移方向） | P1 |
| D12 | `mf_last3bar_auction_share` | 收盘集合竞价占比：最后 3 根 bar（14:57–15:00）金额 / 全日金额，20 日均值 | 中金手册 CICC `cal_liq_lastCallR` | 沪深两市 14:57 后均为集合竞价，1min bar 可直接算 ✓ | P0 |
| D13 | `mf_firstbar_auction_share` | 开盘竞价占比：首根 bar 金额 / 全日金额 | CICC `cal_liq_firstCallR`；光大五《见微知著》同构造 | **先对账**：1min 首根 bar 是否含 9:25 竞价量（与日线量对账），未确认前不进生产 | P2 |

---

## E 族：波动结构与量价博弈（方正分钟系列 + FZ 复现）

机制：方正"多因子选股系列"九个因子全部可由分钟 OHLCV 复现（FZ 仓库 polars 实现逐行可查）。共同防呆：剔除 09:35 前与 14:53 后的 bar（防开盘收盘噪声，与原文一致）。

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| E1 | `mf_dazzling_vol_w20` | 耀眼波动率：激增分钟（`Δv_t > mean(Δv)+std(Δv)`，当日）处，未来 5 根 bar 收益的 std，当日均值后取 20 日 `mean+std` | 方正（1）《成交量激增时刻》；FZ `cal_YaoYanBoDongLv` | 现有放量族是"放量分钟的收益 std"；这里是"**量激增后 5 分钟窗口**的收益 std"（前瞻窗口、量差条件） | P0 |
| E2 | `mf_dazzling_ret_w20` | 耀眼收益率：激增分钟当根收益的均值（截面去均值后 20 日 `mean+std`） | FZ `cal_YaoYanShouYiLv` | 适度冒险（E3）的单成分 | P0 |
| E3 | `mf_moderate_risk_composite` | 适度冒险：`z20(耀眼波动率) + z20(耀眼收益率)`，z 为 20 日 mean+std 后截面标准化 | 方正（1）；FZ `cal_ShiDuMaoXian`（Rank IC −8.89%，ICIR −4.84） | 合成权重照原文等权；若 E1/E2 单因子已强则不叠合成 | P1 |
| E4 | `mf_tide_strong_rate_w20` | 强势半潮汐：主量峰（9 根中心滚动量和 argmax）前"涨潮速率" `((P_peak/P_t −1)/(t_peak − t))` 在量谷处的取值，20 日均值 | 方正（2）《潮汐》；FZ `cal_QiangShiBanChaoXi` | 量峰-价谷的结构速率；全新路径几何对象 | P1 |
| E5 | `mf_tide_weak_std_w20` | 弱势半潮汐：次级量峰退潮速率，20 日 std | FZ `cal_RuoShiBanChaoXi` | 潮汐 = E4 均值 + E5 标准差（原文合成） | P2 |
| E6 | `mf_climb_cov_w20` | 攀登：`cov(ret_to_vol, better_vol)`，其中 `better_vol` = 过去 20 根 bar 的 4 价格 × 5 滞后共 20 值的变异系数；条件：`better_vol ≥ 当日 mean+std` | 方正（3）《勇攀高峰》；FZ `cal_PanDeng` | 高波动 bar 的信息含量协方差；20 日 mean+std 合成 | P1 |
| E7 | `mf_jump_taylor_residual_w20` | 跳跃度（泰勒残差）：`Σ[2(r_pct − r_log) − r_log²]`，r_pct/r_log 为同一区间简单与对数收益，20 日 mean+std | 方正（6）；FZ `cal_TiaoYueDu` | 与 A4 bipower 跳跃互补的另一种跳跃代理（凸性残差） | P1 |
| E8 | `mf_reversal_amplitude_w20` | 反转振幅：`(high−low)/prev_close`，符号由当日跳跃度相对截面均值的方向决定，20 日均值 | 方正（6）；FZ `FanZhuanZhenFu` | 条件振幅（跳跃方向 × 振幅） | P2 |
| E9 | `mf_ambiguity_amount_corr_w20` | 模糊性-金额相关：`ambiguity_t = roll_std_5(roll_std_5(r))`（09:40 起），`corr(ambiguity, amount)` 当日，20 日均值 | 方正《云开雾散》；FZ `cal_MoHuGuanLianDu` | "波动率的波动率"与量的联动；VoV 本体 + 量条件 | P0 |
| E10 | `mf_ambiguity_amount_ratio_w20` | 模糊金额比：高模糊（>当日均值）分钟的 `mean(amount)/mean(全部 amount)`，20 日均值 | FZ `cal_MoHuJinEBi` | E9 的比值版，更稳健 | P1 |
| E11 | `mf_ambiguity_price_gap_w20` | 模糊价差：高模糊分钟金额比 − 股数比（`amount` 与 `volume` 两个口径之差），20 日均值 | FZ `cal_MoHuJiaCha` | 捕捉模糊时段的成交单价偏移（大单倾向） | P2 |
| E12 | `mf_vol_diff_ols_tstd_w20` | 量差回归 t 值标准差：当日 `r_t` 对 `Δv_{t}, Δv_{t−1}, …, Δv_{t−5}` OLS（09:37–14:53），取 5 个量差斜率 t 值的 std | 方正《花隐林间》；FZ `cal_ZhaoMoChenWu` | 比 CPV 相关族更结构化（多阶滞后 + 显著性）；替代被排除的"单笔金额"信息 | P0 |
| E13 | `mf_vol_diff_ols_intercept_w20` | 同一回归截距 t 值的绝对值 × sign(F 统计量 − 截面均值)，20 日均值 | FZ `cal_WuBiGuMu` | 量无法解释的收益残留（消息驱动分钟） | P1 |
| E14 | `mf_volume_follow_ratio_w20` | 跟随系数：Top-10 量分钟之后的 5 分钟窗口内的量占全日比例（09:45 起），20 日 mean+std | 方正《待著而救》；FZ `cal_GenSuiXiShu` | 量峰后的"跟风承接"度量，与放量的持续性相关 | P1 |
| E15 | `mf_volume_game_return_w20` | 量博弈-收益：把当日 bar 按"过去 5 分钟收益"升序/降序各排一次，`Σ[cumsum(v_升序) − cumsum(v_降序)]`，20 日 mean+std | 方正《多空博弈》；FZ `cal_ChengJiaoLiangBoYi_ShouYiLv` | 量在收益排序下的分布不对称性（追涨杀跌 vs 逆势承接） | P1 |
| E16 | `mf_volume_game_position_w20` | 量博弈-位置：同一构造，排序键换成日内相对位置 `(close/当日high + close/当日low)/2 − 1` | FZ `cal_ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi` | 高位放量 vs 低位放量的积分差 | P1 |
| E17 | `mf_amplitude_game_w20` | 振幅博弈：排序键为 5 分钟收益，值为 bar 振幅的同一 cumsum 差 | FZ `cal_ZhenFuBoYi` | 振幅在涨跌排序下的不对称 | P2 |
| E18 | `mf_volume_sync_corr_w20` | 成交量协同：bar 按布林状态（±1/0，基于过去 20 根 bar 均值±std）分组，个股状态量份额与**全市场同状态量份额减自身**的相关系数 | 方正《协同效应》；FZ `cal_ChengJiaoLiangXieTong` | 需全市场横截面（自有数据可算）；与 G 族羊群互补 | P1 |
| E19 | `mf_panic_dispersion_w20` | 惊恐度（草木皆兵核心）：`(|r_d^stock| − |r_d^bench|) / (|r_d^stock| + |r_d^bench| + 0.1)`，bench 为成交额加权横截面日收益；再取 `−Δ(惊恐度两日均值)` 为衰减项，两者之和 20 日均值 | 方正（8）《草木皆兵》；FZ `cal_CaoMuJieBing` | 原文用流通市值加权，这里以成交额加权替代（manifest 冻结）；与 RV 相乘的增强项暂不注册 | P1 |

---

## F 族：路径几何与非流动性

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| F1 | `mf_path_tortuosity_w20` | K 线最短路径弯曲度：单 bar 路径长 `|open−high|+|high−low|+|low−close|`，全日加总 `L`；直线 `D = |day_close − day_open|`；`log(L/max(D, tick))`，20 日均值 | 光大七《基于 K 线最短路径构造的非流动性因子》（TS 变形最优） | 与 Amihud（单位金额冲击）正交的几何非流动性；TS 精确变形公式需与原文 PDF 第 3 节核对后冻结 | P0 |
| F2 | `mf_path_illiquidity_w20` | 路径非流动性：`L / amount_day`（单位金额的路径长度），20 日均值 | 同上 | 与 `mf_amihud_intraday_w20`（|r|/amount）区分：用路径长度而非收益幅度 | P0 |
| F3 | `mf_direction_changes_w20` | 路径换向次数：`Σ 1[sign(Δclose_t) ≠ sign(Δclose_{t−1})] / 240`，20 日均值 | 方正《凤鸣朝阳》日内模式思想的可审计退化版 | 原文用形态聚类（不可审计）；退化为预注册统计量 | P1 |
| F4 | `mf_upper_shadow_minute_share` | 分钟上影占比：`Σ 1[(high−max(open,close)) > (min(open,close)−low)] / 有效bar数`，当日 | 同上（对应日频上下影线因子的分钟内部版，QuantsPlaybook 有日线复现） | 上/下影分钟计数比；形态类最低成本成员 | P2 |

---

## G 族：趋势资金与羊群（横截面条件）

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| G1 | `mf_trend_capital_vwap_gap` | 趋势资金 VWAP 偏离：趋势分钟 = 当日 `v_t > 过去 5 日分钟量 90% 分位`；`(VWAP_趋势 − VWAP_全日)/VWAP_全日`，20 日均值 | 国盛《量价淘金（七）》前传定义（趋势资金交易行为因子年化 ICIR 4.37） | 阈值 90%、回看 5 日为原文冻结参数 | P0 |
| G2 | `mf_trend_capital_ret_w20` | 趋势资金方向收益：趋势分钟量加权收益 `Σ v_t·r_t / Σ v_t`，20 日均值 | 同上 | 与 G1 构成方向+成本两维 | P1 |
| G3 | `mf_herd_follow_ratio_w20` | 羊群效应：极端跟随分钟（趋势分钟后 3 根内、量再超当日 90% 分位且同向）的量 / 趋势分钟量，20 日均值 | 国盛《量价淘金（七）》（月度 Rank IC −0.084，ICIR −4.09） | 跟随窗口 3 分钟为预注册近似；原文细节以 PDF 第 2 节为准 | P1 |

---

## H 族：中金手册独有且未被前面覆盖的对象

CICC 仓库约 55 个因子中，扣除与现有 69 个近似重复的（见文末），独有新增如下：

| # | factor_id | 定义 | 来源 | 区别 / 注意 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| H1 | `mf_vol_range1min_std_w20` | 分钟极值比离散：`std_t(high_t/low_t)`（当日），20 日均值 | 中金手册 `cal_vol_range1min` | bar 内几何（H/L 比）的离散度，不与收盘收益 RV 重复 | P1 |
| H2 | `mf_minute_ols_qrs_w20` | 分钟 QRS：50 根 bar 滚动窗内 `low` 对 `high` 回归的斜率标准化末值 × R² 均值：`R̄²·(β_last − β̄)/std(β)` | 中金手册 `cal_mmt_ols_qrs`（RSRS 的高频化） | 把日线 RSRS 择时指标移到日内；纯 bar 内 OHLC 回归 | P1 |
| H3 | `mf_minute_ols_r2_mean_w20` | 50 根滚动窗 `corr²(low, high)` 的当日均值，20 日均值 | CICC `cal_mmt_ols_corr_square_mean` | H2 的退化成分，单因子检验用 | P2 |
| H4 | `mf_between_momentum_w20` | 去头尾动量：10:00–14:29 区间 `close_last/open_first − 1`，20 日均值 | CICC `cal_mmt_between` | 现有时段族是首尾段；这是**剔除首尾**的中间段动量 | P1 |
| H5 | `mf_doc_vol5_ratio` | 按价格分组筹码 Top-5 占比：分钟按 `close_last/close_t` 分组后金额份额前 5 组之和 | CICC `cal_doc_vol5_ratio`（与 `doc_vol10/50` 同族，取 5 最锐） | 与 `mf_chip_top3_return_bin_share` 的区别：分箱键是**价格比**而非累计收益，且不重叠的原始分组 | P1 |
| H6 | `mf_doc_pdf80` | 筹码 80% 分位：按上述价格比分组，金额累计到 80% 处的组序号（rank 归一） | CICC `cal_doc_pdf80`（60/70/90/95 同族，预注册仅 80） | "多数筹码成交在什么价位"的直接读数；只注册一个分位防挖掘 | P1 |

---

## I 族：日线数据即可实施的同源因子（边界延伸）

这些不是分钟合成的，但是同批研报中机制完整的因子，且与分钟湖的隔夜/时段口径天然衔接；单列以便与分钟因子联合正交。

| # | factor_id | 定义 | 来源 | 优先级 |
| --- | --- | --- | --- | --- |
| I1 | `mf_ideal_amplitude_d20` | 理想振幅：过去 20 日按收盘价排序，最高 5 日振幅均值 − 最低 5 日振幅均值（振幅 = high/low − 1） | 开源《振幅因子的隐藏结构》（多空年化 23.3%，ICIR −2.97）；Kysec `IdealAmplitudeFactor` 逐行复现 | P0 |
| I2 | `mf_ideal_turnover_d20` | 理想换手率：同一框架对换手率切割（需换手率日线字段） | 开源系列（7）后半部分 | P2 |

---

## 排除清单（数据边界外，明确不做）

| 家族 | 所需数据 | 来源 |
| --- | --- | --- |
| W 式切割反转、平均单笔成交金额/流出占比、大单驱动涨幅 | 成交**笔数**（逐笔） | 开源系列（1）、海通 46 |
| 大小单资金流、主买主卖、知情交易概率 VPIN、订单簿/OFI | L2/逐笔方向 | Kysec `money_flow`、海通 L2 系列、招商、天风 |
| 聪明钱 3.0、快照量价背离 | 盘口/逐笔 | 开源、国金 |
| 成交量波动选股（点击率版） | 另类数据（点击量） | 银河 2016 |
| 协同价差（与 top30 相关股票收益差的截面矩阵乘） | 全市场分钟宽矩阵（可算但成本高，暂缓） | FZ `cal_XieTongJiaCha` |

## 近似重复记录（检索到但判定不新增 factor_id）

| 来源因子 | 与现有的重复点 |
| --- | --- |
| CICC `shape_skewVol / kurtVol`（分钟量占比偏/峰度） | `mf_amount_share_skewness / kurtosis`（amount 口径已实施，volume 口径仅量纲差异） |
| CICC `doc_skew / doc_kurt / doc_std` | `mf_chip_return_bin_skewness / kurtosis / std`（分组键近似） |
| CICC `vol_upVol / downVol / upRatio / downRatio` | `mf_downside_semivariance_ratio` + core24 realized 矩 |
| CICC `corr_pvd / pvl / pvr / prvr` | `mf_cpv_dvolume_lead_dprice_*` 与 `mf_return_damount_corr_w20` 已覆盖 lead-lag 与变化率方向 |
| CICC `mmt_am / mmt_pm / mmt_paratio / mmt_last30` | `mf_pm_minus_am_return`、`mf_tail30_return` |
| CICC `liq_amihud_1min` | `mf_amihud_intraday_w20` |
| CICC `trade_headRatio / tailRatio` | `mf_open30_amount_share`、时段族 |
| 光大八 49 时段量占比 | 时段族（8 段）已覆盖主信息，49 段网格化视为参数挖掘 |
| 东北 15 已实现方差/偏度/峰度本体 | core24 `mf_realized_volatility / skewness / kurtosis`（A3 降采样为口径校准） |
| 东方九 日内特质波动率 | 与 A9 `mf_id_vov_w20` 的回归规格冲突处已合并；其"特质偏度/峰度"被 core24 + 财通暴涨暴跌（A5/A6）覆盖 |
| 海通 2017《价量新因子测试》量幅相关 | `mf_return_volume_corr_w20`、`mf_cpv_price_volume_level_w20` |

## 建议实施顺序

1. **第一批（P0，23 个）**：A3（口径校准）、A4–A7（跳跃与尾部）、B1–B4（时间切割反转与 APM）、D1、D5、D6、D9、D10、D12（量分布与 POC）、E1、E2、E9、E12（激增时刻、模糊性、量差回归）、F1、F2（路径非流动性）、G1（趋势资金）、I1（理想振幅）。
2. **第二批（P1，29 个）**：其余 P1 项；UOIDR（A 族合成）与适度冒险（E3）在单因子检验后再定合成权重。
3. **第三批（P2，11 个）**：口径变体与高成本项（D13 需先完成首根 bar 竞价量对账；协同价差需全市场宽矩阵）。
4. 每个因子沿用现有评估协议：方向冻结 → 子样本稳定 → 行业/市值中性化 IC → 与 core24+45 的 Spearman 相关（阈值 |ρ|<0.6）→ 缺失率与换手敏感性。

> 公式若有与原文出入，以仓库代码（FZ/CICC polars 实现）与 PDF 原文为准；本文档的冻结口径以 manifest 为最终裁决。
>
> **2026-09-08 实施后记**：63 个槽位已在 `candidates_v2.rs` 落地，但其中 15 个为全空列（见 `FACTOR_FORMULA_IC_TABLE.md` 附录），实际可用 48 个；逐因子多周期 IC 见该表。
