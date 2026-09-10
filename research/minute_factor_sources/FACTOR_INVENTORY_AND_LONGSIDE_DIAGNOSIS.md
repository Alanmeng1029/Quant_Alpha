# 已实施分钟因子清单 + 多头能力诊断（2026-09-08）

（2026-09-09 重建，内容未变。）本文回答两个问题：**现在到底有哪些候选因子、公式在哪查**；以及**为什么多头弱、还缺什么**。所有实测数字均为 OOS 标签重算（CSI300∪CSI500，2021-04-01 至 2026-08-28，open-to-open h1/h5，超额相对中证500 同口径）。

---

## 一、当前因子全景（132 个已落盘）

| 数据集 | 因子数 | 生产入口 | 详细公式位置 |
| --- | ---: | --- | --- |
| `core24/v1` | 24 | `quant-minute-factor build --factor-set core24` | **manifest.json 内逐因子公式串**（已复制到下方 §1.1） |
| `ohlcv_candidates_v1` | 45 | `--factor-set ohlcv_candidates_v1` | [ANALYSIS.md](ANALYSIS.md)（CPV 差分象限、放量条件、筹码分箱、时段族、频谱族）+ `crates/quant-minute-factor/src/candidates.rs` |
| `ohlcv_candidates_v2` | 63 | `--factor-set ohlcv_candidates_v2` | [CANDIDATES_V2.md](CANDIDATES_V2.md)（逐因子公式、来源、优先级）+ `crates/quant-minute-factor/src/candidates_v2.rs`；**其中 15 个为全空列，实际可用 48** |

正式模型集 `o2o_daily60_minute38_v1` = 60 日频（gtja/wq alpha 族）+ 28 个 v1 筛选 + 10 个 core24 筛选；`plus-v2` 实验再加 33 个 v2 筛选（`configs/candidate_factors_minute_ohlcv_v2_open_to_open_abs_icir_gt2_h1_or_h5.txt`）；`v2only` 实验单独用 33 个 v2 因子建模。

### 1.1 core24 详细公式（摘自 core24/v1 manifest，逐字）

| factor_id | 公式 |
| --- | --- |
| `mf_smart_q_b010_w10_p20` | `Q = S_d / mean(S, [d-9,d] 按日历窗口含信号日，前置≥5天有效)`；`S = smart_vwap/day_vwap`；smart 集 = 分钟按 `|r|/volume^0.10` 降序累计到当日量 20% 为止（跨过 bar 计入）；`r_0=ln(close_0/open_0), r_t=ln(close_t/close_{t-1})`。b025/b050 换 β；logv 版排序键 `|r|/ln(1+volume)`；p15 版阈值 15% |
| `mf_close_to_vwap` | `close_240/day_vwap − 1`，`day_vwap = Σamount/Σvolume`（全日） |
| `mf_tail30_return` | `ln(close_240/close_210)`，仅日内不跨夜 |
| `mf_tail30_amount_share` | `Σamount(211..240)/Σamount(全日)` |
| `mf_tail30_vwap_to_day_vwap` | `tail30_vwap/day_vwap − 1` |
| `mf_pm_minus_am_return` | `ln(close_240/close_120) − ln(close_120/close_0)`（120=11:30 bar） |
| `mf_realized_volatility` | `sqrt(Σr_t², t=1..240)` |
| `mf_downside_semivariance_ratio` | `Σmin(r_t,0)² / Σr_t²` |
| `mf_realized_skewness` | 分钟收益中心化样本偏度 `m3/m2^1.5` |
| `mf_realized_kurtosis` | 分钟收益 Pearson 峰度 `m4/m2²`（非超额） |
| `mf_trend_efficiency` | `|close_240−close_0| / Σ|close_t−close_{t−1}|`（Kaufman 效率） |
| `mf_max_intraday_drawdown` | `min_t(close_t/running_max − 1)` |
| `mf_amount_hhi` | `Σs_t²`，`s_t` 为分钟金额份额 |
| `mf_amount_entropy` | `−Σs_t·ln(s_t)/ln(241)` |
| `mf_top10bar_amount_share` | 最大 10 根 bar 金额之和/全日 |
| `mf_tail30_amount_z20` | 当日尾盘份额对 `[d-20,d-1]` 基线的 z 分（≥10 天） |
| `mf_amount_curve_rmse_w20` | 241 个逐 bar 份额对过去 20 日同 bar 基线的 RMSE |
| `mf_price_impact_b050` | `Σ|r_t| / sqrt(全日volume)` |
| `mf_return_amount_corr` | `corr(r_t, amount_t)` |
| `mf_five_minute_amount_excess_w20` | 49 个 5 分钟桶份额对同桶 20 日基线的最大超额 |

v1 的 45 个与 v2 的 63 个公式见上表链接的两个文档（各自逐因子带公式），不在此复制。**注意：v1 与 v2 的 manifest `formulas` 字段是占位符**，正式口径以 ANALYSIS.md / CANDIDATES_V2.md + Rust 源码为准；建议后续把 manifest 公式串补齐到与 core24 同级。

---

## 二、多头弱的实测证据

### 2.1 模型层（h1，top-80 vs bottom-80 日均超额）

| 实验 | IC | ICIR | top80 | bot80 | spread | 长短比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline（d60+m38） | 0.0327 | 0.27 | **+5.5bp** | −12.9bp | 18.3bp | 0.43 |
| plus-v2（+33 个 v2） | 0.0368 | 0.31 | **+4.5bp** | −16.9bp | 21.4bp | 0.27 |
| v2only（仅 33 个 v2） | 0.0308 | 0.27 | **+4.4bp** | −15.0bp | 19.4bp | 0.29 |

十分位剖面（bp/日，D1 最看空→D10 最看多）：

- baseline：`−12.9 −4.7 −2.6 −0.2 +2.1 +2.1 +3.3 +4.4 +4.1 +5.4`
- plus-v2：`−16.9 −4.3 −2.5 +1.6 +3.0 +3.0 +4.2 +4.2 +4.4 +4.4`

**解读**：排序单调性良好、加 v2 后整体 IC 和 spread 还变好——但增量几乎全部落在 D1（−12.9→−16.9），最看多一侧反而从 5.4bp 压平到 4.4bp。v2only 组合实盘口径 IR −0.12、净收益 15.8% 跑输中证500（25.7%），是"多头弱"的直接体现。有意思的是组合层 plus-v2 的有限换仓 80×5 版本净收益 80.6%（baseline 73.6%，IR 0.61 vs 0.54）——v2 因子通过"避开差票"改善了净结果，只是没有改善买什么。

### 2.2 因子层（|IC|≥0.02 的有效因子，按 IC 方向取"有利侧" vs "可避开的坑"）

| 因子集 | 有效因子数 | 负 IC 占比 | 有利侧均值 | 不利侧均值 | 长短比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| core24 | 6 | 50% | +1.99bp | +3.03bp | 0.66 |
| v1 | 20 | 55% | +0.25bp | +1.00bp | 0.25 |
| v2 | 26 | 73% | +0.75bp | +1.26bp | 0.60 |

v2 的 26 个有效因子里 19 个负 IC（风险/惩罚型：模糊性、潮汐、跳跃、振幅、草木皆兵…），它们识别"谁会输"；有利侧 ≥1bp 的 9 个是 `mf_logsig_v_l2_21`（top +2.5bp，v2 最强多头因子）、`mf_rv_jump_share_w20`（top +1.5bp，**唯一长短比>1 的因子**）、`mf_path_tortuosity_w20`、`mf_between_momentum_w20`、`mf_tail30_sharpe_ratio`、`mf_realized_hyperkurt_w20`、`mf_left_tail_cvar_ratio_w20`、`mf_logvol_intraday_std_w20`、`mf_logsig_v_l2_1`。

### 2.3 换期限救不了多头

h5 口径（超额=(个股5日−中证500 5日)/5）长短比：baseline 0.34、plus-v2 0.01、v2only 0.17——**h5 比 h1 更空头主导**。

---

## 三、诊断结论

1. **不是分钟因子数量问题。** 38→71 个分钟特征，IC 0.0327→0.0368、spread 18.3→21.4bp 都在改善；恶化的是结构：新增 v2 因子 73% 负 IC，边际信息全部堆在"识别差票"一侧。继续按当前家族堆分钟因子，多头不会变好（plus-v2 的 top80 已经从 5.5 掉到 4.5bp）。
2. **是因子类型结构问题，且有部分是结构性的。** 分钟 OHLCV 能表达的机制天然偏风险面：波动/偏度/跳跃/非流动性/反转，全部是"惩罚项"；而 A 股 1 日期限上短期强度本身是反转为主（core24 的 `tail30_return`、`close_to_vwap` 都是负 IC、靠 bottom 尾部赚钱）。真正含"谁在被买起来"信息的订单流/主动买入数据在当前数据边界外。
3. **多头信息目前主要在日频侧。** baseline 的 top80（+5.5bp）优于 plus-v2（+4.5bp），说明日频 60 个 alpha 族承担了大部分选"好票"的职能，分钟因子在组合里的最优角色当前更接近"剔除器+风险控制"。

## 四、还需要什么（按预期性价比排序）

### 4.1 比补因子更优先的三条结构性改法

1. **长短分离建模**：一个 winner 模型（只用正 IC/强度族特征 + 日频 alpha）、一个 loser 模型（全量特征）；买入 = winner 排名高 ∩ loser 排名不差。当前 LGBM 单模型把两端信息混在单调排名里，D10 必然被 D1 的梯度主导。
2. **改用"排除式"组合规则做对照**：十分位剖面显示 D4–D8 贡献了稳定正超额（+2~4bp），D10 并不显著优于 D8/D9。用预测剔除 D1–D3、在剩余票里等权或按日频强度加权，与现在的"买 D10"对照回测（现有 optimizer 直接可跑）。
3. **标签与中性化检查**：对 top80 做行业/市值中性化后的超额复算。若中性化后 top 侧回升，说明多头弱部分来自预测集中在高 beta/拥挤行业而非没有信息。

### 4.2 分钟侧值得补的因子（多头导向，OHLCV 内可算）

现有 132 个里正 IC 的"强势类"只有个位数。下一批应专门补**正 IC 候选**（预期多数在 h1 会偏反转，但值得系统性检验一次）：

| factor_id 建议 | 定义 | 机制 |
| --- | --- | --- |
| `mf_time_above_vwap` | 当日 `close_t > day_vwap` 的分钟占比 | 日内强势持续性 |
| `mf_close_loc_in_range` | `(C−L)/(H−L)` 收盘在日内区间的位置 | 收盘强度（日线版常用，分钟路径版更稳） |
| `mf_gap_hold` | 跳空方向与日内收益方向一致的天数占比（w5） | 缺口确认而非衰竭 |
| `mf_intraday_idio_momentum_w5` | 日内收益对横截面等权市场日内收益回归的日残差，5 日和 | 正交化日内动量 |
| `mf_updown_volume_ratio_w5` | 过去 5 日上涨日总量/下跌日总量 | 量确认的趋势 |
| `mf_upvol_confirmed_share` | 当日上涨且放量（量>当日均值）分钟的金额占比 | 量价配合的上行 |
| `mf_pullback_quality` | 5 日上涨后当日回撤深度 × 当日缩量程度 | 承接质量（回调不破） |
| `mf_new_high_proximity_20d` | `close_240 / max(close, 过去20日)` − 1 | 20 日新高接近度 |
| `mf_am_strong_pm_hold` | 上午强（top 三分位）时下午是否维持的 w20 频率 | 强势延续 |
| `mf_trend_capital_direction_w20` | 已在 CANDIDATES_V2 的 G2（趋势分钟量加权收益），v2 批次已算，**确认其进模型后方向与多头贡献** | 趋势资金方向 |

### 4.3 数据边界内的最后一档

若上述仍不够：与分钟湖可并的**日频资金面**（北向、融资、龙虎榜不需要 L2）是公开可得的"谁在买"数据，是订单流的低成本近似；以及把现有 smart_q 的**多头侧单独特征化**（smart_vwap 比值的上尾行为）。

---

## 附：复现本次数字的口径

- 标签：`_market_labels/csi300_csi500/panel` 的 `open_to_open_h1_raw`，超额 = raw − 中证500 同列；h5 超额 = (raw5 − bench5)/5。
- 分组：按 `pred_h1`/因子值当日截面排序，top/bottom 80（≈10%）与十分位；因子层用五分位。
- 逐因子 CSV：[ic_metrics/](ic_metrics/)（`factor_oos_*.csv` 为 h1 多空分组，`factor_multi_ic_*.csv` 为四周期 IC）。
- 组合层结果：`docs/experiments/optimizer-v2-limited-replacement.md` 与 `results/predict/*/limited-replacement-*/summary.json`。
