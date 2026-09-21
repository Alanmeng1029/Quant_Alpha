# DolphinDB 分钟K线因子 Rust 复现（dos_minute_v1）

`2.分钟K线因子/` 下 31 个 DolphinDB 脚本的 Rust 复现，作为 `quant-minute-factor`
的新 factor set `dos_minute_v1`（31 列宽表，`crates/quant-minute-factor/src/dos_candidates.rs`）。
每个脚本产出一个 股票×交易日 的日频因子；复用既有 loader / MarketContext /
plan_blocks / writer / manifest 基建，输出格式与 ohlcv_candidates_v1/v3 相同
（`<output>/year=YYYY/YYYY-MM-DD.parquet` + `manifest.json` + `quality_report.json`）。

运行：

```bash
cargo run -p quant-minute-factor -- build \
  --catalog A_stock_database/lake/catalog/a_share.duckdb \
  --minute-root A_stock_database/lake/canonical/minute \
  --output <out> --start ... --end ... \
  --factor-set dos_minute_v1 --threads-per-job 1 --jobs 4 \
  --index-codes 000300.SH,000905.SH,000852.SH
```

`--threads-per-job 1` 为强制项：因子 5/8/10/29 消费当日横截面信息，块内必须串行
（与 ohlcv_candidates_v3 同一模式；块间并行不受影响，块布局不变性有测试保证）。

## 列映射与时间窗

| DolphinDB | Rust (canonical lake) | 说明 |
|---|---|---|
| `LastPx` | `close` | |
| `OpenPx` | `open` | |
| `HighPx` | `high` | loader 为此扩展了 high/low 投影（canonical 摄取本就含这两列） |
| `LowPx` | `low` | 同上 |
| `Volume` | `volume_share` | 股数 |
| `Amount` | `amount_cny` | |
| 因子 1 的 `TradeMoney` | `close * volume_share` | 忠实脚本原式，不用 amount_cny |

时间窗（minute_index 已确认：0=09:30、120=11:30、121=13:01、237=14:57、240=15:00）：

- 标准因子（28 个）：`minute_index 0..=237`，对应脚本模板的 `09:30–14:57` 过滤；
- 因子 10/12/19 用 `endTime=15:00`：全 241 根；其中 10/19 再按
  `having rank(TradeTime) between 5:239` 截取 index 5..=239 做回归样本
  （脚本作者的意图是让 5 阶滞后无 null；实际 t=5 仍缺 `ΔV[t-5]`，与 DolphinDB
  ols 一致地按行剔除 null，生效样本为 6..=239）。

股票池沿用现有 point-in-time 指数成分 universe（默认 CSI300∪CSI500∪CSI1000），
不复制脚本 `00%/30%/6%` 前缀粗筛（该过滤是数据卫生目的，canonical 摄取校验已覆盖）。
横截面计算（因子 8 的分钟排名、5/10/29 的截面回归与缩放）使用当日全部加载股票，
与 DolphinDB 日分区 chunk = 全市场口径一致；输出仅保留 universe 成员。

## DolphinDB 语义对齐

- `std`/`mstd` = 样本标准差（n−1）；`covar` = 总体协方差（÷n）；`corr` = Pearson（丢 null 对，要求两侧方差 >0 且 ≥2 对）。
- `kurtosis`（因子 8）按有偏超额口径 `m4/m2² − 3` 实现。
- `percentChange(x)` = `(x−prev)/prev`（返回小数，不乘 100）。
- `mavg/mstd/msum(x,n)` 无 minPeriods 时前 n−1 行为部分窗口；`mstd(x,n,n)`（minPeriods=n）不满窗口为 null。跨日窗口均为行基准（row-based）VecDeque：停牌日窗口压缩、上市不足 20 天出部分窗口值——与 DolphinDB 分区内行语义一致，**不同于** core24 的日历对齐 Ring（后者刻意"留洞"）。
- `rank(x, percent=true)` = min-tie 1-based 排名 ÷ 非空计数，取值 (0,1]（因子 8 的分钟截面排名、10/19 的 `rank(TradeTime)`，后者在严格递增序列上等价于 0-based 行号）。
- `aggrTopN(first, valueCol, sortCol, top, ascending)`：按 sortCol 排序取 top 行、返回 valueCol。并列时取行序最早者（脚本未定义 tie 行为，此处为确定性选择）。
- `ols(Y, X, intercept=true)`：含截距；Y/X 含 null 的行剔除。t 值 = β/√(σ²·(X'X)⁻¹_jj)，F = (SSR/k)/(SSE/(n−k−1))。实现上先对每个回归元做 rms 缩放（t/F 不变）以改善条件数；SSE=0（完美拟合）时 t/F 无穷，输出 null。
- `cdfStudent(240, ·)` 经正则化不完全 Beta（Lentz 连分式）实现，与 scipy `t.cdf` 对拍 <1e-9；`cdfNormal` 经 A&S 7.1.26 erf（|误差| ≤1.5e-7）。

## 与脚本的已记录偏差

1. **跨股票边界的 `deltas`/`percentChange`/`prev`（脚本 bug，已修正）**：
   因子 2 的 `deltas(LastPx)` 与因子 16/17/18 的 `percentChange`/`prev(Amount)`
   在 `group by` 投影前对整个日分区（全部股票）求值，首根 K 线会取到**上一只股票**
   的末根价格/成交额。Rust 按 (股票, 日) 分组重算，首根为 null。
2. **跨日同时股票的滞后（保留）**：因子 3/13 的输入按 SecurityID 重分区，
   首根 K 线的收益/滞后量取**前一交易日 14:57 根**（同一股票），这是脚本的合法
   语义且因果上成立，予以保留（State 中保存前日末根信息）。
3. **因子 23 不截断（保留）**：`(ret−0.1)/0.2` 不做 [0,1] 截断，实测值域约
   [−0.56, −0.32]（分钟收益 ≪0.1 所致）——与脚本一致，属"均匀分布 CDF"的
   未完成形态，保留原式。
4. **因子 30 的 `last()` quirk（保留）**：`dailyLastPx = last(LastPx)` 在按价
   升序的同价分组表上取值 = 当日最高分钟收盘价（非时序收盘价；且窗口止于 14:57）。
   Rust 复刻该语义（`vsaRatio` = 支撑区域下限 − 当日最高分钟收盘价）。
5. **因子 5 以代码为准**：文件头写"残差均值"，代码为 `mavg(residual, 20)`
   （含当日、部分窗口、跳过 null 行），按代码实现。
6. **DolphinDB 无法本地对拍**：一致性靠公式忠实翻译 + 单元测试 +
   Python 独立复算抽样对拍（见下）。

## 因子清单（dos 函数名 → Rust 列名 → 公式）

| # | DolphinDB | Rust 列 | 公式 |
|---|---|---|---|
| 1 | illiqShortCut | mf_dos_illiq_shortcut | Σ(2(H−L)−\|O−C\|)/(close·volume) |
| 2 | positiveConsistVolume | mf_dos_positive_consist_volume | Σ(vol \| \|C−O\|≤0.5\|H−L\| 且 C>前根C)/Σvol（deltas 分组修正） |
| 3 | corrRetLagAdjAmount | mf_dos_corr_ret_lag_adj_amount | corr(\|log1p(ret)\|, prev((amt−μ)/σ))，μ/σ=同分钟前20日 |
| 4 | volTideRatio | mf_dos_vol_tide_ratio | 9根中心邻域量和的峰/前后谷价格变速 |
| 5 | fallCenterDev | mf_dos_fall_center_dev | 截面 OLS(down_g~up_g) 残差的 20日 mavg |
| 6 | consistVolume | mf_dos_consist_volume | Σ(vol \| \|C−O\|≤0.5\|H−L\|)/Σvol |
| 7 | volPropEntropy | mf_dos_vol_prop_entropy | Σ−p·ln p，p=分钟金额占比 |
| 8 | PATV | mf_dos_patv | 分钟截面%rank 的 mean/std + kurtosis |
| 9 | flashVolatility | mf_dos_flash_volatility | ΔVol>μ+σ 时刻的前瞻5根收益std均值 |
| 10 | NoonCanopyAlpha | mf_dos_noon_canopy_alpha | sign(F>截面均值F)·\|截距t\|，OLS(ret~ΔV及1–5阶滞后) |
| 11 | singleVolPropEntropy | mf_dos_single_vol_prop_entropy | Σ−p·ln p，p=量占比×价占比 |
| 12 | resilienceCov | mf_dos_resilience_cov | covar(ret/adjVol, adjVol)，adjVol=(20维OHLC窗口 std/mean)² |
| 13 | corrLagRetAdjAmount | mf_dos_corr_lag_ret_adj_amount | corr(prev(\|log1p(ret)\|), (amt−μ)/σ) |
| 14 | flashReturns | mf_dos_flash_returns | ΔVol>μ+σ 时刻的当期收益均值 |
| 15 | peakClimbingCov | mf_dos_peak_climbing_cov | 同12但限 adjVol>μ+σ |
| 16 | corrRetAmount | mf_dos_corr_ret_amount | corr(\|log1p(ret)\|, amount) |
| 17 | corrRetLagAmount | mf_dos_corr_ret_lag_amount | corr(\|log1p(ret)\|, prev(amount))，首根null |
| 18 | corrLagRetAmount | mf_dos_corr_lag_ret_amount | corr(prev(\|log1p(ret)\|), amount)，首根null |
| 19 | DawnFogVolPersist | mf_dos_dawn_fog_vol_persist | 5个滞后ΔV回归系数t值的std |
| 20 | propTDis | mf_dos_prop_t_dis | Σamt·cdfStudent(240, ret/std(ret))/Σamt |
| 21 | propNormalDis | mf_dos_prop_normal_dis | Σamt·Φ(ret·19.6)/Σamt |
| 22 | propNaiveAct | mf_dos_prop_naive_act | Σamt·cdfStudent(240, ΔC/std(ΔC))/Σamt |
| 23 | propUniformDis | mf_dos_prop_uniform_dis | Σamt·((ret−0.1)/0.2)/Σamt，不截断 |
| 24 | volumePeakCount | mf_dos_volume_peak_count | 超阈值量棒间隔>1分钟的计数（午休91分钟计为间隔） |
| 25 | ratioFuzzinessAmount | mf_dos_ratio_fuzziness_amount | 起雾时刻金额均值/当日金额均值 |
| 26 | ratioFuzzinessVolume | mf_dos_ratio_fuzziness_volume | 起雾时刻量均值/当日量均值 |
| 27 | pDisVol | mf_dos_p_dis_vol | 50%量支撑区域下限 − 当日最高 |
| 28 | bDisVol | mf_dos_b_dis_vol | 支撑区域上限 − 当日最低 |
| 29 | adjFuzzinessDiff | mf_dos_adj_fuzziness_diff | (量比−金额比)，负值经10日std归一后再经截面 s1/s2 缩放 |
| 30 | vsaRatio | mf_dos_vsa_ratio | 支撑区域下限 − 当日最高分钟收盘（last() quirk） |
| 31 | ratioFuzzinessAmtCorr | mf_dos_ratio_fuzziness_amt_corr | corr(fuzziness, amount) |

fuzziness = `mstd(mstd(ret,5,5),5,5)`（minPeriods 满5，前9根为 null）；
"起雾时刻" = fuzziness > 当日 fuzziness 均值。

## 状态与因果

- 纯当日因子：多数（表中标"截面"者除外）只用当日分钟数据。
- 因子 3/13：per-stock 20 日金额曲线 VecDeque（严格先验日，同分钟 μ/σ）。
- 因子 5：当日截面残差 + per-stock 20 日残差窗口（含当日，跳过 null）。
- 因子 29：当日 stage-1 值 + per-stock 10 日窗口（含当日）+ 当日截面缩放。
- warmup 25 个交易日（最大跨日窗口 20 + 余量）；warmup 日不写文件、断点续跑、
  块布局/线程数不影响输出（均有测试）。

## 验证记录（2026-09-19）

- `cargo test -p quant-minute-factor`：48 通过（30 既有 + 18 新增；含 CDF 对拍
  scipy 参考值、OLS t/F 一阶复算、熵/支撑区域/潮汐/波峰计数手算用例、
  fuzziness 双实现对照、块布局不变性集成测试）。
- **core24 回归**：loader 增加 high/low 后，2026-06-01..19 真实数据 25,186 个
  股票日 × 24 因子，新旧输出逐值零差异。
- **dos_minute_v1 真实数据试跑**（2026-06-01..19）：31 因子覆盖率 ≥99.2%，
  值域符合构造约束（相关类 ∈[−1,1]、consist ≤1、pDis ≤0、bDis ≥0、熵 ≤ln238）。
- **Python 独立复算对拍**（000001.SZ @ 2026-06-18）：illiq_shortcut、
  vol_prop_entropy、consist_volume、positive_consist_volume、prop_uniform_dis、
  volume_peak_count、vol_tide_ratio、prop_t_dis（对拍 scipy t.cdf）8 个因子
  逐位一致。

已知未对拍项：DolphinDB 服务端输出（本机不可用）；因子 8 的 kurtosis 口径与
`aggrTopN` 并列 tie 行为按文档/确定性规则选定，若日后与 DolphinDB 实盘输出
对拍出现系统性差异，优先核查这两处。
