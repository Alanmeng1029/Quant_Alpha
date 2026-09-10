# 当前分钟因子完整公示表（132 个 × 多周期 OOS Rank IC）

生成日期：2026-09-08；2026-09-09 重建（原文件被误删，指标数据未变）。IC 为 **OOS 2021-04-01 ~ 2026-08-28** 日均 Spearman Rank IC；标签为 open-to-open 前瞻 h1/h5/h10/h20 收益（CSI300∪CSI500 截面，逐周期剔除 null 标签）。ICIR 为 h1 的 mean(IC)/std(IC)。逐因子原始数据存 [ic_metrics/](ic_metrics/)。
标记：★=正式集 `o2o_daily60_minute38_v1` 的分钟成分；▲=`plus-v2`/`v2only` 实验入选；⚠=**parquet 整列为空，未实现或实现缺陷**（见文末）。公式为压缩口径，逐字口径：core24 见数据集 `manifest.json`，v1 见 [ANALYSIS.md](ANALYSIS.md)，v2 见 [CANDIDATES_V2.md](CANDIDATES_V2.md)。

符号：`r_t=ln(close_t/close_{t−1})`；`w20`=严格过去 20 交易日均值；RV=Σr²；金额优先于股数。

## 一、core24（24 个）

| factor_id | 公式（压缩） | IC h1 | IC h5 | IC h10 | IC h20 | ICIR h1 | 用途 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mf_realized_kurtosis` | 分钟收益峰度m4/m2² | -0.0286 | -0.0392 | -0.0431 | -0.0523 | -0.2214 | ★正式 |
| `mf_max_intraday_drawdown` | min_t(close_t/running_max−1) | +0.0269 | +0.0439 | +0.0459 | +0.0584 | +0.1307 |  |
| `mf_tail30_return` | ln(close_240/close_210) | -0.0268 | -0.0161 | -0.0150 | -0.0135 | -0.2177 | ★正式 |
| `mf_realized_volatility` | sqrt(Σr_t²) | -0.0266 | -0.0315 | -0.0324 | -0.0423 | -0.1654 | ★正式 |
| `mf_close_to_vwap` | close_240/日vwap−1;日vwap=Σamount/Σvolume | -0.0251 | -0.0070 | -0.0032 | -0.0048 | -0.1448 | ★正式 |
| `mf_downside_semivariance_ratio` | Σmin(r,0)²/Σr² | +0.0223 | +0.0124 | +0.0099 | +0.0124 | +0.1697 | ★正式 |
| `mf_tail30_vwap_to_day_vwap` | 尾30vwap/日vwap−1 | -0.0167 | -0.0016 | +0.0026 | +0.0010 | -0.0952 |  |
| `mf_return_amount_corr` | corr(r_t,amount_t) | -0.0154 | -0.0030 | +0.0009 | -0.0007 | -0.1182 | ★正式 |
| `mf_realized_skewness` | 分钟收益偏度m3/m2^1.5 | -0.0154 | -0.0107 | -0.0095 | -0.0125 | -0.1449 | ★正式 |
| `mf_amount_curve_rmse_w20` | 逐bar份额对同bar20日基线RMSE | +0.0150 | +0.0244 | +0.0280 | +0.0397 | +0.0953 |  |
| `mf_smart_q_b050_w10_p20` | 同b010,β=0.50 | +0.0131 | -0.0001 | -0.0004 | -0.0015 | +0.1014 | ★正式 |
| `mf_price_impact_b050` | Σ∣r∣/sqrt(全日volume) | -0.0130 | -0.0127 | -0.0147 | -0.0197 | -0.0853 |  |
| `mf_smart_q_b025_w10_p20` | 同b010,β=0.25 | +0.0123 | -0.0006 | -0.0008 | -0.0002 | +0.0915 |  |
| `mf_smart_q_b025_w10_p15` | 同b025,量阈值15% | +0.0123 | -0.0011 | -0.0013 | -0.0005 | +0.0916 |  |
| `mf_smart_q_logv_w10_p20` | 同b010,排序键=|r|/ln(1+volume) | +0.0116 | -0.0006 | -0.0005 | +0.0012 | +0.0860 |  |
| `mf_smart_q_b010_w10_p20` | Q=S_d/mean(S,[d-9,d]);S=smart_vwap/日vwap;smart集=按|r|/v^0.10降序累计至当日量20% | +0.0116 | -0.0004 | -0.0003 | +0.0010 | +0.0860 |  |
| `mf_trend_efficiency` | ∣C240−C0∣/Σ∣Δclose∣ | -0.0115 | -0.0163 | -0.0188 | -0.0205 | -0.1008 |  |
| `mf_amount_entropy` | −Σs·ln(s)/ln(241) | -0.0102 | -0.0151 | -0.0194 | -0.0289 | -0.0718 |  |
| `mf_amount_hhi` | Σs_t²,s=分钟金额份额 | +0.0086 | +0.0129 | +0.0173 | +0.0258 | +0.0644 |  |
| `mf_tail30_amount_z20` | 尾盘份额对20日基线z分 | -0.0076 | -0.0052 | -0.0045 | -0.0041 | -0.0893 | ★正式 |
| `mf_top10bar_amount_share` | 最大10bar金额/全日 | +0.0074 | +0.0111 | +0.0152 | +0.0233 | +0.0581 |  |
| `mf_five_minute_amount_excess_w20` | 49桶份额对同桶20日基线最大超额 | -0.0038 | -0.0006 | +0.0005 | +0.0051 | -0.0421 | ★正式 |
| `mf_tail30_amount_share` | Σamount(211..240)/全日amount | +0.0014 | +0.0074 | +0.0097 | +0.0162 | +0.0144 |  |
| `mf_pm_minus_am_return` | ln(C240/C120)−ln(C120/C0) | -0.0001 | -0.0034 | -0.0027 | -0.0024 | -0.0007 |  |

## 二、ohlcv_candidates_v1（45 个）

| factor_id | 公式（压缩） | IC h1 | IC h5 | IC h10 | IC h20 | ICIR h1 | 用途 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mf_volume_up_positive_return_std` | −std(放量且r>0分钟) | +0.0375 | +0.0462 | +0.0493 | +0.0621 | +0.2296 | ★正式 |
| `mf_cpv_dvolume_lead_dprice_mm_w20` | mean_20(corr(ΔV_t,ΔP_{t+1}∣−,−)) | -0.0370 | -0.0528 | -0.0604 | -0.0767 | -0.1896 | ★正式 |
| `mf_cpv_dvolume_lead_dprice_pm_w20` | mean_20(corr(ΔV_t,ΔP_{t+1}∣+,−)) | +0.0350 | +0.0519 | +0.0584 | +0.0725 | +0.1913 | ★正式 |
| `mf_cpv_dprice_dvolume_mp_w20` | mean_20(corr(ΔP,ΔV∣ΔP<0,ΔV>0)) | +0.0348 | +0.0509 | +0.0592 | +0.0756 | +0.1777 | ★正式 |
| `mf_volume_up_return_std` | 放量分钟收益std取负(v2复算) | +0.0347 | +0.0441 | +0.0461 | +0.0590 | +0.2150 | ★正式 |
| `mf_segment_return_dispersion` | 8段收益std | -0.0336 | -0.0430 | -0.0446 | -0.0574 | -0.1918 | ★正式 |
| `mf_cpv_dvolume_lead_dprice_pp_w20` | mean_20(corr(ΔV_t,ΔP_{t+1}∣+,+)) | -0.0331 | -0.0488 | -0.0554 | -0.0709 | -0.1881 | ★正式 |
| `mf_high_price_vol_ratio_w20` | mean_20(最高价20%分钟RV/全天RV) | -0.0324 | -0.0413 | -0.0449 | -0.0549 | -0.1611 | ★正式 |
| `mf_cpv_dprice_dvolume_mm_w20` | mean_20(corr(ΔP,ΔV∣ΔP<0,ΔV<0)) | -0.0320 | -0.0413 | -0.0473 | -0.0610 | -0.1966 | ★正式 |
| `mf_cpv_dvolume_lead_dprice_mp_w20` | mean_20(corr(ΔV_t,ΔP_{t+1}∣−,+)) | +0.0317 | +0.0466 | +0.0540 | +0.0679 | +0.1927 | ★正式 |
| `mf_cpv_dprice_dvolume_pp_w20` | mean_20(corr(ΔP,ΔV∣ΔP>0,ΔV>0)) | -0.0315 | -0.0457 | -0.0520 | -0.0664 | -0.1897 | ★正式 |
| `mf_chip_return_bin_std` | 收益分箱金额份额std | +0.0311 | +0.0404 | +0.0419 | +0.0526 | +0.1698 | ★正式 |
| `mf_tail30_open30_rv_ratio` | log(尾30RV/首30RV) | +0.0300 | +0.0394 | +0.0445 | +0.0581 | +0.1864 | ★正式 |
| `mf_chip_top3_return_bin_share` | Top3收益箱份额(v2复算) | +0.0275 | +0.0376 | +0.0393 | +0.0490 | +0.1607 | ★正式 |
| `mf_smart_q_pooled_b025_w10_p20` | 10日分钟池按∣r∣/v^0.25选前20%量的vwap比 | -0.0270 | -0.0372 | -0.0409 | -0.0508 | -0.2129 | ★正式 |
| `mf_chip_return_bin_skewness` | 收益箱份额偏度(v2复算) | +0.0256 | +0.0333 | +0.0343 | +0.0432 | +0.1591 | ★正式 |
| `mf_chip_return_bin_kurtosis` | 收益箱份额峰度 | +0.0243 | +0.0316 | +0.0325 | +0.0408 | +0.1582 | ★正式 |
| `mf_chip_return_q80` | 金额加权累计收益80%分位 | -0.0239 | -0.0150 | -0.0121 | -0.0145 | -0.1540 | ★正式 |
| `mf_cpv_dprice_dvolume_pm_w20` | mean_20(corr(ΔP,ΔV∣ΔP>0,ΔV<0)) | +0.0222 | +0.0314 | +0.0354 | +0.0459 | +0.1789 | ★正式 |
| `mf_pm_am_rv_ratio` | log(下午RV/上午RV) | +0.0219 | +0.0333 | +0.0388 | +0.0518 | +0.1465 | ★正式 |
| `mf_tail30_open30_amihud_ratio` | 尾30/首30 Amihud | +0.0183 | +0.0193 | +0.0224 | +0.0286 | +0.1886 | ★正式 |
| `mf_amihud_intraday_w20` | Amihud(v2复算) | +0.0182 | +0.0298 | +0.0370 | +0.0514 | +0.1041 | ★正式 |
| `mf_cpv_price_volume_level_w20` | mean_20(corr(close,volume)) | -0.0180 | -0.0187 | -0.0178 | -0.0203 | -0.1191 | ★正式 |
| `mf_cpv_segment_std_w20` | mean_20(8段corr(P,V)的std) | -0.0176 | -0.0178 | -0.0175 | -0.0211 | -0.1405 |  |
| `mf_minute_amount_seasonality_return_corr_w20` | mean_20(corr(r,份额/20日同时刻均值)) | -0.0154 | -0.0030 | +0.0006 | -0.0013 | -0.1304 | ★正式 |
| `mf_return_damount_corr_w20` | mean_20(corr(r,Δamount)) | -0.0137 | -0.0151 | -0.0182 | -0.0253 | -0.0976 | ★正式 |
| `mf_return_volume_corr_w20` | mean_20(corr(r,volume)) | -0.0124 | -0.0117 | -0.0142 | -0.0211 | -0.0872 |  |
| `mf_return_amount_corr_w20` | mean_20(corr(r,amount)) | -0.0123 | -0.0117 | -0.0142 | -0.0211 | -0.0869 |  |
| `mf_amount_periodicity_peak_share_w20` | 去季节性DFT主峰功率占比,再平滑 | -0.0100 | -0.0094 | -0.0118 | -0.0178 | -0.1200 |  |
| `mf_open30_amount_share` | 首30min金额/全日 | -0.0096 | -0.0200 | -0.0206 | -0.0261 | -0.0847 | ★正式 |
| `mf_tail30_amount_to_segment_median` | 尾30金额/中位段 | -0.0080 | -0.0056 | -0.0050 | -0.0009 | -0.0963 | ★正式 |
| `mf_tail30_day_amihud_ratio` | 尾30/全天Amihud | +0.0078 | +0.0056 | +0.0055 | +0.0055 | +0.1216 | ★正式 |
| `mf_am_pm_amount_ratio` | log(上午/下午金额) | -0.0066 | -0.0147 | -0.0180 | -0.0232 | -0.0598 |  |
| `mf_open30_tail30_amount_ratio` | log(尾30/首30金额) | -0.0065 | -0.0172 | -0.0188 | -0.0263 | -0.0582 |  |
| `mf_cpv_segment_last30_w20` | mean_20(末段corr(P,V)) | -0.0065 | -0.0101 | -0.0152 | -0.0181 | -0.0702 |  |
| `mf_amount_share_kurtosis` | 当日金额份额峰度 | +0.0052 | +0.0068 | +0.0111 | +0.0154 | +0.0630 |  |
| `mf_amount_share_skewness` | 当日金额份额偏度 | +0.0048 | +0.0064 | +0.0106 | +0.0154 | +0.0540 |  |
| `mf_tail30_rv_z20` | 尾盘RV z20 | +0.0040 | +0.0020 | +0.0017 | +0.0020 | +0.0510 |  |
| `mf_low_price_vol_ratio_w20` | mean_20(最低价20%分钟RV/全天RV) | +0.0028 | -0.0035 | -0.0078 | -0.0156 | +0.0194 |  |
| `mf_price_dvolume_corr_w20` | mean_20(corr(close,Δvolume)) | +0.0023 | +0.0021 | -0.0021 | -0.0051 | +0.0268 |  |
| `mf_amount_periodicity_band_power_w20` | 同频谱中频带功率占比,再平滑 | +0.0016 | +0.0002 | +0.0014 | +0.0036 | +0.0249 |  |
| `mf_open30_amount_z20` | 开盘份额z20(v2复算) | +0.0016 | -0.0030 | -0.0019 | -0.0031 | +0.0173 |  |
| `mf_segment_return_lag1_autocorr` | 8段收益一阶自相关 | +0.0015 | +0.0005 | +0.0005 | +0.0017 | +0.0170 |  |
| `mf_tail30_minus_open30_ra_corr` | 尾30−首30 corr(r,amount) | -0.0013 | -0.0060 | -0.0072 | -0.0080 | -0.0124 |  |
| `mf_tail30_minus_open30_return` | r尾30−r首30 | +0.0012 | -0.0007 | -0.0039 | -0.0014 | +0.0078 |  |

## 三、ohlcv_candidates_v2（63 个）

| factor_id | 公式（压缩） | IC h1 | IC h5 | IC h10 | IC h20 | ICIR h1 | 用途 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mf_amplitude_game_w20` | 按5min收益排序的振幅cumsum差,mean_20 | -0.0388 | -0.0513 | -0.0567 | -0.0704 | -0.1818 | ▲plus-v2 |
| `mf_tide_strong_rate_w20` | 主量峰涨潮速率,mean_20 | -0.0382 | -0.0520 | -0.0602 | -0.0748 | -0.2173 | ▲plus-v2 |
| `mf_minute_ols_qrs_w20` | 50bar低~高回归βz末值×R̄²,mean_20 | -0.0379 | -0.0535 | -0.0608 | -0.0781 | -0.1827 | ▲plus-v2 |
| `mf_ambiguity_amount_corr_w20` | corr(模糊性,金额),mean_20 | -0.0375 | -0.0544 | -0.0622 | -0.0799 | -0.1792 | ▲plus-v2 |
| `mf_ambiguity_amount_ratio_w20` | 高模糊分钟金额比,mean_20 | -0.0367 | -0.0505 | -0.0566 | -0.0728 | -0.2162 | ▲plus-v2 |
| `mf_ideal_amplitude_d20` | 20日高价5日振幅−低价5日振幅 | -0.0366 | -0.0535 | -0.0637 | -0.0860 | -0.1840 | ▲plus-v2 |
| `mf_volume_up_return_std` | 放量分钟收益std取负(v2复算) | +0.0365 | +0.0529 | +0.0606 | +0.0782 | +0.1793 | ★正式 |
| `mf_tide_weak_std_w20` | 次级峰退潮速率,std_20 | -0.0360 | -0.0458 | -0.0522 | -0.0634 | -0.1994 | ▲plus-v2 |
| `mf_left_tail_mean_w20` | mean_20(最差5%分钟收益均值) | +0.0338 | +0.0493 | +0.0567 | +0.0740 | +0.1704 | ▲plus-v2 |
| `mf_jump_taylor_residual_w20` | Σ[2(r_pct−r_log)−r_log²],mean_20 | -0.0337 | -0.0485 | -0.0536 | -0.0654 | -0.1759 | ▲plus-v2 |
| `mf_ambiguity_price_gap_w20` | 高模糊分钟金额比−股数比,mean_20 | -0.0309 | -0.0412 | -0.0444 | -0.0518 | -0.1848 | ▲plus-v2 |
| `mf_logsig_v_l2_21` | 量路径签名L2(2,1)分量 | +0.0308 | +0.0404 | +0.0448 | +0.0563 | +0.1906 | ▲plus-v2 |
| `mf_rv_jump_share_w20` | mean_20(max(RV−BPV,0)/RV) | +0.0307 | +0.0444 | +0.0518 | +0.0678 | +0.1399 | ▲plus-v2 |
| `mf_path_tortuosity_w20` | log(路径长/净位移),mean_20 | +0.0287 | +0.0383 | +0.0440 | +0.0584 | +0.1507 | ▲plus-v2 |
| `mf_chip_return_bin_skewness` | 收益箱份额偏度(v2复算) | +0.0280 | +0.0356 | +0.0400 | +0.0508 | +0.1612 | ★正式 |
| `mf_dazzling_vol_w20` | 激增分钟后5bar收益std,mean_20 | -0.0280 | -0.0421 | -0.0480 | -0.0612 | -0.1525 | ▲plus-v2 |
| `mf_between_momentum_w20` | ln(C209/C30)去头尾动量,mean_20 | -0.0278 | -0.0370 | -0.0406 | -0.0530 | -0.1230 | ▲plus-v2 |
| `mf_realized_hyperkurt_w20` | mean_20(Σr⁶/RV³),5min降采样 | -0.0274 | -0.0359 | -0.0401 | -0.0491 | -0.2095 | ▲plus-v2 |
| `mf_vol_diff_ols_intercept_w20` | 同一回归截距t绝对值合成,mean_20 | -0.0273 | -0.0366 | -0.0392 | -0.0485 | -0.1912 | ▲plus-v2 |
| `mf_tail30_sharpe_ratio` | r尾30/√Σr²尾30 | -0.0271 | -0.0167 | -0.0165 | -0.0151 | -0.2323 | ▲plus-v2 |
| `mf_logsig_v_l2_1` | 量路径签名L1:ΣΔln(amount) | +0.0262 | +0.0390 | +0.0431 | +0.0565 | +0.1565 | ▲plus-v2 |
| `mf_left_tail_cvar_ratio_w20` | mean_20(最差5%分钟Σr²/RV) | -0.0260 | -0.0347 | -0.0403 | -0.0529 | -0.1521 |  |
| `mf_pm_open30_to_am_open30_vol_ratio` | mean_20(log(下午开盘30min/上午开盘30min金额)) | +0.0232 | +0.0381 | +0.0438 | +0.0538 | +0.1526 | ▲plus-v2 |
| `mf_dazzling_ret_w20` | 激增分钟收益均值,mean_20 | -0.0229 | -0.0307 | -0.0358 | -0.0464 | -0.1479 | ▲plus-v2 |
| `mf_vol_diff_ols_tstd_w20` | r对Δv滞后OLS斜率t值std,mean_20 | -0.0222 | -0.0343 | -0.0406 | -0.0484 | -0.2268 | ▲plus-v2 |
| `mf_logvol_intraday_std_w20` | mean_20(std(ln(5min金额))) | -0.0202 | -0.0343 | -0.0412 | -0.0546 | -0.1149 | ▲plus-v2 |
| `mf_direction_changes_w20` | sign(Δclose)变号率,mean_20 | -0.0197 | -0.0213 | -0.0228 | -0.0283 | -0.1197 |  |
| `mf_ideal_turnover_d20` | 同框架对换手率切割 | -0.0186 | -0.0298 | -0.0348 | -0.0410 | -0.1443 | ▲plus-v2 |
| `mf_realized_skew_5m_w20` | mean_20(5min收益偏度) | -0.0174 | -0.0218 | -0.0238 | -0.0289 | -0.1275 | ▲plus-v2 |
| `mf_realized_hyperskew_w20` | mean_20(Σr⁵/RV^2.5),5min降采样 | -0.0161 | -0.0204 | -0.0217 | -0.0252 | -0.1420 | ▲plus-v2 |
| `mf_volume_game_return_w20` | 按5min收益排序的量cumsum差,mean_20 | -0.0145 | -0.0204 | -0.0235 | -0.0315 | -0.0935 | ▲plus-v2 |
| `mf_tail_volume_weighted_return` | 尾30量加权收益 | -0.0143 | -0.0077 | -0.0068 | -0.0071 | -0.1646 | ▲plus-v2 |
| `mf_upper_shadow_minute_share` | 上影分钟占比 | -0.0137 | -0.0165 | -0.0207 | -0.0319 | -0.0942 |  |
| `mf_intraday_momentum_sum_w5` | Σ_5日ln(close/open) | -0.0125 | -0.0073 | -0.0110 | -0.0113 | -0.0700 |  |
| `mf_last3bar_auction_share` | 末3bar金额/全日,mean_20 | -0.0121 | -0.0060 | -0.0061 | +0.0001 | -0.1482 | ▲plus-v2 |
| `mf_volume_game_position_w20` | 按日内位置排序的量cumsum差,mean_20 | -0.0119 | -0.0150 | -0.0139 | -0.0188 | -0.0894 | ▲plus-v2 |
| `mf_doc_pdf80` | 价格比分组金额80%分位组序 | +0.0117 | +0.0165 | +0.0218 | +0.0328 | +0.0736 |  |
| `mf_reversal_hour_weighted_d20` | Σ_dΣ_t(t/240)·r_{d,t}/Σw | -0.0105 | -0.0194 | -0.0294 | -0.0413 | -0.0679 | ▲plus-v2 |
| `mf_apm_tstat_w20` | 上下午残差差t统计量 | -0.0086 | -0.0105 | -0.0104 | -0.0168 | -0.0540 |  |
| `mf_chip_top3_return_bin_share` | Top3收益箱份额(v2复算) | -0.0059 | -0.0093 | -0.0127 | -0.0185 | -0.0354 | ★正式 |
| `mf_minute_ols_r2_mean_w20` | 50bar corr²(low,high)均值,mean_20 | -0.0057 | -0.0086 | -0.0105 | -0.0150 | -0.0487 |  |
| `mf_bottomvol_ret_w20` | Bottom-50量分钟收益连乘,mean_20 | -0.0044 | -0.0099 | -0.0092 | -0.0061 | -0.0597 | ▲plus-v2 |
| `mf_firstbar_auction_share` | 首bar金额/全日,mean_20 | +0.0035 | -0.0036 | -0.0033 | -0.0053 | +0.0367 |  |
| `mf_climb_cov_w20` | cov(r/波动,波动)∣高波动bar,mean_20 | +0.0032 | +0.0080 | +0.0098 | +0.0101 | +0.0491 | ▲plus-v2 |
| `mf_amihud_intraday_w20` | Amihud(v2复算) | +0.0032 | +0.0111 | +0.0143 | +0.0205 | +0.0257 | ★正式 |
| `mf_close_vs_poc` | (close−POC)/(日高−日低) | -0.0017 | +0.0092 | +0.0113 | +0.0124 | -0.0123 |  |
| `mf_realized_skew_pos_w20` | mean_20(Σ_{r>0}r³/Σr³) | -0.0012 | +0.0001 | +0.0012 | +0.0010 | -0.0305 |  |
| `mf_realized_skew_neg_w20` | mean_20(Σ_{r<0}r³/Σr³) | +0.0012 | -0.0001 | -0.0012 | -0.0010 | +0.0305 |  |
| `mf_id_vov_w20` | std_20(特质RV) | — | — | — | — | — | ⚠ 全空列 |
| `mf_left_tail_uncertainty_w20` | std_20(左尾均值) | — | — | — | — | — | ⚠ 全空列 |
| `mf_overnight_gap_sum_w5` | Σ_5日ln(open/prev_close) | — | — | — | — | — | ⚠ 全空列 |
| `mf_smart_q_5m_b025_w10_p20` | smart_q的5min版 | — | — | — | — | — | ⚠ 全空列 |
| `mf_open30_amount_z20` | 开盘份额z20(v2复算) | — | — | — | — | — | ⚠ 全空列 |
| `mf_upvol_share_stability_w20` | −std_20(上行分钟金额占比) | — | — | — | — | — | ⚠ 全空列 |
| `mf_moderate_risk_composite` | z(耀眼波动率)+z(耀眼收益率) | — | — | — | — | — | ⚠ 全空列 |
| `mf_reversal_amplitude_w20` | 振幅×跳跃方向,mean_20 | — | — | — | — | — | ⚠ 全空列 |
| `mf_volume_follow_ratio_w20` | 量峰后5min量占比,mean_20 | — | — | — | — | — | ⚠ 全空列 |
| `mf_volume_sync_corr_w20` | 布林状态量份额与市场corr,mean_20 | — | — | — | — | — | ⚠ 全空列 |
| `mf_panic_dispersion_w20` | 惊恐度+衰减项,mean_20 | — | — | — | — | — | ⚠ 全空列 |
| `mf_trend_capital_vwap_gap` | 趋势分钟vwap/全日vwap−1 | — | — | — | — | — | ⚠ 全空列 |
| `mf_trend_capital_ret_w20` | 趋势分钟量加权收益,mean_20 | — | — | — | — | — | ⚠ 全空列 |
| `mf_herd_follow_ratio_w20` | 极端跟随量/趋势量,mean_20 | — | — | — | — | — | ⚠ 全空列 |
| `mf_vol_range1min_std_w20` | std(high/low),mean_20 | — | — | — | — | — | ⚠ 全空列 |

## 附：⚠ 全空列清单（15 个，candidates_v2.rs 实现缺陷）

- `mf_id_vov_w20`
- `mf_left_tail_uncertainty_w20`
- `mf_overnight_gap_sum_w5`
- `mf_smart_q_5m_b025_w10_p20`
- `mf_open30_amount_z20`
- `mf_upvol_share_stability_w20`
- `mf_moderate_risk_composite`
- `mf_reversal_amplitude_w20`
- `mf_volume_follow_ratio_w20`
- `mf_volume_sync_corr_w20`
- `mf_panic_dispersion_w20`
- `mf_trend_capital_vwap_gap`
- `mf_trend_capital_ret_w20`
- `mf_herd_follow_ratio_w20`
- `mf_vol_range1min_std_w20`

这些槽位在 `raw()`/`finalize()` 中未被赋值（如 `mf_overnight_gap_sum_w5` 需日线前收盘价、`mf_open30_amount_z20` 的 v2 复算缺少当日原子量、G 族/协同类需全市场横截面），导致整列 null、被筛选静默跳过。修复前请勿将它们计入'63 个 v2 因子'的可用数量——**实际可用 v2 因子为 48 个**。

## 附：各集 h1 IC 符号结构

- core24：正 IC 11 个，负 IC 13 个
- v1：正 IC 24 个，负 IC 21 个
- v2：正 IC 13 个，负 IC 35 个

## 附：多周期读法速览

- **负 IC 随周期增强**（h1→h20 绝对值放大）：amplitude_game、tide_strong_rate、minute_ols_qrs、ambiguity 族、ideal_amplitude、realized_kurtosis——风险/惩罚类，长周期更空头主导。
- **仅 h1 有效**（h5 起衰减到零）：tail30_return、close_to_vwap、smart_q 族——日内反转类。
- **稀缺的多周期正 IC**（多头候选种子）：max_intraday_drawdown、amount_curve_rmse、rv_jump_share、logsig_v_l2_21、path_tortuosity、left_tail_mean 等。