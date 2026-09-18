# 正式 QFQ CSI500 基线

日期：2026-09-18

## 正式定义

当前正式日频因子口径固定为前复权（QFQ）。生产基线直接读取已经落盘并验证过的 60 个日频因子，不以当前公式代码重新生成的结果替换它们。

- 日频因子根目录：`A_stock_database/lake/derived/factors`
- 因子清单：`configs/candidate_factors_daily_o2o_candidate60.txt`
- 60 因子工件集合 SHA-256：`eb097bf9b74ee053f5338873696444dd94eb2ea667772f38c901072305347dd9`
- 价格口径：QFQ OHLC/VWAP；成交量保持原始成交量。
- 训练池：点时 CSI300 + CSI500。
- 组合池：点时 CSI500。
- 训练：Rust LightGBM；756 个交易日训练窗；6 日标签成熟间隔；22 个滚动窗口。
- 组合：CSI500 Top100，退出 Top120，每日最多替换 3 只，保留 2% 现金并计交易费用。

分钟 45 因子继续读取原正式缓存。本次基线确认不重算、不替换，也不把新的 CSI500+CSI1000 分钟数据集混入基线。

## Rust 复现结果

| 指标 | 结果 |
|---|---:|
| 区间 | 2021-04-02 至 2026-08-28 |
| 组合总收益 | 108.95% |
| CSI500 收益 | 25.69% |
| 超额复利收益 | 60.43% |
| 年化超额收益 | 9.16% |
| 超额 Sharpe（243） | 1.064455 |
| 超额最大回撤 | 14.89% |
| 日均买入换手 | 3.08% |

复现产物：

- 特征缓存：`A_stock_database/lake/derived/predict_features_o2o_daily60_minute45_v2`
- Rust 模型与预测：`results/predict/research-oos-qfq-old105-csi300-csi500-rust-raw-universe-v2`
- 完整 HTML：`results/predict/research-oos-qfq-old105-csi300-csi500-rust-raw-universe-v2/backtests/pure_csi500_top100_swap3/report.html`

## 版本边界

`QFQ` 只描述价格口径，不能替代因子工件版本。当前公式引擎重新计算出来的同名 QFQ 因子尚未与上述生产快照逐值一致，因此不得覆盖正式目录或作为旧基线的等价输入。

已确认的不一致来源包括：

- 冻结因子生成时，`daily_qfq` 来自 `daily_aggregated raw OHLC × snapshot=2026-08-28 vendor_qfq_ratio`；当前同名视图会在存在直接 BaoStock 前复权日线时优先使用后者。2025 年重叠样本的价格相对差异中位数约 0.176%，99% 分位约 3.01%，最大约 7.81%。
- Raw/QFQ 通用加载器曾把 QFQ 成交量条件从 `volume_share >= 0` 改为 `volume_share > 0`，导致停牌日从价格时间轴中消失，改变 delay、rolling 和递归状态。
- 部分 GTJA 公式包含递归 EMA。只读取252日 warm-up 后计算一天不能严格恢复从2018年开始累积的状态，需要完整历史或版本化状态检查点。

因此，日更必须读取独立、不可变、版本化的基线 QFQ 视图，不能继续依赖会随数据源优先级变化的 `daily_qfq` 动态视图。

CSI500+CSI1000 扩展必须从这个冻结基线出发：先保持 CSI500 侧输入与预测可复现，再单独补齐 CSI1000 日频因子。任何重算版本都需要新的因子版本号和独立回测，不得继续使用正式基线的名称。

## Rust 单日缓存对拍

使用 Rust `build_feature_cache` 对 2025-06-30 单独重建105因子缓存，并与正式历史缓存逐值比较：

| 检查项 | 结果 |
|---|---:|
| 正式缓存股票数 | 800 |
| Rust重建股票数 | 799 |
| 共同主键 | 799 |
| 日频因子完全一致列 | 60/60 |
| 分钟因子完全一致列 | 45/45 |
| 不一致单元格 | 0 |
| 最大绝对误差 | 0 |

一只股票的数量差异来自Rust生产链明确排除的异常证券 `000937.SZ`。该结果证明Rust缓存拼接和类型转换可以无损复现正式缓存；它不证明Rust已经实现日频60公式。当前日频60仍由Python/Polars生成，Rust只读取其Parquet结果。
