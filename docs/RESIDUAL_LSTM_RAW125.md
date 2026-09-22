# 当前125因子上的残差LSTM复现

候选配置：`configs/prediction_sequence_lstm_residual_raw125_v1.json`。

沿用105因子残差模型的架构和超参数：20日序列、128/64隐藏层、循环层残差、零初始化的输出残差、dropout=0.1、Adam学习率3e-5、weight decay=1e-4、batch size=512、梯度裁剪1、最多30 epoch、MSE验证早停patience=4、随机种子20260908。每个窗口重新初始化，在内层验证中选择epoch数，再在完整756日训练集上从头训练相同轮数。

输入改为当前正式原始价格125因子（60日频＋45分钟＋20 DOS），缺失指示使实际输入为250通道。正式因子注册表中补全了三个既有分钟因子清单路径，清单顺序已与缓存125列逐列核对，因子集合本身未改动。

新标签使用原始行情完整性筛选的沪深300＋中证500联合池、前复权开盘超额收益，与当前125因子LGBM一致。外层756日训练窗口与样本外之间留11日，与当前H1/H5/H10 LGBM的训练截止日一致；LSTM只预测H1/H5，内层仍用687日拟合＋6日隔离＋63日验证。旧105配置的默认6日外层间隔保持不变。

序列缓存会剔除历史不足20个连续交易日的样本，因此公平比较使用LSTM和LGBM的相同键集合。当前运行样本覆盖率约98.76%。相较旧105因子LSTM，除新增20因子外，日频因子价格口径及外层截止日也变了，不能将结果变化全部解释为新增因子的贡献。

## 命令

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/quant-alpha-mpl \
  /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data.predict sequence-oos \
  --config configs/prediction_sequence_lstm_residual_raw125_v1.json --prepare-only

PYTHONPATH=src MPLCONFIGDIR=/tmp/quant-alpha-mpl PYTORCH_ENABLE_MPS_FALLBACK=0 \
  /Users/alanmxy/anaconda3/envs/ml311/bin/python -m a_share_data.predict sequence-oos \
  --config configs/prediction_sequence_lstm_residual_raw125_v1.json --skip-backtests

MPLCONFIGDIR=/tmp/quant-alpha-mpl /Users/alanmxy/anaconda3/envs/ml311/bin/python \
  scripts/evaluate_residual_lstm125.py
```

训练器使用配置中的独立PyTorch环境，并要求可用的Apple MPS GPU，不进行CPU回退。沙箱无法访问GPU时，需要获准在正常主机环境运行训练命令。

`--skip-backtests` 跳过旧的Top100/swap3评估，随后评估脚本使用当前五期优化器。四次目标生成和两次回测严格串行。复用旧报告实际子组合口径：80%核心为中证500、单票上限1%；20%集中组合为联合池、单票上限5%；98%投入，买卖各2bp，100股整手。

## 产物与恢复

输出根目录 `results/predict/sequence-lstm-residual-raw125-v1/`：

- `sequence_cache/`：独立的125因子缓存，身份包含标签股票池与数据指纹。
- `full/lstm/quarters/month=*/`：逐季checkpoint、预测、训练曲线、完成标记和SHA256。
- `full/predictions/lstm125.parquet`：合并预测；`lgbm_default125_common.parquet` 为共同样本基线。
- `full/report.json`：IC、分年/分季度、预测相关性与配对差异评估。
- `full/multiperiod_comparison/report_offline.html`：单文件离线成本后对照，含图表和年度/季度表。

训练命令可在相同配置、代码、数据和Python环境下恢复。完成的窗口先校验身份和文件哈希，再复用；未完成窗口从头训练。不要在同一个输出目录下修改参数或训练代码后强行续跑。后续评估脚本也复用已完成阶段，更换模型后应使用新目录。

本任务不切换正式模型、不删除旧105因子LSTM产物，也不恢复已回退的IC/MSE早停实验。

## 本次完成结果（2026-09-21）

22个季度全部完成，均使用MPS，最佳轮数1～7、中位数3。原始预测1,028,291行；末日798行因缺少下一交易日不进入执行评估。全部季度checkpoint和预测哈希校验通过。共同样本、同组合规则下：

| 指标 | 残差LSTM125 | LGBM125 |
|---|---:|---:|
| H1 Rank IC | 0.029379 | 0.037942 |
| H5 Rank IC | 0.035793 | 0.035092 |
| 净年化收益 | 17.76% | 21.16% |
| 净值最大回撤 | 29.76% | 27.44% |
| 超额年化（243日） | 12.17% | 14.50% |
| 超额Sharpe（243日） | 1.560 | 1.403 |
| 超额最大回撤 | 11.79% | 14.68% |
| 日均买入换手 | 44.22% | 42.91% |

20日分块、2000次配对bootstrap：H1 Rank IC差异（LSTM减LGBM）为-0.008563，95%区间[-0.015847,-0.002693]；H5差异0.000701，区间[-0.014864,0.016926]。因此H1落后，H5没有明确胜出证据。

2025Q2的LSTM H1/H5 Rank IC为0.036697/0.057975，高于LGBM的0.018235/0.021974；但该季净收益4.34%低于7.98%，日均买入换手39.04%对1.79%。恢复信号差异及换手并未自动带来更高收益。本次结果不足以支持直接替换正式LGBM。

离线报告已包含嵌入式图表、6年及22季度的双模型表格。当前LGBM共同样本回测与旧完整样本正式报告不同，应使用本次共同样本结果做模型比较。
