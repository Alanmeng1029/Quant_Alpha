# Rust 日频生产（125 因子）

生产入口是 `quant-production`。它按信号日依次完成：

1. 从原始日线计算 Rust Daily60；
2. 读取已有 Minute45 + DOS20，若目标日尚未生产则调用 Rust 分钟引擎增量计算；
3. 按冻结 manifest 顺序合并为 125 因子单日宽表；
4. 选择不晚于信号月的最新季度 H1、H5 LightGBM 模型并预测；
5. 用 `H1,(H5-H1)/4×4` 构造五期收益路径，分别求解单票上限 1% 和 5% 的
   无风险项、双边 2bps 成本感知 optimizer；
6. 按 80%/20% 合并两个袖套并净额化，生成下一交易日目标持仓。后续日期自动读取
   前一生产日两个袖套各自的目标权重，而不是把净额组合错误地当成单一袖套状态。

## 首次构建

```bash
cargo build --release \
  -p quant-daily-factor \
  -p quant-minute-factor \
  -p quant-lgbm-train --bins
```

## 每日命令

在仓库根目录执行：

```bash
target/release/quant-production run --date YYYY-MM-DD
```

首次开始真实持仓、没有任何历史生产仓位时必须显式使用：

```bash
target/release/quant-production run --date YYYY-MM-DD --start-flat
```

默认产物目录：

```text
results/production/raw_daily60_minute45_dos20_h1h5_blend_80_20_v1/date=YYYY-MM-DD/
├── daily60.parquet
├── daily60.manifest.json
├── factors.parquet
├── prediction_h1.parquet
├── prediction_h5.parquet
├── prediction.parquet
├── prediction_csi500.parquet
├── positions.parquet
├── positions.manifest.json
├── orders.csv
├── sleeves/
│   ├── core/       # 80%，CSI500，单票上限1%
│   ├── alpha/      # 20%，CSI300∪CSI500，单票上限5%
│   └── blend/      # 80/20净额目标
└── run_manifest.json
```

其中：

- `factors.parquet`：CSI300 ∪ CSI500 的 125 因子单日宽表；
- `prediction.parquet`：联合池的 `raw_h1/pred_h1` 与 `raw_h5/pred_h5`；
- `positions.parquet`：下一交易日 80%/20% 净额目标权重；
- `orders.csv`：默认按 1,000 万元资金、信号日收盘价和 100 股整手四舍五入生成的
  下一交易日估算买入计划；可用 `--capital` 和 `--lot-size` 调整；
- `run_manifest.json`：本次行数、实际选择的 H1/H5 模型、策略配置和上一生产日来源。

分钟原始分区、日线和指数成分必须先进入 DuckDB catalog。命令会验证目标日是已观测交易日，并使用 `A_stock_database/交易日历.csv` 确定尚未产生行情数据的下一交易日；同时拒绝未来模型、空因子表、重复键和错误因子数。

需要强制重算已经存在的分钟日文件时使用：

```bash
target/release/quant-production run --date YYYY-MM-DD --replace-minute
```

可以用 `--previous-production-dir /path/to/date=YYYY-MM-DD` 显式指定上一生产日；
未指定且不使用 `--start-flat` 时，会自动选择生产目录中日期早于信号日、同时包含
core/alpha 两个袖套状态的最近一期。
