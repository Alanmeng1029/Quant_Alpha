# Rust 日频生产（125 因子）

生产入口是 `quant-production`。它按信号日依次完成：

1. 从原始日线计算 Rust Daily60；
2. 读取已有 Minute45 + DOS20，若目标日尚未生产则调用 Rust 分钟引擎增量计算；
3. 按冻结 manifest 顺序合并为 125 因子单日宽表；
4. 选择不晚于信号月的最新季度 H1 LightGBM 模型并预测；
5. 自动读取此前最近一期目标持仓，按 2bps 双边成本生成下一交易日目标持仓。

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

默认产物目录：

```text
results/production/raw_daily60_minute45_dos20_h1_v1/date=YYYY-MM-DD/
├── daily60.parquet
├── daily60.manifest.json
├── factors.parquet
├── prediction.parquet
├── prediction.manifest.json
├── positions.parquet
├── positions.manifest.json
└── run_manifest.json
```

其中：

- `factors.parquet`：CSI300 ∪ CSI500 的 125 因子单日宽表；
- `prediction.parquet`：`raw_h1` 与横截面标准化后的 `pred_h1`；
- `positions.parquet`：下一交易日 CSI500 目标权重和对应的 `raw_h1`；
- `run_manifest.json`：本次行数、实际选择的模型和上一期持仓来源。

分钟原始分区、日线和指数成分必须先进入 DuckDB catalog。命令会验证目标日是已观测交易日，并使用 `A_stock_database/交易日历.csv` 确定尚未产生行情数据的下一交易日；同时拒绝未来模型、空因子表、重复键和错误因子数。

需要强制重算已经存在的分钟日文件时使用：

```bash
target/release/quant-production run --date YYYY-MM-DD --replace-minute
```

可以用 `--previous-positions /path/to/positions.parquet` 显式指定上一期持仓；未指定时会自动选择生产目录中日期早于信号日的最近一期 `positions.parquet`。
