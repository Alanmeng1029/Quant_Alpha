#!/bin/zsh
set -euo pipefail

cd /Users/alanmxy/Documents/Quant_Alpha
exec /usr/bin/time -l /Users/alanmxy/Documents/Quant_Alpha/target/release/quant-backtest factor-eval \
  --catalog A_stock_database/lake/catalog/a_share.duckdb \
  --factor A_stock_database/lake/derived/factors/gtja_alpha001_qfq/v1/factor.parquet \
  --config configs/factor_eval.yaml \
  --output results/performance_benchmarks \
  --run-id serial_8t_probe/csi300 \
  >> /private/tmp/quantalpha-serial-probe.log 2>&1
