# Raw factor production contract v2

Date: 2026-09-19.

## Fixed contract

- Every model feature is calculated from unadjusted OHLCV. Feature production must not read `daily_qfq`, `baostock_qfq_daily`, adjustment ratios, or historical `*_qfq_v1` factor artifacts.
- Economic labels and portfolio returns remain QFQ open-to-open. H1, H5 and H10 retain the existing CSI500 excess-return definition. A signal formed on T enters at T+1 and exits at T+2, T+6 or T+11 respectively; the shared label-maturity gap is eleven market sessions.
- One point-in-time, de-duplicated `CSI300 ∪ CSI500 ∪ CSI1000` factor dataset is produced. Cross-sectional factors use that full union. Training subsets only filter rows; they never recalculate factor values.
- Model A trains on `CSI300 ∪ CSI500`. Model B trains on CSI1000 after excluding date-level overlaps with CSI300 or CSI500.
- `000937.SZ` remains excluded before feature standardization, training, prediction, and portfolio construction.

The machine-readable definition is
[`configs/formal_factor_sets/o2o_raw_daily60_minute45_csi300_csi500_csi1000_v2.json`](../../configs/formal_factor_sets/o2o_raw_daily60_minute45_csi300_csi500_csi1000_v2.json).

## Runtime boundary

The 45 minute factors are already implemented in `quant-minute-factor` from raw minute bars. Their old CSI500+CSI1000 values remain useful as intermediate evidence, but the formal v2 dataset requires one rebuild to add CSI300-only rows and to make cross-sectional factors use the full three-index union. Subsequent runs are incremental and replay only the required warm-up state.

The 60 raw daily factors currently have a Python/Polars reference evaluator. This is an oracle, not the final production runtime. The production Rust implementation must emit one wide, year-partitioned dataset containing all 60 columns in a single pass. It must reuse shared rolling state and common intermediate columns rather than scan and write the panel sixty times.

## Required Rust daily-factor design

1. Load raw daily OHLCV and point-in-time three-index membership once per calendar block.
2. Replay the maximum required warm-up before each independently reproducible output block.
3. Calculate shared primitives (`returns`, `vwap`, delays, rolling moments, ranks and correlations) once and reuse them across formulas.
4. Advance recursive EMA/SMA state sequentially by market date; parallelize independent stocks and non-recursive formula groups inside a date.
5. Write one atomic Parquet file per year (or per date during incremental production), with `trade_date`, `ts_code`, membership flags and 60 raw factor columns.
6. Record source hashes, formula registry, universe policy, warm-up, row counts, exclusions and per-partition hashes in the manifest.
7. Compare every Rust column against the Python oracle before promotion. Key sets must match exactly; ordinary floating operations use a declared tolerance and recursive formulas require full-history replay checks.

## Preserved and retired artifacts

Historical QFQ factor Parquet files and `predict_features_o2o_daily60_minute45_v2` are retained as frozen research evidence. They are not valid upstream inputs for v2 production.

The two derived caches and prediction runs named `research-oos-qfq-daily60-minute45-*` were retired because they reused frozen QFQ daily factors while being interpreted as current-QFQ baselines.
