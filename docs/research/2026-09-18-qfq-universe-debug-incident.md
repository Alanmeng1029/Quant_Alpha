# QFQ universe debug incident — 2026-09-18

## Summary

The first Rust raw-universe control accidentally reintroduced `000937.SZ`, a symbol already excluded by the Python research pipeline because of a confirmed vendor adjustment anomaly.  The resulting model and backtest must not be used.

This was initially misdiagnosed as a difference between `index_trading_universe` and the new raw eligibility rules.  A full set audit and deterministic rerun disproved that diagnosis.

## What differed

`index_trading_universe` is the point-in-time index membership intersected with complete daily observations and a finite, positive vendor QFQ ratio.  The raw universe uses point-in-time membership plus positive raw OHLCV/amount and `complete_trading`, without requiring the vendor QFQ ratio.

Across 2018-01-02 through 2026-08-28, before joining the feature cache:

- legacy universe rows: 1,586,700;
- raw universe had 2,644 additional symbol-date memberships and no missing legacy memberships;
- all 2,644 additions had `validation_status=invalid_nonpositive` for the vendor adjustment;
- the additions covered 11 symbols from 2018-01-31 through 2021-03-23;
- the two universes were identical from 2022 onward.

The largest raw-only membership counts were `600188.SH` (706), `601919.SH` (693), `600039.SH` (414), `601225.SH` (375), and `600546.SH` (292).

Those 2,644 memberships had no rows in the old QFQ105 feature cache.  After the feature inner join, the corrected raw universe and `index_trading_universe` therefore produced the same 1,586,700-row training panel.

## Labels were not changed to raw

Both Rust modes calculate labels from `daily_qfq.qfq_open`:

- H1: T+1 open entry to T+2 open exit, less the CSI500 return;
- H5: T+1 open entry to T+6 open exit, less the CSI500 return.

The raw-universe name describes eligibility and factor inputs; it does not change the economic label to an unadjusted return.  In the pre-feature universe audit, QFQ-label coverage was effectively identical: H1 99.846% and H5 99.576% for raw, versus H1 99.846% and H5 99.576% for legacy.

## Root cause and proof

The invalid first control omitted the repository-wide `000937.SZ` exclusion.  It contained exactly one extra row on every one of the 2,080 panel dates.  Each 756-date model window consequently had exactly 756 extra training rows.  This changed LightGBM bagging, early stopping, trees, predictions, and the low-turnover portfolio path.

After restoring the exclusion:

- the 500+1000 panel fell from 2,997,521 to 2,995,441 rows, exactly 2,080 fewer;
- corrected raw and legacy QFQ105 training logs were identical for all 22 windows;
- saved LightGBM model SHA256 values matched;
- `predictions.parquet` matched byte-for-byte with SHA256 `5741b6f61333a0766302bb884451f5a53988db12f50faa4dcdc8c0904fa5f645`;
- the corrected pure-CSI500 backtests were numerically identical.

## Canonical policy

- Always exclude symbols listed in `configs/infeasible_execution_symbols.txt` before feature standardization, training, prediction, and portfolio construction.
- A universe ablation is valid only after comparing panel row counts, per-window training rows, model hashes, and prediction hashes.
- Do not infer a universe effect from backtest divergence until deterministic same-input reruns match.
- The invalid directories removed during cleanup were superseded by the corrected-v2 runs documented in `raw105-csi500-csi1000-corrected-v2.md`.
