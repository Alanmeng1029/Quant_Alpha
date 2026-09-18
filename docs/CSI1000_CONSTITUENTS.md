# CSI1000 historical constituents

The local data lake contains point-in-time CSI1000 (`000852.SH`) membership from
2018 onward. The source is the `csi1000.txt` interval file in the
`chenditc/investment_data` Qlib release dated 2026-09-17.

The importer converts inclusive source intervals into month-end snapshots and
merges them with the existing CSI300 and CSI500 annual constituent Parquet files:

```bash
PYTHONPATH=src:. python scripts/import_chenditc_csi1000.py \
  --source external/investment_data/extracted/qlib_bin/instruments/csi1000.txt \
  --data-root A_stock_database \
  --start 2018-01-01
PYTHONPATH=src python -m a_share_data.cli build-catalog --data-root A_stock_database
```

Current coverage is 2018-01-31 through 2026-09-17, comprising 105 snapshots and
105,012 membership rows. Twelve snapshots from 2022-12-30 through 2023-11-30
contain 1,001 members. This source anomaly is intentionally preserved. All other
snapshots contain exactly 1,000 members. The importer records
`source_name=chenditc/investment_data` and
`source_method=qlib_interval_month_end_snapshot`, validates primary-key
uniqueness, and backs up every annual Parquet file before replacement.

Query snapshots through `index_monthly_constituents`; query the intersection
with the valid daily trading universe through `index_trading_universe`.
