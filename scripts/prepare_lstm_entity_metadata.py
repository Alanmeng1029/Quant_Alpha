"""Prepare point-in-time categorical endpoint metadata for the LSTM pilot."""
from datetime import date
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from a_share_data.sequence_data import SequenceCache


ROOT = Path("results/predict/sequence-lstm-residual-raw125-v1")
CACHE = ROOT / "sequence_cache/7b65e9b210f84533"
WINDOWS = ROOT / "full/windows.json"
OUTPUT = ROOT / "entity_embedding_pilot/month=2025-04"
CATALOG = Path("A_stock_database/lake/catalog/a_share.duckdb")
MARKET_CAP = Path("A_stock_database/lake/derived/risk_inputs/csi500_akshare_monthly_float_mv/akshare_daily_market_cap.parquet")


cache = SequenceCache(CACHE)
spec = next(x for x in json.loads(WINDOWS.read_text()) if x["signal"].startswith("2025-04"))
dates = cache.metadata.select("trade_date", "day_index").unique().sort("day_index")
day_by_date = dict(dates.iter_rows())
train_days = np.asarray([day_by_date[date.fromisoformat(x)] for x in spec["train_dates"]], np.int32)
test_days = np.asarray([day_by_date[date.fromisoformat(x)] for x in spec["test_dates"]], np.int32)
ids = np.unique(np.concatenate((cache.sample_ids_for_days(train_days), cache.sample_ids_for_days(test_days))))
ends = np.asarray(cache.sample_rows[ids], dtype=np.int64)
endpoint = cache.metadata.filter(pl.col("row_index").is_in(ends)).select(
    "row_index", "trade_date", "ts_code", "day_index")
first, last = endpoint["trade_date"].min(), endpoint["trade_date"].max()
connection = duckdb.connect(str(CATALOG), read_only=True)
try:
    connection.register("endpoint", endpoint.to_arrow())
    attributes = pl.from_arrow(connection.execute(f"""
        WITH liquidity AS (
          SELECT trade_date, ts_code,
                 avg(amount_cny) OVER (
                   PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                 ) AS liquidity_20
          FROM daily_aggregated
          WHERE trade_date BETWEEN DATE '{first}' - INTERVAL 40 DAY AND DATE '{last}'
        ), cap AS (
          SELECT CAST(trade_date AS DATE) trade_date, ts_code, float_mv
          FROM read_parquet('{MARKET_CAP}')
          WHERE CAST(trade_date AS DATE) BETWEEN DATE '{first}' AND DATE '{last}'
        )
        SELECT e.row_index, e.trade_date, e.ts_code, e.day_index,
               i.industry, c.float_mv, l.liquidity_20
        FROM endpoint e
        LEFT JOIN instruments i USING(ts_code)
        LEFT JOIN cap c USING(trade_date, ts_code)
        LEFT JOIN liquidity l USING(trade_date, ts_code)
        ORDER BY e.row_index
    """).arrow())
finally:
    connection.close()
if attributes.height != endpoint.height:
    raise ValueError("entity attributes do not align one-to-one with sequence endpoints")


def decile(column: str) -> pl.Expr:
    count = pl.col(column).count().over("trade_date")
    rank = pl.col(column).rank("ordinal").over("trade_date")
    return (pl.when(pl.col(column).is_not_null() & (count > 0))
            .then((((rank - 1) * 10 / count).floor() + 1).clip(1, 10))
            .otherwise(0).cast(pl.Int64))


attributes = attributes.with_columns(decile("float_mv").alias("size_group"),
                                     decile("liquidity_20").alias("liquidity_group"))
industries = sorted(x for x in attributes.filter(pl.col("day_index").is_in(train_days.tolist()))
                    ["industry"].drop_nulls().unique().to_list())
industry_map = {name: i + 1 for i, name in enumerate(industries)}
attributes = attributes.with_columns(
    pl.col("industry").replace_strict(industry_map, default=0).cast(pl.Int64).alias("industry_id"))
OUTPUT.mkdir(parents=True, exist_ok=True)
attributes.select("row_index", "industry_id", "size_group", "liquidity_group").write_parquet(
    OUTPUT / "entity_categories.parquet", compression="zstd")
audit = {
    "industry_categories": len(industries),
    "industry_unknown_rate": float(attributes["industry_id"].eq(0).mean()),
    "size_unknown_rate": float(attributes["size_group"].eq(0).mean()),
    "liquidity_unknown_rate": float(attributes["liquidity_group"].eq(0).mean()),
    "size_source": str(MARKET_CAP),
    "liquidity_definition": "20-row trailing mean daily amount_cny including signal date",
    "industry_limitation": "static instruments.industry; not point-in-time industry history",
    "endpoint_codes": attributes["ts_code"].n_unique(),
}
(OUTPUT / "entity_categories.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps(audit, ensure_ascii=False, indent=2))
