"""Prepare industry and point-in-time liquidity groups for a paired LSTM pilot."""
import argparse
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
CATALOG = Path("A_stock_database/lake/catalog/a_share.duckdb")

parser = argparse.ArgumentParser()
parser.add_argument("--signal-month", default="2025-04")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
OUTPUT = args.output or ROOT / f"entity_embedding_pilot_v2/month={args.signal_month}"

cache = SequenceCache(CACHE)
spec = next(x for x in json.loads(WINDOWS.read_text()) if x["signal"].startswith(args.signal_month))
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
        )
        SELECT e.row_index, e.trade_date, e.ts_code, e.day_index,
               i.industry, l.liquidity_20
        FROM endpoint e
        LEFT JOIN instruments i USING(ts_code)
        LEFT JOIN liquidity l USING(trade_date, ts_code)
        ORDER BY e.row_index
    """).arrow())
finally:
    connection.close()
if attributes.height != endpoint.height:
    raise ValueError("entity attributes do not align one-to-one with endpoints")

count = pl.col("liquidity_20").count().over("trade_date")
rank = pl.col("liquidity_20").rank("ordinal").over("trade_date")
attributes = attributes.with_columns(
    pl.when(pl.col("liquidity_20").is_not_null() & (count > 0))
      .then((((rank - 1) * 10 / count).floor() + 1).clip(1, 10))
      .otherwise(0).cast(pl.Int64).alias("liquidity_group"))

training = attributes.filter(pl.col("day_index").is_in(train_days.tolist()))
industry_counts = training.drop_nulls("industry").group_by("industry").agg(
    pl.len().alias("rows"), pl.col("ts_code").n_unique().alias("stocks"))
kept = sorted(industry_counts.filter((pl.col("rows") >= 1000) & (pl.col("stocks") >= 5))
              ["industry"].to_list())
# 0 is learnable UNKNOWN, 1 is learnable OTHER, retained industries start at 2.
industry_map = {name: i + 2 for i, name in enumerate(kept)}
attributes = attributes.with_columns(
    pl.when(pl.col("industry").is_null()).then(0)
      .otherwise(pl.col("industry").replace_strict(industry_map, default=1))
      .cast(pl.Int64).alias("industry_id"))
OUTPUT.mkdir(parents=True, exist_ok=True)
attributes.select("row_index", "industry_id", "liquidity_group").write_parquet(
    OUTPUT / "entity_categories.parquet", compression="zstd")
audit = {
    "industry_embedding_count": len(kept) + 2,
    "retained_industries": len(kept),
    "industry_unknown_rate": float(attributes["industry_id"].eq(0).mean()),
    "industry_other_rate": float(attributes["industry_id"].eq(1).mean()),
    "liquidity_unknown_rate": float(attributes["liquidity_group"].eq(0).mean()),
    "industry_rule": "retain training categories with >=1000 rows and >=5 stocks; pool remainder as OTHER",
    "industry_limitation": "static instruments.industry; not point-in-time industry history",
    "liquidity_definition": "point-in-time 20-row trailing mean amount_cny, daily union-universe decile",
    "endpoint_codes": attributes["ts_code"].n_unique(),
}
(OUTPUT / "entity_categories.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps(audit, ensure_ascii=False, indent=2))
