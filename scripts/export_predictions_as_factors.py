#!/usr/bin/env python3
"""Materialize model predictions in the standard long factor artifact format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SIGNALS = {
    "h1": pl.col("raw_h1"),
    "h5_dailyized": pl.col("raw_h5") / 5.0,
    "h1h5_50_50": 0.5 * pl.col("raw_h1") + 0.1 * pl.col("raw_h5"),
    "h1h5_25_75": 0.25 * pl.col("raw_h1") + 0.15 * pl.col("raw_h5"),
    "h1h5_10_90": 0.10 * pl.col("raw_h1") + 0.18 * pl.col("raw_h5"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True,
                        help="MODEL=path/to/predictions.parquet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for item in args.prediction:
        model, source_text = item.split("=", 1)
        source_path = Path(source_text)
        source = pl.read_parquet(source_path).with_columns(pl.col("trade_date").cast(pl.Date))
        for signal, expression in SIGNALS.items():
            factor_id = f"prediction_{model}_{signal}"
            output = args.output / factor_id
            output.mkdir(parents=True, exist_ok=True)
            factor = (source.select("trade_date", "ts_code", expression.alias("factor_value"))
                      .filter(pl.col("factor_value").is_finite()))
            factor.write_parquet(output / "factor.parquet", compression="zstd")
            manifest = {
                "factor_id": factor_id,
                "source_prediction": str(source_path.resolve()),
                "signal": signal,
                "rows": factor.height,
                "definition": str(expression),
                "artifact_format": "standard_long_factor_v1",
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": len(args.prediction),
                      "signals_per_model": len(SIGNALS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
