"""Run the default-seed entity LSTM over every formal rolling OOS window."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl


ROOT = Path("results/predict/sequence-lstm-residual-raw125-v1")
OUTPUT = ROOT / "entity_embedding_full_v1"
QUARTERS = OUTPUT / "quarters"
WINDOWS = ROOT / "full/windows.json"
TRAINER = Path("/Users/alanmxy/大学/大学/研二上/论文/论文candidate/.venv-research/bin/python")
SEED = 20260908
REUSABLE = {
    "2025-04": ROOT / "entity_embedding_pilot_v2/month=2025-04",
    "2026-04": ROOT / "entity_embedding_pilot_v2/month=2026-04",
}


def run(command: list[object], python: Path = Path(sys.executable)) -> None:
    print("RUN", " ".join(map(str, [python, *command])), flush=True)
    env = {**os.environ, "PYTHONPATH": "src", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    subprocess.run([str(python), *map(str, command)], check=True, env=env)


windows = json.loads(WINDOWS.read_text())
sources = []
for window in windows:
    month = window["signal"][:7]
    if month in REUSABLE:
        source = REUSABLE[month]
    else:
        source = QUARTERS / f"month={month}"
        prediction = source / "paired_predictions.parquet"
        if not prediction.exists():
            run(["scripts/prepare_lstm_entity_metadata_v2.py", "--signal-month", month, "--output", source])
            run([
                "scripts/train_lstm_entity_embedding_pilot_v2.py",
                "--signal-month", month,
                "--entity-data", source / "entity_categories.parquet",
                "--output", source,
                "--seeds", SEED,
                "--models", "entity",
            ], TRAINER)
    sources.append((month, source))

frames = []
audits = []
for month, source in sources:
    frame = pl.read_parquet(source / "paired_predictions.parquet").filter(pl.col("seed") == SEED)
    required = {"trade_date", "ts_code", "entity_h1", "entity_h5"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing entity predictions in {source}")
    frames.append(frame.select(
        "trade_date", "ts_code",
        pl.col("entity_h1").alias("raw_h1"),
        pl.col("entity_h5").alias("raw_h5"),
    ))
    audits.append({"signal_month": month, **json.loads((source / "entity_categories.json").read_text())})

predictions = pl.concat(frames).sort("trade_date", "ts_code")
if predictions.select(pl.struct("trade_date", "ts_code").is_duplicated().any()).item():
    raise ValueError("duplicate rolling OOS predictions")
reference = pl.read_parquet(ROOT / "full/predictions/lgbm_default125_common.parquet").select(
    "trade_date", "ts_code", "execution_date")
predictions = predictions.join(reference, on=["trade_date", "ts_code"], how="inner")
OUTPUT.mkdir(parents=True, exist_ok=True)
predictions.write_parquet(OUTPUT / "predictions.parquet", compression="zstd")
(OUTPUT / "metadata_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2))
print(json.dumps({
    "seed": SEED,
    "windows": len(sources),
    "rows": predictions.height,
    "start": str(predictions["trade_date"].min()),
    "end": str(predictions["trade_date"].max()),
}, ensure_ascii=False), flush=True)
