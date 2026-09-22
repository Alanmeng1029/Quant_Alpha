"""Build matched 2026Q2 seed-20260908 prediction ledgers for serial backtests."""
from pathlib import Path

import polars as pl


ROOT = Path("results/predict/sequence-lstm-residual-raw125-v1")
PILOT = ROOT / "entity_embedding_pilot_v2/month=2026-04"
OUTPUT = PILOT / "backtest_comparison"

SEED = 20260908
paired = pl.read_parquet(PILOT / "paired_predictions.parquet").filter(pl.col("seed") == SEED)
keys = paired.select("trade_date", "ts_code").unique().sort("trade_date", "ts_code")
lgbm_all = pl.read_parquet(ROOT / "full/predictions/lgbm_default125_common.parquet")
lgbm = lgbm_all.join(keys, on=["trade_date", "ts_code"], how="semi").sort("trade_date", "ts_code")
if lgbm.height != keys.height:
    raise ValueError("LGBM does not cover every paired pilot sample")
execution = lgbm.select("trade_date", "ts_code", "execution_date")

models = {}
for model in ("baseline", "entity"):
    models[model] = (
        paired.select("trade_date", "ts_code",
                      pl.col(f"{model}_h1").alias("raw_h1"),
                      pl.col(f"{model}_h5").alias("raw_h5"))
        .join(execution, on=["trade_date", "ts_code"], how="inner")
        .sort("trade_date", "ts_code")
    )
models["lgbm"] = lgbm.select("trade_date", "ts_code", "raw_h1", "raw_h5", "execution_date")

core_reference = pl.read_parquet(
    "results/predict/dos-minute20-model-comparison-v1/daily60_minute45_dos20/csi500_predictions.parquet"
).select("trade_date", "ts_code")

for name, frame in models.items():
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(folder / "predictions.parquet", compression="zstd")
    core = frame.join(core_reference, on=["trade_date", "ts_code"], how="semi")
    core.write_parquet(folder / "csi500_predictions.parquet", compression="zstd")
    if frame.select("trade_date", "ts_code").n_unique() != keys.height:
        raise ValueError(f"{name} sample mismatch")
    if core.group_by("trade_date").len()["len"].min() < 98:
        raise ValueError(f"{name} has too few CSI500 names for the 1% sleeve")
    print(name, frame.height, core.height)
