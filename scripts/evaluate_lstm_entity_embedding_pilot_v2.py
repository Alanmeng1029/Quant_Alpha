"""Evaluate the paired 2025Q2 entity-embedding pilot on formal OOS labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from a_share_data.predict import build_labels
from a_share_data.sequence_train import FactorLSTM
from train_lstm_entity_embedding_pilot_v2 import EntityConditionedLSTM


def model_metrics(frame: pl.DataFrame, model: str) -> dict:
    result = {}
    for horizon in ("h1", "h5"):
        pred = f"{model}_{horizon}"
        label = f"excess_{horizon}"
        daily = (
            frame.select("trade_date", pred, label)
            .drop_nulls()
            .group_by("trade_date")
            .agg(pl.corr(pl.col(pred).rank(), pl.col(label).rank()).alias("rank_ic"))
            .sort("trade_date")
        )
        result[horizon] = {
            "days": daily.height,
            "mean_rank_ic": float(daily["rank_ic"].mean()),
            "positive_ratio": float((daily["rank_ic"] > 0).mean()),
            "daily_rank_ic": daily.to_dicts(),
        }
    return result


def main(args) -> None:
    predictions = pl.read_parquet(args.output / "paired_predictions.parquet")
    assert predictions.height > 0
    prediction_columns = [f"{model}_{horizon}" for model in ("baseline", "entity") for horizon in ("h1", "h5")]
    assert predictions.select(pl.col(prediction_columns).is_finite().all()).row(0) == (True,) * 4
    audit = json.loads((args.output / "entity_categories.json").read_text())

    labels = build_labels(
        args.catalog,
        start=str(predictions["trade_date"].min()),
        end=str(predictions["trade_date"].max()),
        universe_index_codes=("000300.SH", "000905.SH"),
        raw_eligible_universe=True,
    ).select("trade_date", "ts_code", "excess_h1", "excess_h5")
    lgbm = pl.read_parquet(args.lgbm).select(
        "trade_date", "ts_code", pl.col("pred_h1").alias("lgbm_h1"), pl.col("pred_h5").alias("lgbm_h5"))
    joined = (predictions.drop("target_h1", "target_h5")
              .join(labels, on=["trade_date", "ts_code"], how="left")
              .join(lgbm, on=["trade_date", "ts_code"], how="left"))

    runs = []
    for seed in sorted(joined["seed"].unique().to_list()):
        sample = joined.filter(pl.col("seed") == seed)
        baseline = model_metrics(sample, "baseline")
        entity = model_metrics(sample, "entity")
        lgbm_metrics = model_metrics(sample, "lgbm")
        delta = {h: entity[h]["mean_rank_ic"] - baseline[h]["mean_rank_ic"] for h in ("h1", "h5")}
        runs.append({"seed": seed, "baseline": baseline, "entity": entity, "lgbm": lgbm_metrics,
                     "entity_minus_baseline": delta})

    compact_runs = []
    for run in runs:
        compact_runs.append({
            "seed": run["seed"],
            "baseline": {h: {k: v for k, v in run["baseline"][h].items() if k != "daily_rank_ic"} for h in ("h1", "h5")},
            "entity": {h: {k: v for k, v in run["entity"][h].items() if k != "daily_rank_ic"} for h in ("h1", "h5")},
            "lgbm": {h: {k: v for k, v in run["lgbm"][h].items() if k != "daily_rank_ic"} for h in ("h1", "h5")},
            "entity_minus_baseline": run["entity_minus_baseline"],
        })
    aggregate = {
        "baseline": {h: float(np.mean([r["baseline"][h]["mean_rank_ic"] for r in runs])) for h in ("h1", "h5")},
        "entity": {h: float(np.mean([r["entity"][h]["mean_rank_ic"] for r in runs])) for h in ("h1", "h5")},
        "lgbm": {h: float(np.mean([r["lgbm"][h]["mean_rank_ic"] for r in runs])) for h in ("h1", "h5")},
        "entity_minus_baseline": {h: float(np.mean([r["entity_minus_baseline"][h] for r in runs])) for h in ("h1", "h5")},
    }

    channels = 250
    for seed in sorted(joined["seed"].unique().to_list()):
        baseline = FactorLSTM(channels, hidden=(128, 64), dropout=.1, recurrent_residual=True,
                              output_residual=True, zero_init_output_residual=True)
        entity = EntityConditionedLSTM(channels, audit["industry_embedding_count"])
        baseline.load_state_dict(torch.load(args.output / f"baseline_seed{seed}.pt", map_location="cpu", weights_only=True))
        entity.load_state_dict(torch.load(args.output / f"entity_seed{seed}.pt", map_location="cpu", weights_only=True))

    report = {
        "signal_month": args.signal_month,
        "label_definition": "formal build_labels; CSI300+CSI500 raw eligible universe; executable open-to-open excess labels",
        "prediction_rows": joined.height,
        "rows_with_h1_label": joined["excess_h1"].is_not_null().sum(),
        "rows_with_h5_label": joined["excess_h5"].is_not_null().sum(),
        "checkpoint_count": 6,
        "runs": compact_runs,
        "aggregate": aggregate,
    }
    (args.output / "formal_label_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    root = Path("results/predict/sequence-lstm-residual-raw125-v1/entity_embedding_pilot_v2/month=2025-04")
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("A_stock_database/lake/catalog/a_share.duckdb"))
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--signal-month", default="2025-04")
    parser.add_argument("--lgbm", type=Path, default=Path("results/predict/sequence-lstm-residual-raw125-v1/full/predictions/lgbm_default125_common.parquet"))
    main(parser.parse_args())
