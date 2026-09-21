#!/usr/bin/env python3
"""Refit one frozen LSTM candidate and measure a true quarterly OOS Rank IC."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from a_share_data.sequence_data import SequenceCache
from a_share_data.sequence_train import _epoch, _targets, require_mps
from tune_sequence_lstm import CandidateLSTM


def rank_ic(frame: pl.DataFrame, prediction: str, target: str) -> dict[str, float | int]:
    daily = (frame.filter(pl.col(prediction).is_finite() & pl.col(target).is_finite())
             .group_by("trade_date")
             .agg(pl.corr(prediction, target, method="spearman").alias("ic"))
             .drop_nulls("ic"))
    return {"days": daily.height, "mean_rank_ic": float(daily["ic"].mean()),
            "positive_ratio": float((daily["ic"] > 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--old-lstm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    cache = SequenceCache(args.cache)
    window = json.loads(args.windows.read_text())[0]
    dates = cache.metadata.select("trade_date", "day_index").unique().sort("day_index")
    day_by_date = dict(dates.iter_rows())
    train_days = np.asarray([day_by_date[date.fromisoformat(value)]
                             for value in window["train_dates"]], np.int32)
    test_days = np.asarray([day_by_date[date.fromisoformat(value)]
                            for value in window["test_dates"]], np.int32)
    full_ids = cache.sample_ids_for_days(train_days)
    test_ids = cache.sample_ids_for_days(test_days)
    ends = np.asarray(cache.sample_rows[full_ids], dtype=np.int64)
    full_ids = full_ids[np.isfinite(np.asarray(cache.targets[ends])).any(axis=1)]
    full_y, full_mask, means, scales = _targets(cache, full_ids, winsorize=True)

    device = require_mps()
    torch.manual_seed(args.seed)
    model = CandidateLSTM(cache.factor_count * 2, [128, 64], .10,
                          recurrent_residual=True, output_residual=True,
                          zero_init_output_residual=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=1e-4, eps=1e-8)
    training = []
    for epoch in range(1, args.epochs + 1):
        loss = _epoch(cache, full_ids, full_y, full_mask, model, device,
                      args.batch_size, optimizer, args.seed + 10_000 + epoch)
        training.append({"epoch": epoch, "training_loss": loss})
        print(json.dumps(training[-1]), flush=True)

    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_ids), args.batch_size):
            x, _, _ = cache.batch(test_ids[start:start + args.batch_size])
            predictions.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    predicted = np.concatenate(predictions) * scales + means
    test_ends = np.asarray(cache.sample_rows[test_ids], dtype=np.int64)
    metadata = (cache.metadata.filter(pl.col("row_index").is_in(test_ends))
                .select("row_index", "trade_date", "ts_code").sort("row_index"))
    actual = np.asarray(cache.targets[test_ends])
    candidate = metadata.with_columns(
        pl.Series("pred_h1", predicted[:, 0]), pl.Series("pred_h5", predicted[:, 1]),
        pl.Series("excess_h1", actual[:, 0]), pl.Series("excess_h5", actual[:, 1]))
    keys = candidate.select("trade_date", "ts_code", "excess_h1", "excess_h5")
    comparisons = {"candidate": candidate}
    for name, path in (("old_lstm", args.old_lstm), ("lgbm", args.baseline)):
        comparisons[name] = (pl.read_parquet(path).select(
                                 "trade_date", "ts_code", "raw_h1", "raw_h5")
                             .join(keys, on=["trade_date", "ts_code"], how="inner")
                             .rename({"raw_h1": "pred_h1", "raw_h5": "pred_h5"}))
    metrics = {name: {"h1": rank_ic(frame, "pred_h1", "excess_h1"),
                      "h5": rank_ic(frame, "pred_h5", "excess_h5")}
               for name, frame in comparisons.items()}
    payload = {"signal": window["signal"], "test_start": window["test_dates"][0],
               "test_end": window["test_dates"][-1], "epochs": args.epochs,
               "learning_rate": args.learning_rate, "dropout": .10,
               "weight_decay": 1e-4, "recurrent_residual": True,
               "output_residual": True, "zero_init_output_residual": True,
               "training": training, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_parquet(args.output.with_suffix(".parquet"), compression="zstd")
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
