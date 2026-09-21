#!/usr/bin/env python3
"""Evaluate the frozen residual LSTM on selected rolling OOS windows."""
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


def labeled(cache: SequenceCache, sample_ids: np.ndarray) -> np.ndarray:
    ends = np.asarray(cache.sample_rows[sample_ids], dtype=np.int64)
    return sample_ids[np.isfinite(np.asarray(cache.targets[ends])).any(axis=1)]


def build_model(cache: SequenceCache, device: torch.device) -> CandidateLSTM:
    return CandidateLSTM(cache.factor_count * 2, [128, 64], .10,
                         recurrent_residual=True, output_residual=True,
                         zero_init_output_residual=True).to(device)


def rank_ic(frame: pl.DataFrame, prediction: str, target: str) -> dict[str, float | int]:
    daily = (frame.filter(pl.col(prediction).is_finite() & pl.col(target).is_finite())
             .group_by("trade_date")
             .agg(pl.corr(prediction, target, method="spearman").alias("ic"))
             .drop_nulls("ic"))
    return {"days": daily.height, "mean_rank_ic": float(daily["ic"].mean()),
            "positive_ratio": float((daily["ic"] > 0).mean())}


def metrics(frame: pl.DataFrame) -> dict:
    return {horizon: rank_ic(frame, f"pred_{horizon}", f"excess_{horizon}")
            for horizon in ("h1", "h5")}


def predict(cache: SequenceCache, model: torch.nn.Module, sample_ids: np.ndarray,
            device: torch.device, batch_size: int, means: np.ndarray,
            scales: np.ndarray) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_ids), batch_size):
            x, _, _ = cache.batch(sample_ids[start:start + batch_size])
            output.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(output) * scales + means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--old-lstm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = SequenceCache(args.cache)
    windows = [item for item in json.loads(args.windows.read_text())
               if item["signal"].startswith(args.year + "-")]
    if not windows:
        raise ValueError(f"no windows for year {args.year}")
    dates = cache.metadata.select("trade_date", "day_index").unique().sort("day_index")
    day_by_date = dict(dates.iter_rows())
    device = require_mps()
    frames = []
    audits = []
    for window in windows:
        train_days = np.asarray([day_by_date[date.fromisoformat(value)]
                                 for value in window["train_dates"]], np.int32)
        test_days = np.asarray([day_by_date[date.fromisoformat(value)]
                                for value in window["test_dates"]], np.int32)
        fit_ids = labeled(cache, cache.sample_ids_for_days(train_days[:687]))
        valid_ids = labeled(cache, cache.sample_ids_for_days(train_days[-63:]))
        full_ids = labeled(cache, cache.sample_ids_for_days(train_days))
        test_ids = cache.sample_ids_for_days(test_days)
        fit_y, fit_mask, means, scales = _targets(cache, fit_ids, winsorize=True)
        valid_y, valid_mask, _, _ = _targets(
            cache, valid_ids, winsorize=False, means=means, scales=scales)

        torch.manual_seed(args.seed)
        model = build_model(cache, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     weight_decay=1e-4, eps=1e-8)
        history = []
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, args.max_epochs + 1):
            train_loss = _epoch(cache, fit_ids, fit_y, fit_mask, model, device,
                                args.batch_size, optimizer, args.seed + epoch)
            valid_loss = _epoch(cache, valid_ids, valid_y, valid_mask, model, device,
                                args.batch_size, None, args.seed)
            row = {"epoch": epoch, "training_loss": train_loss,
                   "validation_loss": valid_loss}
            history.append(row)
            print(json.dumps({"signal": window["signal"], **row}), flush=True)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

        full_y, full_mask, final_means, final_scales = _targets(
            cache, full_ids, winsorize=True)
        torch.manual_seed(args.seed)
        final_model = build_model(cache, device)
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate,
                                           weight_decay=1e-4, eps=1e-8)
        refit = []
        for epoch in range(1, best_epoch + 1):
            loss = _epoch(cache, full_ids, full_y, full_mask, final_model, device,
                          args.batch_size, final_optimizer, args.seed + 10_000 + epoch)
            refit.append({"epoch": epoch, "training_loss": loss})
        predicted = predict(cache, final_model, test_ids, device, args.batch_size,
                            final_means, final_scales)
        ends = np.asarray(cache.sample_rows[test_ids], dtype=np.int64)
        actual = np.asarray(cache.targets[ends])
        rows = pl.DataFrame({"row_index": ends, "pred_h1": predicted[:, 0],
                             "pred_h5": predicted[:, 1], "excess_h1": actual[:, 0],
                             "excess_h5": actual[:, 1]})
        frame = (rows.join(cache.metadata.select("row_index", "trade_date", "ts_code"),
                           on="row_index", how="left")
                 .with_columns(pl.lit(window["signal"][:7]).alias("quarter")))
        frames.append(frame)
        audits.append({"signal": window["signal"], "best_epoch": best_epoch,
                       "best_validation_loss": best_loss, "selection_history": history,
                       "refit_history": refit, "test_rows": frame.height})

    candidate = pl.concat(frames).sort(["trade_date", "ts_code"])
    keys = candidate.select("trade_date", "ts_code", "quarter", "excess_h1", "excess_h5")
    comparisons = {"candidate": candidate}
    for name, path in (("old_lstm", args.old_lstm), ("lgbm", args.baseline)):
        comparisons[name] = (pl.read_parquet(path).select(
                                 "trade_date", "ts_code", "raw_h1", "raw_h5")
                             .join(keys, on=["trade_date", "ts_code"], how="inner")
                             .rename({"raw_h1": "pred_h1", "raw_h5": "pred_h5"}))
    result_metrics = {}
    for name, frame in comparisons.items():
        result_metrics[name] = {"combined": metrics(frame), "quarters": {}}
        for quarter, group in frame.group_by("quarter", maintain_order=True):
            key = quarter[0] if isinstance(quarter, tuple) else quarter
            result_metrics[name]["quarters"][str(key)] = metrics(group)
    payload = {"year": args.year, "learning_rate": args.learning_rate,
               "dropout": .10, "weight_decay": 1e-4, "patience": args.patience,
               "architecture": [128, 64], "recurrent_residual": True,
               "output_residual": True, "zero_init_output_residual": True,
               "audits": audits, "metrics": result_metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_parquet(args.output.with_suffix(".parquet"), compression="zstd")
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), "metrics": result_metrics}))


if __name__ == "__main__":
    main()
