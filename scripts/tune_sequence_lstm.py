#!/usr/bin/env python3
"""Compare LSTM optimization curves on a pre-OOS validation split."""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from a_share_data.sequence_data import SequenceCache
from a_share_data.sequence_train import _epoch, _targets, require_mps


class CandidateLSTM(nn.Module):
    def __init__(self, channels: int, hidden: list[int], dropout: float,
                 recurrent_residual: bool = False, output_residual: bool = False,
                 zero_init_output_residual: bool = False):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer")
        recurrent: list[nn.Module] = []
        residuals: list[nn.Module] = []
        width = channels
        for next_width in hidden:
            recurrent.append(nn.LSTM(width, next_width, batch_first=True))
            if recurrent_residual:
                residuals.append(nn.Linear(width, next_width, bias=False))
            width = next_width
        self.recurrent = nn.ModuleList(recurrent)
        self.residuals = nn.ModuleList(residuals)
        self.dropout = nn.Dropout(dropout)
        self.recurrent_residual = recurrent_residual
        self.output = nn.Linear(width, 2)
        self.output_residual = nn.Linear(channels, 2) if output_residual else None
        if self.output_residual is not None and zero_init_output_residual:
            nn.init.zeros_(self.output_residual.weight)
            nn.init.zeros_(self.output_residual.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        raw_last = values[:, -1]
        for index, layer in enumerate(self.recurrent):
            previous = values
            values, _ = layer(values)
            if self.recurrent_residual:
                values = values + self.residuals[index](previous)
            if index + 1 < len(self.recurrent):
                values = self.dropout(values)
        result = self.output(values[:, -1])
        if self.output_residual is not None:
            result = result + self.output_residual(raw_last)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--fold-fit-days", type=int, nargs="+", default=[687])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = SequenceCache(args.cache)
    window = json.loads(args.windows.read_text())[0]
    specs = json.loads(args.specs.read_text())
    dates = cache.metadata.select("trade_date", "day_index").unique().sort("day_index")
    day_by_date = dict(dates.iter_rows())
    train_days = np.asarray(
        [day_by_date[date.fromisoformat(value)] for value in window["train_dates"]], np.int32)
    def labeled(sample_ids: np.ndarray) -> np.ndarray:
        ends = np.asarray(cache.sample_rows[sample_ids], dtype=np.int64)
        return sample_ids[np.isfinite(np.asarray(cache.targets[ends])).any(axis=1)]

    folds = []
    for fit_days in args.fold_fit_days:
        validation_start = fit_days + 6
        validation_stop = validation_start + 63
        if fit_days < 1 or validation_stop > len(train_days):
            raise ValueError(f"invalid fold fit days: {fit_days}")
        fit_ids = labeled(cache.sample_ids_for_days(train_days[:fit_days]))
        valid_ids = labeled(cache.sample_ids_for_days(
            train_days[validation_start:validation_stop]))
        fit_y, fit_mask, means, scales = _targets(cache, fit_ids, winsorize=True)
        valid_y, valid_mask, _, _ = _targets(
            cache, valid_ids, winsorize=False, means=means, scales=scales)
        folds.append({"fit_days": fit_days, "fit_ids": fit_ids, "valid_ids": valid_ids,
                      "fit_y": fit_y, "fit_mask": fit_mask, "valid_y": valid_y,
                      "valid_mask": valid_mask,
                      "fit_start": window["train_dates"][0],
                      "fit_end": window["train_dates"][fit_days - 1],
                      "validation_start": window["train_dates"][validation_start],
                      "validation_end": window["train_dates"][validation_stop - 1]})
    device = require_mps()
    results = []
    for spec in specs:
        fold_results = []
        for fold in folds:
            torch.manual_seed(args.seed)
            model = CandidateLSTM(
                cache.factor_count * 2, spec["hidden"], spec["dropout"],
                recurrent_residual=spec.get("recurrent_residual", False),
                output_residual=spec.get("output_residual", False),
                zero_init_output_residual=spec.get("zero_init_output_residual", False)).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=spec["learning_rate"], eps=1e-8,
                weight_decay=spec["weight_decay"])
            history = []
            started = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                train_loss = _epoch(cache, fold["fit_ids"], fold["fit_y"], fold["fit_mask"],
                                    model, device, args.batch_size, optimizer, args.seed + epoch)
                valid_loss = _epoch(cache, fold["valid_ids"], fold["valid_y"],
                                    fold["valid_mask"], model, device, args.batch_size, None,
                                    args.seed)
                row = {"epoch": epoch, "training_loss": train_loss,
                       "validation_loss": valid_loss}
                history.append(row)
                print(json.dumps({"name": spec["name"], "fit_days": fold["fit_days"],
                                  **row}), flush=True)
            best = min(history, key=lambda row: row["validation_loss"])
            fold_results.append({"fit_days": fold["fit_days"],
                                 "fit_start": fold["fit_start"], "fit_end": fold["fit_end"],
                                 "validation_start": fold["validation_start"],
                                 "validation_end": fold["validation_end"],
                                 "fit_samples": len(fold["fit_ids"]),
                                 "validation_samples": len(fold["valid_ids"]),
                                 "best_epoch": best["epoch"],
                                 "best_validation_loss": best["validation_loss"],
                                 "seconds": time.perf_counter() - started, "history": history})
        losses = [fold["best_validation_loss"] for fold in fold_results]
        results.append({**spec, "mean_best_validation_loss": float(np.mean(losses)),
                        "worst_best_validation_loss": float(np.max(losses)),
                        "best_epochs": [fold["best_epoch"] for fold in fold_results],
                        "folds": fold_results})
    payload = {"window_signal": window["signal"], "train_start": window["train_dates"][0],
               "train_end": window["train_dates"][-1], "fold_fit_days": args.fold_fit_days,
               "epochs": args.epochs,
               "batch_size": args.batch_size, "seed": args.seed, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "candidates": len(results)}))


if __name__ == "__main__":
    main()
