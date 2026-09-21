"""Isolated PyTorch/MPS trainer for compact Rust sequence caches."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

from a_share_data.sequence_data import (SequenceCache, standardized_targets,
                                        target_statistics, winsorize_targets)


class FactorLSTM(nn.Module):
    def __init__(self, channels: int, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.0, recurrent_residual: bool = False,
                 output_residual: bool = False,
                 zero_init_output_residual: bool = False):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer")
        recurrent = []
        residuals = []
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


def require_mps() -> torch.device:
    """Probe an actual MPS backward/update/serialization cycle; never fall back."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS CPU fallback is forbidden; set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable in this process")
    device = torch.device("mps")
    model = nn.Linear(3, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = model(torch.ones((4, 3), device=device)).square().mean()
    loss.backward(); optimizer.step()
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    clone = nn.Linear(3, 1).to(device); clone.load_state_dict(torch.load(buffer, map_location=device))
    torch.mps.synchronize()
    return device


def masked_horizon_mse(prediction: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for horizon in range(2):
        valid = mask[:, horizon]
        if valid.any():
            losses.append((prediction[valid, horizon] - target[valid, horizon]).square().mean())
    if not losses:
        raise ValueError("batch has no finite targets")
    return torch.stack(losses).mean()


def _targets(cache: SequenceCache, sample_ids: np.ndarray, *, winsorize: bool,
             means: np.ndarray | None = None, scales: np.ndarray | None = None):
    ends = np.asarray(cache.sample_rows[sample_ids], dtype=np.int64)
    raw = np.asarray(cache.targets[ends], dtype=np.float32).copy()
    days = cache.metadata["day_index"].to_numpy()[ends]
    fitted = winsorize_targets(raw, days) if winsorize else raw
    if means is None or scales is None:
        means, scales = target_statistics(fitted)
    values, mask = standardized_targets(fitted, means, scales)
    return values, mask, means, scales


def _epoch(cache: SequenceCache, sample_ids: np.ndarray, target_values: np.ndarray,
           target_mask: np.ndarray, model: nn.Module, device: torch.device,
           batch_size: int, optimizer: torch.optim.Optimizer | None, seed: int) -> float:
    training = optimizer is not None
    model.train(training)
    lookup = {int(sample): index for index, sample in enumerate(sample_ids)}
    total = 0.0; count = 0
    order = np.asarray(sample_ids, dtype=np.int64).copy()
    if training:
        np.random.default_rng(seed).shuffle(order)
    with (torch.enable_grad() if training else torch.no_grad()):
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size]
            x, _, _ = cache.batch(batch_ids)
            positions = np.fromiter((lookup[int(i)] for i in batch_ids), dtype=np.int64)
            y = torch.from_numpy(target_values[positions]).to(device)
            mask = torch.from_numpy(target_mask[positions]).to(device)
            prediction = model(torch.from_numpy(x).to(device))
            loss = masked_horizon_mse(prediction, y, mask)
            if training:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(batch_ids); count += len(batch_ids)
    if device.type == "mps": torch.mps.synchronize()
    return total / max(count, 1)


def fit_window(cache: SequenceCache, fit_ids: np.ndarray, valid_ids: np.ndarray,
               full_ids: np.ndarray, test_ids: np.ndarray, *, device: torch.device,
               batch_size: int, max_epochs: int, patience: int, learning_rate: float,
               seed: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.0,
               weight_decay: float = 0.0, recurrent_residual: bool = False,
               output_residual: bool = False,
               zero_init_output_residual: bool = False) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    # Window-local initialization is essential: skipping verified earlier windows
    # during resume must not change any later model.
    torch.manual_seed(seed)
    def labeled(sample_ids: np.ndarray) -> np.ndarray:
        ends = np.asarray(cache.sample_rows[sample_ids], dtype=np.int64)
        return sample_ids[np.isfinite(np.asarray(cache.targets[ends])).any(axis=1)]
    original_counts = {"fit": len(fit_ids), "validation": len(valid_ids), "full": len(full_ids)}
    fit_ids, valid_ids, full_ids = labeled(fit_ids), labeled(valid_ids), labeled(full_ids)
    if min(len(fit_ids), len(valid_ids), len(full_ids), len(test_ids)) < 1:
        raise ValueError("empty labeled training/validation split or empty prediction split")
    fit_y, fit_mask, means, scales = _targets(cache, fit_ids, winsorize=True)
    valid_y, valid_mask, _, _ = _targets(cache, valid_ids, winsorize=False, means=means, scales=scales)
    model_args = {"hidden": hidden, "dropout": dropout,
                  "recurrent_residual": recurrent_residual,
                  "output_residual": output_residual,
                  "zero_init_output_residual": zero_init_output_residual}
    model = FactorLSTM(cache.factor_count * 2, **model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8,
                                 weight_decay=weight_decay)
    best_loss = float("inf"); best_epoch = 0; best_state = None; stale = 0; history = []
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        training_loss = _epoch(cache, fit_ids, fit_y, fit_mask, model, device,
                               batch_size, optimizer, seed + epoch)
        validation_loss = _epoch(cache, valid_ids, valid_y, valid_mask, model, device,
                                 batch_size, None, seed)
        history.append({"epoch": epoch, "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "seconds": time.perf_counter() - started})
        print(json.dumps(history[-1]), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss; best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience: break
    if best_state is None: raise RuntimeError("validation did not produce a finite checkpoint")

    # The validation split selects only the epoch count.  Reinitialize and fit all
    # 756 dates so final training has the same historical span as LightGBM.
    torch.manual_seed(seed)
    full_y, full_mask, final_means, final_scales = _targets(cache, full_ids, winsorize=True)
    final_model = FactorLSTM(cache.factor_count * 2, **model_args).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, eps=1e-8,
                                       weight_decay=weight_decay)
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        loss = _epoch(cache, full_ids, full_y, full_mask, final_model, device,
                      batch_size, final_optimizer, seed + 10_000 + epoch)
        refit_history.append({"epoch": epoch, "training_loss": loss})

    predictions = []
    final_model.eval()
    with torch.no_grad():
        for x, _, _ in cache.iter_batches(test_ids, batch_size, shuffle=False, seed=seed):
            raw = final_model(torch.from_numpy(x).to(device)).cpu().numpy()
            predictions.append(raw * final_scales + final_means)
    prediction = np.concatenate(predictions) if predictions else np.empty((0, 2), np.float32)
    audit = {"best_epoch": best_epoch, "best_validation_loss": best_loss,
             "selection_history": history, "refit_history": refit_history,
             "fit_samples": len(fit_ids), "validation_samples": len(valid_ids),
             "full_training_samples": len(full_ids), "test_samples": len(test_ids),
             "excluded_unlabeled": {key: original_counts[key] - value for key, value in
                                    (("fit", len(fit_ids)), ("validation", len(valid_ids)),
                                     ("full", len(full_ids)))},
             "architecture": {"hidden": list(hidden), "channels": cache.factor_count * 2,
                              "sequence_length": cache.sequence_length, "dropout": dropout,
                              "recurrent_residual": recurrent_residual,
                              "output_residual": output_residual,
                              "zero_init_output_residual": zero_init_output_residual},
             "optimizer": "Adam", "learning_rate": learning_rate,
             "weight_decay": weight_decay,
             "gradient_clip_norm": 1.0, "device": str(device), "cpu_fallback": False,
             "runtime": {"python": sys.version, "torch": str(torch.__version__),
                         "machine": platform.machine()}}
    state = {"state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
             "target_means": final_means, "target_scales": final_scales}
    return prediction, audit, state


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache = SequenceCache(args.cache)
    specs = json.loads(args.windows.read_text(encoding="utf-8"))
    if args.max_windows: specs = specs[:args.max_windows]
    device = require_mps()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dates = cache.metadata.select("trade_date", "day_index").unique().sort("day_index")
    day_by_date = dict(dates.iter_rows())
    outputs = []
    for spec in specs:
        destination = args.output / "quarters" / f"month={spec['signal'][:7]}"
        complete_path = destination / "complete.json"
        source_hashes = {name: _digest(Path(__file__).with_name(name)) for name in
                         ("sequence_train.py", "sequence_data.py")}
        identity = hashlib.sha256(json.dumps({"cache": cache.manifest["fingerprint"], "spec": spec,
            "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "hidden_sizes": args.hidden_sizes, "dropout": args.dropout,
            "recurrent_residual": args.recurrent_residual,
            "output_residual": args.output_residual,
            "zero_init_output_residual": args.zero_init_output_residual,
            "seed": args.seed, "source_hashes": source_hashes,
            "python": sys.version, "torch": str(torch.__version__)}, sort_keys=True).encode()).hexdigest()
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            prediction_path = destination / "predictions.parquet"
            checkpoint_path = destination / "checkpoint.pt"
            if complete.get("identity") != identity or complete.get("prediction_sha256") != _digest(prediction_path) or complete.get("checkpoint_sha256") != _digest(checkpoint_path):
                raise ValueError(f"invalid completed window {spec['signal']}")
            outputs.append(pl.read_parquet(prediction_path)); continue
        destination.mkdir(parents=True, exist_ok=True)
        train_days = np.array([day_by_date[date.fromisoformat(d)] for d in spec["train_dates"]], np.int32)
        fit_days = train_days[:687]
        valid_days = train_days[-63:]
        if train_days[-63 - 6 - 1] != fit_days[-1]:
            raise ValueError("inner validation split does not contain the six-day purge")
        test_days = np.array([day_by_date[date.fromisoformat(d)] for d in spec["test_dates"]], np.int32)
        fit_ids = cache.sample_ids_for_days(fit_days)
        valid_ids = cache.sample_ids_for_days(valid_days)
        full_ids = cache.sample_ids_for_days(train_days)
        test_ids = cache.sample_ids_for_days(test_days)
        prediction, audit, state = fit_window(cache, fit_ids, valid_ids, full_ids, test_ids,
            device=device, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, learning_rate=args.learning_rate, seed=args.seed,
            hidden=tuple(args.hidden_sizes), dropout=args.dropout,
            weight_decay=args.weight_decay, recurrent_residual=args.recurrent_residual,
            output_residual=args.output_residual,
            zero_init_output_residual=args.zero_init_output_residual)
        ends = np.asarray(cache.sample_rows[test_ids], dtype=np.int64)
        result = cache.metadata.filter(pl.col("row_index").is_in(ends)).select("trade_date", "ts_code").with_columns(
            pl.Series("raw_h1", prediction[:, 0]), pl.Series("raw_h5", prediction[:, 1]))
        prediction_path = destination / "predictions.parquet"
        checkpoint_path = destination / "checkpoint.pt"
        result.write_parquet(prediction_path, compression="zstd")
        torch.save(state, checkpoint_path)
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if set(restored) != {"state_dict", "target_means", "target_scales"} or \
                set(restored["state_dict"]) != set(state["state_dict"]):
            raise ValueError("checkpoint reload validation failed")
        (destination / "training.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        complete = {"identity": identity, "signal": spec["signal"],
                    "prediction_sha256": _digest(prediction_path),
                    "checkpoint_sha256": _digest(checkpoint_path), **{k: audit[k] for k in
                    ("best_epoch", "best_validation_loss", "full_training_samples", "test_samples")}}
        temporary = complete_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(complete, indent=2), encoding="utf-8")
        os.replace(temporary, complete_path)
        outputs.append(result)
    combined = pl.concat(outputs).unique(["trade_date", "ts_code"]).sort(["trade_date", "ts_code"])
    combined_path = args.output / "raw_predictions.parquet"
    combined.write_parquet(combined_path, compression="zstd")
    return {"windows": len(specs), "rows": combined.height, "predictions": str(combined_path),
            "device": str(device)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quant-sequence-train")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--recurrent-residual", action="store_true")
    parser.add_argument("--output-residual", action="store_true")
    parser.add_argument("--zero-init-output-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.max_epochs, args.patience, *args.hidden_sizes) < 1: raise ValueError("positive training limits required")
    if not (0 <= args.dropout < 1 and args.weight_decay >= 0): raise ValueError("invalid regularization")
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__": main()
