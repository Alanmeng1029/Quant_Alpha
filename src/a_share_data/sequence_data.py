"""Read-only mmap contract shared by the Rust cache and PyTorch trainer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SequenceCache:
    """Validated mmap view over the compact Rust sequence cache."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or self.manifest.get("version") != 1:
            raise ValueError("incomplete or unsupported sequence cache")
        self.rows = int(self.manifest["rows"])
        self.samples = int(self.manifest["samples"])
        self.sequence_length = int(self.manifest["sequence_length"])
        self.factor_count = int(self.manifest["factor_count"])
        expected = {
            "values.f32": self.rows * self.factor_count * 4,
            "missing.u8": self.rows * self.factor_count,
            "targets.f32": self.rows * 2 * 4,
            "sample_rows.u64": self.samples * 8,
        }
        for name, size in expected.items():
            path = self.root / name
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"sequence cache file has wrong size: {path}")
        self.values = np.memmap(self.root / "values.f32", dtype="<f4", mode="r",
                                shape=(self.rows, self.factor_count))
        self.missing = np.memmap(self.root / "missing.u8", dtype="u1", mode="r",
                                 shape=(self.rows, self.factor_count))
        self.targets = np.memmap(self.root / "targets.f32", dtype="<f4", mode="r",
                                 shape=(self.rows, 2))
        self.sample_rows = np.memmap(self.root / "sample_rows.u64", dtype="<u8", mode="r",
                                     shape=(self.samples,))
        self.metadata = pl.read_parquet(self.root / "metadata.parquet").sort("row_index")
        if self.metadata.height != self.rows:
            raise ValueError("metadata row count differs from binary arrays")
        if not np.array_equal(self.metadata["row_index"].to_numpy(), np.arange(self.rows)):
            raise ValueError("metadata row indexes are not contiguous")
        ends = np.asarray(self.sample_rows, dtype=np.int64)
        if len(ends) and (ends.min() < self.sequence_length - 1 or ends.max() >= self.rows):
            raise ValueError("sample row outside cache bounds")

    @property
    def sample_end_days(self) -> np.ndarray:
        return self.metadata["day_index"].to_numpy()[self.sample_rows]

    def sample_ids_for_days(self, day_indexes: np.ndarray | list[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.sample_end_days, np.asarray(day_indexes, dtype=np.int32)))

    def batch(self, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return X `[B,L,2F]`, targets `[B,2]`, and ending row indexes."""
        ids = np.asarray(sample_ids, dtype=np.int64)
        ends = np.asarray(self.sample_rows[ids], dtype=np.int64)
        offsets = np.arange(self.sequence_length - 1, -1, -1, dtype=np.int64)
        rows = ends[:, None] - offsets[None, :]
        values = np.asarray(self.values[rows], dtype=np.float32)
        missing = np.asarray(self.missing[rows], dtype=np.float32)
        return np.concatenate((values, missing), axis=2), np.asarray(self.targets[ends]), ends

    def iter_batches(self, sample_ids: np.ndarray, batch_size: int, *, shuffle: bool,
                     seed: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        order = np.asarray(sample_ids, dtype=np.int64).copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            yield self.batch(order[start:start + batch_size])


def winsorize_targets(targets: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Clip each horizon to its 1/99 percentiles within each signal date."""
    result = np.asarray(targets, dtype=np.float32).copy()
    for day in np.unique(days):
        indexes = np.flatnonzero(days == day)
        for horizon in range(2):
            values = result[indexes, horizon]
            finite = np.isfinite(values)
            if finite.any():
                lo, hi = np.quantile(values[finite], (.01, .99))
                values[finite] = np.clip(values[finite], lo, hi)
                result[indexes, horizon] = values
    return result


def target_statistics(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(targets, axis=0).astype(np.float32)
    scales = np.nanstd(targets, axis=0).astype(np.float32)
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("training targets lack finite, nonconstant values")
    return means, scales


def standardized_targets(targets: np.ndarray, means: np.ndarray,
                         scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(targets)
    values = np.zeros_like(targets, dtype=np.float32)
    normalized = (targets - means) / scales
    values[mask] = normalized[mask]
    return values, mask
