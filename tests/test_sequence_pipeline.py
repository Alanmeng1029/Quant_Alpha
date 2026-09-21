from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import subprocess

import numpy as np
import polars as pl
import pytest

from a_share_data.sequence_data import (SequenceCache, standardized_targets,
                                        target_statistics, winsorize_targets)


def _panel(path: Path) -> None:
    rows = []
    for code, days in (("A", range(5)), ("B", (0, 1, 3, 4))):
        for day in days:
            rows.append({"trade_date": date(2024, 1, 1) + timedelta(days=day),
                         "ts_code": code, "day_index": day,
                         "f1": None if code == "A" and day == 1 else float(day),
                         "f2": float(day + 10), "excess_h1": float(day) / 100,
                         "excess_h5": None if day == 4 else float(day) / 50})
    pl.DataFrame(rows).with_columns(
        pl.col("day_index").cast(pl.Int32), pl.col("f1", "f2", "excess_h1", "excess_h5").cast(pl.Float32)
    ).sort(["ts_code", "day_index"]).select(
        "trade_date", "ts_code", "day_index", "f1", "f2", "excess_h1", "excess_h5"
    ).write_parquet(path)


def test_rust_cache_preserves_windows_missing_flags_and_gaps(tmp_path: Path) -> None:
    binary = Path(__file__).resolve().parents[1] / "target/release/quant-sequence-cache"
    if not binary.is_file(): pytest.skip("release quant-sequence-cache binary is not built")
    source = tmp_path / "panel.parquet"; factors = tmp_path / "factors.json"
    output = tmp_path / "cache"; _panel(source); factors.write_text('["f1", "f2"]')
    subprocess.run([str(binary), "--input", str(source), "--output", str(output),
                    "--factor-ids", str(factors), "--sequence-length", "3",
                    "--fingerprint", "fixture"], check=True)
    cache = SequenceCache(output)
    # A yields end days 2,3,4. B contains a gap and yields no three-day sequence.
    assert cache.samples == 3
    x, y, ends = cache.batch(np.array([0]))
    assert x.shape == (1, 3, 4)
    assert x[0, 1, 0] == 0.0
    assert x[0, 1, 2] == 1.0
    assert np.allclose(y[0], [.02, .04])
    assert cache.metadata["ts_code"][int(ends[0])] == "A"


def test_target_preparation_is_daily_and_masked() -> None:
    raw = np.array([[0., 1.], [1., np.nan], [100., 3.], [4., 4.]], np.float32)
    days = np.array([1, 1, 1, 2])
    clipped = winsorize_targets(raw, days)
    assert clipped[2, 0] < 100
    assert np.isnan(clipped[1, 1])
    means, scales = target_statistics(clipped)
    values, mask = standardized_targets(clipped, means, scales)
    assert values.shape == mask.shape == raw.shape
    assert values[1, 1] == 0 and not mask[1, 1]


def test_sequence_cache_rejects_truncated_binary(tmp_path: Path) -> None:
    root = tmp_path / "cache"; root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"version": 1, "status": "complete",
        "rows": 1, "samples": 0, "sequence_length": 1, "factor_count": 1}))
    (root / "values.f32").write_bytes(b"")
    with pytest.raises(ValueError, match="wrong size"):
        SequenceCache(root)
