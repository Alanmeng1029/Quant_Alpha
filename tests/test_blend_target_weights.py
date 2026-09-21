from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "blend_target_weights.py"
SPEC = importlib.util.spec_from_file_location("blend_target_weights", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_targets(path: Path, weights: list[tuple[str, float]]) -> None:
    pl.DataFrame(
        {
            "trade_date": ["2026-01-05"] * len(weights),
            "execution_date": ["2026-01-06"] * len(weights),
            "ts_code": [code for code, _ in weights],
            "target_weight": [weight for _, weight in weights],
        }
    ).write_parquet(path)


def test_blend_targets_nets_overlapping_names(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_targets(first, [("A", 0.49), ("B", 0.49)])
    _write_targets(second, [("A", 0.20), ("C", 0.78)])

    targets, daily, summary = MODULE.blend_targets(
        [first, second], [0.8, 0.2], invested_weight=0.98
    )

    actual = dict(targets.select("ts_code", "target_weight").iter_rows())
    assert actual == pytest.approx({"A": 0.432, "B": 0.392, "C": 0.156})
    assert daily["sum_weight"][0] == pytest.approx(0.98)
    assert summary["max_weight"] == pytest.approx(0.432)


def test_blend_targets_rejects_allocations_that_do_not_sum_to_one(tmp_path: Path) -> None:
    target = tmp_path / "target.parquet"
    _write_targets(target, [("A", 0.98)])

    with pytest.raises(ValueError, match="sum to one"):
        MODULE.blend_targets([target], [0.8], invested_weight=0.98)
