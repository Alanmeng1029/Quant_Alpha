"""Deterministic CSI500/CSI1000 candidate selection.

This module owns the investment-policy invariants.  It deliberately has no
model or data-source dependency so the Python research runner and Rust daily
runtime can share the same compact decision fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CSI500 = "000905.SH"
CSI1000 = "000852.SH"


@dataclass(frozen=True)
class SleeveRule:
    index_code: str
    target: int
    entry_rank: int
    exit_rank: int


RULES = (SleeveRule(CSI500, 80, 80, 96), SleeveRule(CSI1000, 20, 20, 24))


@dataclass(frozen=True)
class Decision:
    sleeve: str
    code: str
    action: str
    reason: str
    rank: int | None


def assign_sleeves(csi500: Iterable[str], csi1000: Iterable[str]) -> dict[str, str]:
    """CSI500 wins every historic overlap deterministically."""
    out = {code: CSI500 for code in csi500}
    out.update({code: CSI1000 for code in csi1000 if code not in out})
    return out


def blend_joint_scores(
    codes: Iterable[str], pred_h1: Iterable[float], pred_h5: Iterable[float]
) -> dict[str, float]:
    """Return a 50/50 H1/H5 blend after joint cross-sectional z-scoring."""
    code_values = list(codes)
    h1 = np.asarray(list(pred_h1), dtype=float)
    h5 = np.asarray(list(pred_h5), dtype=float)
    if len(code_values) != len(h1) or len(h1) != len(h5):
        raise ValueError("codes, pred_h1 and pred_h5 must have the same length")
    finite = np.isfinite(h1) & np.isfinite(h5)

    def zscore(values: np.ndarray) -> np.ndarray:
        std = values.std()
        return (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)

    result = np.full(len(h1), np.nan)
    if finite.any():
        result[finite] = 0.5 * zscore(h1[finite]) + 0.5 * zscore(h5[finite])
    return {code: float(score) for code, score in zip(code_values, result, strict=True) if np.isfinite(score)}


def _ranked(scores: dict[str, float], members: set[str]) -> list[str]:
    return sorted((c for c in members if c in scores), key=lambda c: (-scores[c], c))


def decide(
    scores: dict[str, float],
    memberships: dict[str, str],
    held: dict[str, set[str]],
    forced_exits: set[str] | None = None,
    shared_limit: int = 3,
    sleeve_limits: dict[str, int] | None = None,
    rules: tuple[SleeveRule, ...] = RULES,
) -> tuple[dict[str, set[str]], list[Decision]]:
    """Apply sleeve buffers with one shared replacement budget.

    Forced exits are recorded first.  Their refills consume the shared buy
    budget before ordinary rank-driven replacements.  A holding outside its
    assigned sleeve is always treated as a forced exit.
    """
    if shared_limit < 0:
        raise ValueError("shared_limit must be non-negative")
    forced_exits = set(forced_exits or ())
    target = {rule.index_code: set(held.get(rule.index_code, set())) for rule in rules}
    decisions: list[Decision] = []
    buy_slots = shared_limit
    per_sleeve_slots = dict(sleeve_limits) if sleeve_limits is not None else None
    if per_sleeve_slots is not None and any(per_sleeve_slots.get(rule.index_code, 0) < 0 for rule in RULES):
        raise ValueError("sleeve replacement limits must be non-negative")
    ranked: dict[str, list[str]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for rule in rules:
        members = {code for code, sleeve in memberships.items() if sleeve == rule.index_code}
        ranked[rule.index_code] = _ranked(scores, members)
        ranks[rule.index_code] = {code: n + 1 for n, code in enumerate(ranked[rule.index_code])}
        for code in sorted(target[rule.index_code]):
            if code in forced_exits or memberships.get(code) != rule.index_code:
                target[rule.index_code].remove(code)
                decisions.append(Decision(rule.index_code, code, "sell", "forced_exit", ranks[rule.index_code].get(code)))
    # Forced refills precede normal swaps and share the same three buys.
    for rule in rules:
        for code in ranked[rule.index_code]:
            slots = per_sleeve_slots.get(rule.index_code, 0) if per_sleeve_slots is not None else buy_slots
            if len(target[rule.index_code]) >= rule.target or slots == 0:
                break
            if code not in target[rule.index_code] and code not in forced_exits:
                target[rule.index_code].add(code)
                if per_sleeve_slots is None:
                    buy_slots -= 1
                else:
                    per_sleeve_slots[rule.index_code] -= 1
                decisions.append(Decision(rule.index_code, code, "buy", "forced_refill", ranks[rule.index_code][code]))
    # Compete ordinary exits by rank / sleeve target, then sleeve and code.
    candidates: list[tuple[float, str, str, SleeveRule]] = []
    for rule in rules:
        for code in target[rule.index_code]:
            rank = ranks[rule.index_code].get(code)
            if rank is not None and rank > rule.exit_rank:
                candidates.append((rank / rule.target, rule.index_code, code, rule))
    for _, sleeve, code, rule in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
        slots = per_sleeve_slots.get(sleeve, 0) if per_sleeve_slots is not None else buy_slots
        if slots == 0:
            continue
        pool = [x for x in ranked[sleeve] if x not in target[sleeve] and x not in forced_exits]
        if not pool:
            continue
        target[sleeve].remove(code)
        decisions.append(Decision(sleeve, code, "sell", "rank_exit", ranks[sleeve].get(code)))
        replacement = pool[0]
        target[sleeve].add(replacement)
        if per_sleeve_slots is None:
            buy_slots -= 1
        else:
            per_sleeve_slots[sleeve] -= 1
        decisions.append(Decision(sleeve, replacement, "buy", "rank_refill", ranks[sleeve][replacement]))
    return target, decisions
