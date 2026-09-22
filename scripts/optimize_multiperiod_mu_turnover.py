#!/usr/bin/env python3
"""Receding-horizon H1/H5 optimizer with explicit per-period turnover costs.

For each signal date the optimizer plans five target portfolios.  H1 is the
period-1 expected excess return and (H5-H1)/4 is the expected incremental
return in periods 2..5.  Every transition pays buy and sell costs.  Only the
first target portfolio is emitted; the next signal date solves a fresh plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buy-bps", type=float, default=2.0)
    parser.add_argument("--sell-bps", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=0.01)
    parser.add_argument("--invested-weight", type=float, default=0.98)
    parser.add_argument(
        "--previous-positions",
        type=Path,
        help="Optional prior target/executed weights for the first signal date",
    )
    parser.add_argument("--term-structure", choices=("h1h5", "h1h5h10"), default="h1h5")
    parser.add_argument("--max-days", type=int)
    return parser.parse_args()


def solve_path(
    codes: list[str],
    alphas: np.ndarray,
    previous: dict[str, float],
    eligible: np.ndarray,
    invested_weight: float,
    max_weight: float,
    buy_cost: float,
    sell_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve w[t], buy[t], sell[t] for a multi-period linear program."""
    n = len(codes)
    horizon = alphas.shape[0]
    block = horizon * n
    # Variables are [w, buy, sell], each flattened as period-major arrays.
    objective = np.empty(3 * block)
    objective[:block] = -alphas.ravel()
    objective[block:2 * block] = buy_cost
    objective[2 * block:] = sell_cost

    # Five budget equalities plus a holdings-flow equality per name and period.
    equality = lil_matrix((horizon + block, 3 * block), dtype=float)
    rhs = np.zeros(horizon + block)
    for period in range(horizon):
        start = period * n
        equality[period, start:start + n] = 1.0
        rhs[period] = invested_weight
        for name in range(n):
            row = horizon + start + name
            weight = start + name
            buy = block + start + name
            sell = 2 * block + start + name
            equality[row, weight] = 1.0
            equality[row, buy] = -1.0
            equality[row, sell] = 1.0
            if period == 0:
                rhs[row] = previous.get(codes[name], 0.0)
            else:
                equality[row, weight - n] = -1.0

    weight_bounds = [
        (0.0, max_weight if eligible[name] else 0.0)
        for _period in range(horizon)
        for name in range(n)
    ]
    bounds = weight_bounds + [(0.0, None)] * (2 * block)
    result = linprog(
        objective,
        A_eq=equality.tocsr(),
        b_eq=rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"multi-period optimizer failed: {result.message}")
    weights = result.x[:block].reshape(horizon, n)
    buys = result.x[block:2 * block].reshape(horizon, n)
    sells = result.x[2 * block:].reshape(horizon, n)
    return weights, buys, sells, float(-result.fun)


def main() -> None:
    args = parse_args()
    if not (0 < args.max_weight <= args.invested_weight <= 1):
        raise ValueError("require 0 < max_weight <= invested_weight <= 1")
    source = (
        pl.read_parquet(args.predictions)
        .with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("execution_date").cast(pl.Date),
            pl.col("raw_h1").cast(pl.Float64),
            pl.col("raw_h5").cast(pl.Float64),
        )
        .filter(
            pl.col("execution_date").is_not_null()
            & pl.col("raw_h1").is_finite()
            & pl.col("raw_h5").is_finite()
            & (pl.col("raw_h10").is_finite() if args.term_structure == "h1h5h10" else pl.lit(True))
        )
    )
    # H1/H5 models (including the two-head LSTM) do not predict H10. This
    # placeholder is internal and unused by the five-period forecast curve.
    if args.term_structure == "h1h5" and "raw_h10" not in source.columns:
        source = source.with_columns(pl.lit(0.0).alias("raw_h10"))
    partitions = source.partition_by("trade_date", as_dict=True, maintain_order=True)
    previous: dict[str, float] = {}
    if args.previous_positions is not None:
        prior = pl.read_parquet(args.previous_positions)
        weight_column = (
            "target_weight" if "target_weight" in prior.columns
            else "weight" if "weight" in prior.columns
            else None
        )
        if weight_column is None:
            raise ValueError("previous positions need target_weight or weight")
        if prior.select(pl.col("ts_code").is_duplicated().any()).item():
            raise ValueError("previous positions contain duplicate ts_code values")
        previous = {
            str(code): float(weight)
            for code, weight in prior.select("ts_code", weight_column).iter_rows()
            if weight is not None and np.isfinite(weight) and weight > 0
        }
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for day_number, (key, frame) in enumerate(partitions.items()):
        if args.max_days is not None and day_number >= args.max_days:
            break
        trade_date = key[0] if isinstance(key, tuple) else key
        frame = frame.sort("ts_code")
        current_codes = frame["ts_code"].to_list()
        current = set(current_codes)
        # Keep names that must be liquidated after leaving the eligible universe.
        codes = sorted(current | set(previous))
        forecasts = {
            code: (float(h1), float(h5), float(h10))
            for code, h1, h5, h10 in frame.select("ts_code", "raw_h1", "raw_h5", "raw_h10").iter_rows()
        }
        h1 = np.array([forecasts.get(code, (0.0, 0.0, 0.0))[0] for code in codes])
        h5 = np.array([forecasts.get(code, (0.0, 0.0, 0.0))[1] for code in codes])
        first_five = np.vstack([h1, *([(h5 - h1) / 4.0] * 4)])
        if args.term_structure == "h1h5h10":
            h10 = np.array([forecasts.get(code, (0.0, 0.0, 0.0))[2] for code in codes])
            alphas = np.vstack([first_five, *([(h10 - h5) / 5.0] * 5)])
        else:
            alphas = first_five
        eligible = np.array([code in current for code in codes])
        weights, buys, sells, objective = solve_path(
            codes, alphas, previous, eligible, args.invested_weight, args.max_weight,
            args.buy_bps / 10_000.0, args.sell_bps / 10_000.0,
        )
        first = weights[0]
        # Ineligible names are not allowed to remain in the emitted portfolio.
        if any(first[index] > 1e-10 for index, code in enumerate(codes) if code not in current):
            raise RuntimeError("optimizer retained an ineligible name")
        previous = {code: float(weight) for code, weight in zip(codes, first) if weight > 1e-12}
        execution_date = frame["execution_date"][0]
        rows.extend(
            {"trade_date": trade_date, "execution_date": execution_date,
             "ts_code": code, "target_weight": weight}
            for code, weight in previous.items()
        )
        diagnostics.append({
            "trade_date": trade_date,
            "execution_date": execution_date,
            "objective": objective,
            "first_buy_turnover": float(buys[0].sum()),
            "first_sell_turnover": float(sells[0].sum()),
            "planned_buy_turnover": float(buys.sum()),
            "planned_sell_turnover": float(sells.sum()),
            "first_holding_count": len(previous),
            "period_1_to_2_weight_change": float(np.abs(weights[1] - weights[0]).sum()),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.output / "target_weights.parquet", compression="zstd")
    diagnostic_frame = pl.DataFrame(diagnostics)
    diagnostic_frame.write_parquet(args.output / "optimizer_daily.parquet", compression="zstd")
    summary = {
        "optimizer": f"{alphas.shape[0]}-period receding-horizon linear mu-minus-turnover-cost, no risk term",
        "term_structure": args.term_structure,
        "forecast_curve": (["raw_h1", "(raw_h5-raw_h1)/4 repeated for periods 2..5"]
                           if args.term_structure == "h1h5" else
                           ["raw_h1", "(raw_h5-raw_h1)/4 repeated for periods 2..5",
                            "(raw_h10-raw_h5)/5 repeated for periods 6..10"]),
        "predictions": str(args.predictions.resolve()),
        "previous_positions": (
            str(args.previous_positions.resolve()) if args.previous_positions else None
        ),
        "buy_bps": args.buy_bps,
        "sell_bps": args.sell_bps,
        "max_weight": args.max_weight,
        "invested_weight": args.invested_weight,
        "days": diagnostic_frame.height,
        "mean_first_buy_turnover": float(diagnostic_frame["first_buy_turnover"].mean()),
        "mean_first_sell_turnover": float(diagnostic_frame["first_sell_turnover"].mean()),
        "mean_planned_buy_turnover": float(diagnostic_frame["planned_buy_turnover"].mean()),
        "mean_period_1_to_2_weight_change": float(diagnostic_frame["period_1_to_2_weight_change"].mean()),
    }
    (args.output / "optimizer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
