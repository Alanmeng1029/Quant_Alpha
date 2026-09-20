"""Exact no-risk portfolio optimizer with asymmetric linear trading costs.

The objective is

    max mu'w - buy_cost * sum((w-w_prev)+) - sell_cost * sum((w_prev-w)+)

subject to a fixed invested weight and per-name bounds.  With no covariance
term or other coupled constraints this is a separable concave piecewise-linear
problem: retaining existing weight has marginal value ``mu + sell_cost`` and
adding new weight has marginal value ``mu - buy_cost``.  Sorting those weight
segments therefore gives the exact optimum without a generic solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    signal = parser.add_mutually_exclusive_group(required=True)
    signal.add_argument("--horizon", type=int, choices=(1, 5))
    signal.add_argument("--signal", choices=("h1", "h5", "h1h5"))
    parser.add_argument("--h1-weight", type=float, default=0.5)
    parser.add_argument("--h5-weight", type=float, default=0.5)
    parser.add_argument("--buy-bps", type=float, default=2.0)
    parser.add_argument("--sell-bps", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=0.01)
    parser.add_argument("--invested-weight", type=float, default=0.98)
    return parser.parse_args()


def optimize_day(
    codes: list[str],
    mu: np.ndarray,
    previous: dict[str, float],
    invested_weight: float,
    max_weight: float,
    buy_cost: float,
    sell_cost: float,
) -> tuple[np.ndarray, float, float, float]:
    old = np.array([min(previous.get(code, 0.0), max_weight) for code in codes])
    segments: list[tuple[float, str, int, float]] = []
    for index, (code, forecast, held) in enumerate(zip(codes, mu, old)):
        if held > 0:
            segments.append((float(forecast + sell_cost), code, index, float(held)))
        room = max_weight - held
        if room > 1e-15:
            segments.append((float(forecast - buy_cost), code, index, float(room)))
    segments.sort(key=lambda row: (-row[0], row[1], row[2]))
    weights = np.zeros(len(codes))
    remaining = invested_weight
    for _, _, index, capacity in segments:
        allocation = min(capacity, remaining)
        weights[index] += allocation
        remaining -= allocation
        if remaining <= 1e-12:
            break
    if remaining > 1e-9:
        raise ValueError("max_weight and eligible universe cannot satisfy invested_weight")
    prior_total = sum(previous.values())
    forced_sales = max(0.0, prior_total - old.sum())
    buys = float(np.maximum(weights - old, 0.0).sum())
    sells = float(np.maximum(old - weights, 0.0).sum() + forced_sales)
    objective = float(mu @ weights - buy_cost * buys - sell_cost * sells)
    return weights, buys, sells, objective


def main() -> None:
    args = parse_args()
    if not (0 < args.max_weight <= args.invested_weight <= 1):
        raise ValueError("require 0 < max_weight <= invested_weight <= 1")
    signal = args.signal or f"h{args.horizon}"
    if signal == "h1":
        mu_expression = pl.col("raw_h1")
        required_mu_columns = ["raw_h1"]
    elif signal == "h5":
        mu_expression = pl.col("raw_h5")
        required_mu_columns = ["raw_h5"]
    else:
        if args.h1_weight < 0 or args.h5_weight < 0 or args.h1_weight + args.h5_weight <= 0:
            raise ValueError("multi-period signal weights must be non-negative with a positive sum")
        weight_sum = args.h1_weight + args.h5_weight
        mu_expression = (
            args.h1_weight * pl.col("raw_h1")
            + args.h5_weight * pl.col("raw_h5") / 5.0
        ) / weight_sum
        required_mu_columns = ["raw_h1", "raw_h5"]
    source = (
        pl.read_parquet(args.predictions)
        .with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("execution_date").cast(pl.Date),
            mu_expression.alias("optimizer_mu"),
        )
        .filter(
            pl.col("execution_date").is_not_null()
            & pl.all_horizontal([pl.col(column).is_finite() for column in required_mu_columns])
        )
    )
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    buy_cost = args.buy_bps / 10_000.0
    sell_cost = args.sell_bps / 10_000.0
    for key, frame in source.partition_by("trade_date", as_dict=True, maintain_order=True).items():
        signal_date = key[0] if isinstance(key, tuple) else key
        frame = frame.sort("ts_code")
        codes = frame.get_column("ts_code").to_list()
        mu = frame.get_column("optimizer_mu").to_numpy()
        weights, buys, sells, objective = optimize_day(
            codes,
            mu,
            previous,
            args.invested_weight,
            args.max_weight,
            buy_cost,
            sell_cost,
        )
        execution_date = frame.get_column("execution_date")[0]
        previous = {
            code: float(weight)
            for code, weight in zip(codes, weights)
            if weight > 1e-12
        }
        for code, weight, forecast in zip(codes, weights, mu):
            if weight > 1e-12:
                rows.append(
                    {
                        "trade_date": signal_date,
                        "execution_date": execution_date,
                        "ts_code": code,
                        "target_weight": float(weight),
                        "mu": float(forecast),
                        "signal": signal,
                        "optimizer": "mu_minus_asymmetric_turnover_cost_no_risk_v1",
                    }
                )
        diagnostics.append(
            {
                "trade_date": signal_date,
                "execution_date": execution_date,
                "buy_turnover_target": buys,
                "sell_turnover_target": sells,
                "objective": objective,
                "holding_count": len(previous),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.output / "target_weights.parquet", compression="zstd")
    pl.DataFrame(diagnostics).write_parquet(args.output / "optimizer_daily.parquet", compression="zstd")
    summary = {
        "optimizer": "exact separable mu-minus-asymmetric-turnover-cost, no risk term",
        "predictions": str(args.predictions.resolve()),
        "signal": signal,
        "h1_weight": args.h1_weight if signal == "h1h5" else None,
        "h5_weight": args.h5_weight if signal == "h1h5" else None,
        "buy_bps": args.buy_bps,
        "sell_bps": args.sell_bps,
        "max_weight": args.max_weight,
        "invested_weight": args.invested_weight,
        "days": len(diagnostics),
        "mean_target_buy_turnover": float(np.mean([row["buy_turnover_target"] for row in diagnostics])),
        "mean_target_sell_turnover": float(np.mean([row["sell_turnover_target"] for row in diagnostics])),
    }
    (args.output / "optimizer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
