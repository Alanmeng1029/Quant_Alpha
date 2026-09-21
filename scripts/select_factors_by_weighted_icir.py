"""Select a fixed factor pool by exponentially weighted daily Rank-ICIR.

This is the selection rule used by the original Daily60 research catalog:
rank factors on the maximum absolute annualized Rank-ICIR across H1 and H5,
using the latest 504 trading days through 2024-12-31 and a 126-day half-life.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--universe", default="csi300_csi500")
    parser.add_argument("--return-kind", default="open_to_open_raw")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--as-of", default="2024-12-31")
    parser.add_argument("--window-days", type=int, default=504)
    parser.add_argument("--half-life-days", type=float, default=126.0)
    parser.add_argument(
        "--min-abs-icir",
        type=float,
        help="Keep factors whose maximum absolute H1/H5 ICIR exceeds this threshold.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=60,
        help="Maximum candidates after thresholding; use 0 for no count limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.source.glob(f"*/{args.universe}/daily_ic.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no daily_ic.parquet files found for {args.universe} under {args.source}"
        )

    frames = []
    for path in paths:
        factor_id = path.parts[-3]
        frames.append(
            pl.read_parquet(path)
            .filter(
                (pl.col("return_kind") == args.return_kind)
                & pl.col("horizon").is_in(args.horizons)
            )
            .select(
                pl.lit(factor_id).alias("factor_id"),
                "trade_date",
                "horizon",
                "sample_count",
                "rank_ic",
            )
        )

    history = pl.concat(frames).drop_nulls("rank_ic").sort(
        ["factor_id", "horizon", "trade_date"]
    )
    cutoff = history.filter(pl.col("trade_date") <= pl.lit(args.as_of).str.to_date())
    dates = cutoff.get_column("trade_date").unique().sort().tail(args.window_days)
    first_date = dates.min()
    last_date = dates.max()
    window = cutoff.filter(pl.col("trade_date").is_between(first_date, last_date))

    scores = []
    for (factor_id, horizon), group in window.partition_by(
        ["factor_id", "horizon"], as_dict=True
    ).items():
        values = group.sort("trade_date").get_column("rank_ic").to_numpy()
        weights = 0.5 ** (
            (len(values) - 1 - np.arange(len(values))) / args.half_life_days
        )
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        scores.append(
            {
                "factor_id": factor_id,
                "horizon": horizon,
                "days": len(values),
                "weighted_rank_ic": mean,
                "weighted_rank_icir": mean / std * np.sqrt(252)
                if std > 1e-12
                else 0.0,
            }
        )

    score = pl.DataFrame(scores).with_columns(
        pl.col("weighted_rank_icir").abs().alias("abs_weighted_rank_icir")
    )
    candidates = (
        score.group_by("factor_id")
        .agg(
            pl.col("abs_weighted_rank_icir").max().alias("selection_score"),
            pl.col("weighted_rank_icir")
            .sort_by("abs_weighted_rank_icir", descending=True)
            .first()
            .alias("selected_horizon_icir"),
            pl.col("horizon")
            .sort_by("abs_weighted_rank_icir", descending=True)
            .first()
            .alias("selected_horizon"),
        )
        .sort(["selection_score", "factor_id"], descending=[True, False])
    )
    if args.min_abs_icir is not None:
        candidates = candidates.filter(pl.col("selection_score") > args.min_abs_icir)
    if args.max_candidates > 0:
        candidates = candidates.head(args.max_candidates)
    candidates = candidates.with_row_index("rank", offset=1)

    args.output.mkdir(parents=True, exist_ok=True)
    history.write_parquet(args.output / "daily_rank_ic.parquet", compression="zstd")
    score.sort(["factor_id", "horizon"]).write_csv(args.output / "factor_scores.csv")
    candidates.write_csv(args.output / "candidates.csv")
    (args.output / "factor_ids.txt").write_text(
        "# Fixed factor pool selected by exponentially weighted daily Rank-ICIR.\n"
        + "\n".join(candidates.get_column("factor_id").to_list())
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "method": "max_abs_exponentially_weighted_annualized_daily_rank_icir",
        "source": str(args.source),
        "universe": args.universe,
        "return_kind": args.return_kind,
        "horizons": args.horizons,
        "as_of": args.as_of,
        "selection_window_trading_days": args.window_days,
        "half_life_trading_days": args.half_life_days,
        "min_abs_icir": args.min_abs_icir,
        "max_candidates": args.max_candidates,
        "candidate_count": candidates.height,
        "history_rows": history.height,
        "first_window_date": str(first_date),
        "last_window_date": str(last_date),
    }
    (args.output / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "top": candidates.head(10).to_dicts()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
