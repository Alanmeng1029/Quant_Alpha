"""Compact factor-evaluation artifacts into a leakage-safe training candidate catalog."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


SOURCE = Path("results/factor_batches/existing-factors-open-open-v1")
OUTPUT = Path("A_stock_database/lake/derived/factor_research")
AS_OF = "2024-12-31"
WINDOW_DAYS = 504
HALF_LIFE_DAYS = 126
MAX_CANDIDATES = 60


def main() -> None:
    frames = []
    for path in sorted(SOURCE.glob("*/csi300_csi500/daily_ic.parquet")):
        factor_id = path.parts[-3]
        frames.append(pl.read_parquet(path).filter(
            (pl.col("return_kind") == "open_to_open_raw") & pl.col("horizon").is_in([1, 5])
        ).select(pl.lit(factor_id).alias("factor_id"), "trade_date", "horizon", "sample_count", "rank_ic"))
    history = pl.concat(frames).drop_nulls("rank_ic").sort(["factor_id", "horizon", "trade_date"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history.write_parquet(OUTPUT / "o2o_daily_rank_ic_h1_h5.parquet", compression="zstd")
    cutoff = history.filter(pl.col("trade_date") <= pl.lit(AS_OF).str.to_date())
    dates = cutoff.get_column("trade_date").unique().sort().tail(WINDOW_DAYS)
    window = cutoff.filter(pl.col("trade_date").is_in(dates))
    scores = []
    for (factor_id, horizon), group in window.partition_by(["factor_id", "horizon"], as_dict=True).items():
        values = group.sort("trade_date").get_column("rank_ic").to_numpy()
        weight = .5 ** ((len(values) - 1 - np.arange(len(values))) / HALF_LIFE_DAYS)
        mean = float(np.average(values, weights=weight))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=weight)))
        scores.append({"factor_id": factor_id, "horizon": horizon, "days": len(values), "weighted_rank_ic": mean, "weighted_rank_icir": mean / std * np.sqrt(252) if std > 1e-12 else 0.0})
    score = pl.DataFrame(scores).with_columns(pl.col("weighted_rank_icir").abs().alias("abs_weighted_rank_icir"))
    candidates = score.group_by("factor_id").agg(
        pl.col("abs_weighted_rank_icir").max().alias("selection_score"),
        pl.col("weighted_rank_icir").sort_by("abs_weighted_rank_icir", descending=True).first().alias("selected_horizon_icir"),
    ).sort("selection_score", descending=True).head(MAX_CANDIDATES).with_row_index("rank", offset=1)
    candidates.write_csv(OUTPUT / "candidates_pre2025_top60.csv")
    (OUTPUT / "candidates_pre2025_top60.txt").write_text("# Fixed research candidate pool; selected using O2O Rank IC through 2024-12-31 only.\n" + "\n".join(candidates["factor_id"].to_list()) + "\n", encoding="utf-8")
    (OUTPUT / "catalog_manifest.json").write_text(json.dumps({"as_of": AS_OF, "selection_window_trading_days": WINDOW_DAYS, "half_life_trading_days": HALF_LIFE_DAYS, "candidate_count": candidates.height, "history_rows": history.height, "source": str(SOURCE)}, indent=2), encoding="utf-8")
    print(json.dumps({"history_rows": history.height, "candidates": candidates.height, "top": candidates.head(10).to_dicts()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
