# Fixed-98 LGBM rolling OOS, 2026-09-08

> 历史基线：本实验用于确定 98 因子阶段采用默认 LightGBM，而不是年度 Rank-IC 调参版本。98 因子集现已由 105 因子 `o2o_daily60_minute45_v2` 替代；当前正式结果见 [`../PRODUCTION_BACKTEST.md`](../PRODUCTION_BACKTEST.md)。

## Scope

- Input: formal `o2o_daily60_minute38_v1` cache: 60 daily factors plus 38 minute-v1 factors; minute-v2 is excluded.
- Labels: H1/H5 open-to-open excess return versus CSI500.
- Outer test: 2021-04-01 through 2026-08-28, 22 quarterly blocks, 756 prior trading dates and a six-trading-day label-maturity gap.
- Inner validation: three sequential 63-day blocks in the final 189 training dates, each separated from its fitting segment by six trading dates.
- Execution: unchanged Optimizer V2, 80 names, entry 80, exit 96, max five replacements, 100-share lots, charged and independent zero-cost accounts.

## Models compared

1. `lgbm_default98`: fixed default tree structure; quarterly early stopping still uses the final validation segment.
2. `lgbm_tuned98`: 12 pre-drawn LGBM configurations searched annually, separately for H1/H5, selected by three-fold mean daily Rank IC and frozen during the remainder of that year.

## OOS result

| Model | H1 Rank IC | H5 Rank IC | H1/H5 annualized ICIR | V2 net return | CSI500 IR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default LGBM | 0.03392 | 0.02991 | 4.23 / 2.94 | 71.38% | 0.627 |
| Annual Rank-IC tuned LGBM | 0.03464 | 0.03104 | 4.07 / 3.07 | 57.90% | 0.536 |

The tuned-minus-default 20-trading-day block bootstrap intervals include zero: H1 +0.00072, 95% CI [-0.00312, 0.00447]; H5 +0.00113, 95% CI [-0.00652, 0.00928]. It is not evidence that annual Rank-IC tuning improves generalization.

## Diagnosis

- Daily Top80 overlap is only 58.4%; Top96 overlap is 61.5%.
- Default Top80 has mean future blended excess return 8.74 bp/day versus 5.24 bp/day for the tuned model, a -3.50 bp/day difference for the tuned selection.
- Costs are nearly identical (about 5.8% cumulative), so the V2 gap is selection quality, not execution cost.

## Decision

在 98 因子版本内部保留 `lgbm_default98`，不晋级 Rank-IC 调参版本。该结论随后成为 105 因子版本继续采用默认 LightGBM 的历史依据，但本页不再代表当前正式因子集。

MLP was paused because this environment reports `torch.backends.mps.is_available() == False`, so it would train on CPU. A follow-up LGBM experiment selecting annual parameters by validation Top80 future return was started but intentionally stopped before results and its partial outputs were deleted. It must be restarted as a fresh exploratory experiment if needed.
