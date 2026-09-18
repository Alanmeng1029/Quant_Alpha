# Raw105 CSI500/CSI1000 corrected-v2 backtest record

## Model and data contract

- Model: one joint Rust LightGBM trained on point-in-time CSI500 and CSI1000 members.
- Features: 60 raw daily factors plus 45 raw minute factors.
- Training: 756 trading dates, six-date label maturity gap, quarterly OOS refit, H1/H5 equal score blend.
- Labels: QFQ open-to-open economic excess returns versus CSI500.
- Explicit exclusion: `000937.SZ`.
- OOS portfolio period: 2021-04-02 through 2026-08-28, 1,311 sessions.
- Costs: buy 2.1 bp, sell 7.1 bp; cash reserve 2%; 100-share lots.
- Metric convention: daily portfolio return less CSI500 daily return, compounded; Sharpe annualized with 243.
- Canonical model root: `results/predict/research-oos-raw-csi500-csi1000-rust-corrected-v2`.

The corrected feature panel has 2,995,441 rows, 2,080 dates, 105 features, and 22 quarterly OOS windows.

## Portfolio definitions

| Portfolio | CSI500 sleeve | CSI1000 sleeve | Capital blend |
|---|---|---|---|
| Pure CSI500 | Top100, exit beyond 120, max 3 replacements/day | ignored in top-level NAV | 100/0 |
| Pure CSI1000 | ignored in top-level NAV | Top100, exit beyond 200, max 5 replacements/day | 0/100 |
| Independent 80/20 | Top100, exit beyond 120, max 3/day | Top100, exit beyond 200, max 5/day | 80/20 |

The 80/20 portfolio maintains two independent real-holdings accounts and blends their daily returns.  With the 2% reserve, normal NAV exposure is approximately 78.4% CSI500 stocks, 19.6% CSI1000 stocks, and 2% cash.

## Corrected results

| Portfolio | Total return | Compounded excess | Excess Sharpe | Excess max drawdown | Avg buy turnover | 2025 excess | 2026 excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure CSI500 | 103.33% | 54.05% | 0.774 | 30.64% | 3.12% | -7.46% | -4.05% |
| Pure CSI1000 | 122.84% | 73.89% | 0.803 | 34.40% | 5.12% | -1.90% | -8.71% |
| Independent 80/20 | 108.68% | 59.09% | 0.822 | 31.42% | 3.52% | -6.25% | -5.09% |

Pure CSI1000 excess is deliberately measured against CSI500, matching the single benchmark convention used by every strategy report.

## Annual compounded excess

| Year | Pure CSI500 | Pure CSI1000 | Independent 80/20 |
|---|---:|---:|---:|
| 2021 partial | 5.31% | 21.37% | 8.31% |
| 2022 | 22.89% | 33.43% | 25.81% |
| 2023 | 16.40% | 17.59% | 16.14% |
| 2024 | 15.18% | 1.97% | 12.97% |
| 2025 | -7.46% | -1.90% | -6.25% |
| 2026 through Aug 28 | -4.05% | -8.71% | -5.09% |

## Interpretation boundary

The corrected run confirms that the earlier large divergence was caused by the omitted infeasible-symbol filter.  It does not resolve the economic deterioration after 2025: both sleeves remain negative versus CSI500 in 2026, and CSI1000 is weaker.  Future model or portfolio changes must compare against these corrected-v2 artifacts rather than any removed debug run.

The HTML comparison is stored at `results/predict/research-oos-raw-csi500-csi1000-rust-corrected-v2/report/report.html`.
