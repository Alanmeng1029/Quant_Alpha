from __future__ import annotations

import numpy as np
import pandas as pd

from a_share_data.risk_attribution import (
    MarketRiskConfig,
    estimate_cross_sectional_factor_returns,
    rolling_factor_risk_attribution,
    rolling_market_risk_attribution,
)


def test_market_risk_model_recovers_beta_and_has_no_lookahead() -> None:
    rng = np.random.default_rng(42)
    market = rng.normal(0.0, 0.01, 420)
    portfolio = 0.0002 + 1.2 * market + rng.normal(0.0, 0.004, 420)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=420, freq="B"),
            "portfolio_return": portfolio,
            "market_return": market,
        }
    )
    config = MarketRiskConfig(lookback=252, half_life=60, minimum_observations=126)
    original = rolling_market_risk_attribution(frame, config)

    changed = frame.copy()
    changed.loc[changed.index[-1], "portfolio_return"] = 10.0
    revised = rolling_market_risk_attribution(changed, config)

    assert 1.1 < original.iloc[-1]["beta"] < 1.3
    assert np.isclose(original.iloc[-1]["beta"], revised.iloc[-1]["beta"])
    assert np.isclose(
        original.iloc[-1]["predicted_variance"], revised.iloc[-1]["predicted_variance"]
    )
    assert np.allclose(
        original["factor_risk_share"] + original["specific_risk_share"], 1.0
    )


def test_cross_sectional_factor_returns_and_multifactor_variance_identity() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    factor_one = rng.normal(0.0, 0.01, len(dates))
    factor_two = rng.normal(0.0, 0.006, len(dates))
    portfolio = 0.8 * factor_one - 0.4 * factor_two + rng.normal(0.0, 0.002, len(dates))
    time_series = pd.DataFrame(
        {"date": dates, "portfolio_return": portfolio, "factor_one": factor_one, "factor_two": factor_two}
    )
    result = rolling_factor_risk_attribution(
        time_series,
        ["factor_one", "factor_two"],
        MarketRiskConfig(lookback=252, half_life=60, minimum_observations=126),
    )
    final = result.iloc[-1]
    assert 0.7 < final["beta_factor_one"] < 0.9
    assert -0.5 < final["beta_factor_two"] < -0.3
    contributions = (
        final["variance_contribution_factor_one"]
        + final["variance_contribution_factor_two"]
        + final["specific_variance"]
    )
    assert np.isclose(contributions, final["predicted_variance"])

    cross_section = []
    for date in dates[:3]:
        exposure = rng.normal(size=200)
        for value in exposure:
            cross_section.append(
                {
                    "date": date,
                    "future_excess_return": 0.003 + 0.02 * value,
                    "regression_weight": 1.0,
                    "size": value,
                }
            )
    estimated = estimate_cross_sectional_factor_returns(pd.DataFrame(cross_section), ["size"])
    assert np.allclose(estimated["size_factor_return"], 0.02)
