from __future__ import annotations

import numpy as np
import pandas as pd

from a_share_data.factor_risk_model import (
    FactorRiskConfig,
    build_factor_risk_snapshot,
    estimate_cross_sectional_factor_model,
    forecast_portfolio_risk,
)


def _synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(19)
    dates = pd.date_range("2020-01-01", periods=180, freq="B")
    codes = [f"S{i:03d}" for i in range(120)]
    exposure = rng.normal(size=len(codes))
    rows = []
    for date in dates:
        factor_return = rng.normal(0.0, 0.01)
        residual = rng.normal(0.0, 0.004, len(codes))
        for code, value, error in zip(codes, exposure, residual, strict=True):
            rows.append(
                {
                    "date": date,
                    "ts_code": code,
                    "size": value,
                    "regression_weight": 1.0,
                    "future_excess_return": 0.001 + value * factor_return + error,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame({"ts_code": codes, "size": exposure})


def test_snapshot_is_psd_and_portfolio_identity_holds() -> None:
    panel, current = _synthetic_panel()
    factors, residuals = estimate_cross_sectional_factor_model(panel, ["size"])
    config = FactorRiskConfig(
        lookback=126,
        minimum_factor_observations=80,
        minimum_specific_observations=20,
    )
    snapshot, _ = build_factor_risk_snapshot(
        panel["date"].max() + pd.Timedelta(days=1),
        current,
        ["size"],
        factors,
        residuals,
        config,
    )
    covariance = snapshot.covariance()
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0.0

    weights = {code: (0.01 if index < 10 else 0.0) for index, code in enumerate(snapshot.securities)}
    forecast = forecast_portfolio_risk(snapshot, weights)
    vector = np.array([weights[code] for code in snapshot.securities])
    assert np.isclose(forecast.daily_variance, vector @ covariance @ vector)
    assert np.isclose(
        forecast.factor_variance,
        sum(forecast.factor_contributions.values()),
    )


def test_snapshot_does_not_use_as_of_or_future_returns() -> None:
    panel, current = _synthetic_panel()
    factors, residuals = estimate_cross_sectional_factor_model(panel, ["size"])
    as_of = factors.iloc[-1]["date"]
    config = FactorRiskConfig(
        lookback=126,
        minimum_factor_observations=80,
        minimum_specific_observations=20,
    )
    original, _ = build_factor_risk_snapshot(
        as_of, current, ["size"], factors, residuals, config
    )
    factors.loc[factors["date"] >= as_of, "size_factor_return"] = 100.0
    residuals.loc[residuals["date"] >= as_of, "specific_return"] = 100.0
    revised, _ = build_factor_risk_snapshot(
        as_of, current, ["size"], factors, residuals, config
    )
    assert np.allclose(original.factor_covariance, revised.factor_covariance)
    assert np.allclose(original.specific_variance, revised.specific_variance)
