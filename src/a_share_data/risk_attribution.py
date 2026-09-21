"""Small, auditable risk-attribution models for strategy return series."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketRiskConfig:
    lookback: int = 252
    half_life: float = 60.0
    forward_window: int = 20
    minimum_observations: int = 126


def _ew_weights(length: int, half_life: float) -> np.ndarray:
    age = np.arange(length - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-math.log(2.0) * age / half_life)
    return weights / weights.sum()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    centered = values - _weighted_mean(values, weights)
    return float(np.dot(weights, centered * centered))


def rolling_market_risk_attribution(
    frame: pd.DataFrame,
    config: MarketRiskConfig = MarketRiskConfig(),
) -> pd.DataFrame:
    """Estimate a lagged one-factor market model and attribute forecast variance.

    ``frame`` must contain ``date``, ``portfolio_return`` and ``market_return``.
    The estimate for row *t* uses only rows before *t*.  The optional forward
    realised volatility is an evaluation target and never enters the estimate.
    """

    required = {"date", "portfolio_return", "market_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    data = frame.loc[:, ["date", "portfolio_return", "market_return"]].copy()
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    y_all = data["portfolio_return"].to_numpy(dtype=np.float64)
    x_all = data["market_return"].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []

    for index in range(len(data)):
        start = max(0, index - config.lookback)
        y = y_all[start:index]
        x = x_all[start:index]
        valid = np.isfinite(y) & np.isfinite(x)
        y = y[valid]
        x = x[valid]
        if len(y) < config.minimum_observations:
            continue

        weights = _ew_weights(len(y), config.half_life)
        design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
        sqrt_weights = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * sqrt_weights[:, None], y * sqrt_weights, rcond=None
        )
        alpha, beta = (float(coefficients[0]), float(coefficients[1]))
        residual = y - design @ coefficients
        market_variance = _weighted_variance(x, weights)
        specific_variance = _weighted_variance(residual, weights)
        factor_variance = beta * beta * market_variance
        predicted_variance = max(factor_variance + specific_variance, 1e-12)
        historical_variance = max(_weighted_variance(y, weights), 1e-12)
        expected_return = _weighted_mean(y, weights)

        future = y_all[index : index + config.forward_window]
        future = future[np.isfinite(future)]
        forward_volatility = (
            float(np.std(future, ddof=1) * math.sqrt(TRADING_DAYS))
            if len(future) == config.forward_window
            else math.nan
        )
        realised = y_all[index]
        surprise_squared = (
            float((realised - expected_return) ** 2) if np.isfinite(realised) else math.nan
        )
        total_variance = _weighted_variance(y, weights)
        r_squared = (
            1.0 - specific_variance / total_variance if total_variance > 0.0 else math.nan
        )
        rows.append(
            {
                "date": data.iloc[index]["date"],
                "portfolio_return": realised,
                "market_return": x_all[index],
                "expected_return": expected_return,
                "alpha_daily": alpha,
                "beta": beta,
                "r_squared": r_squared,
                "market_variance": market_variance,
                "factor_variance": factor_variance,
                "specific_variance": specific_variance,
                "predicted_variance": predicted_variance,
                "historical_variance": historical_variance,
                "predicted_volatility_ann": math.sqrt(predicted_variance * TRADING_DAYS),
                "forward_20d_volatility_ann": forward_volatility,
                "factor_risk_share": factor_variance / predicted_variance,
                "specific_risk_share": specific_variance / predicted_variance,
                "surprise_squared": surprise_squared,
                "within_95": (
                    abs(realised - expected_return) <= 1.96 * math.sqrt(predicted_variance)
                    if np.isfinite(realised)
                    else False
                ),
            }
        )

    return pd.DataFrame(rows)


def rolling_factor_risk_attribution(
    frame: pd.DataFrame,
    factor_columns: list[str],
    config: MarketRiskConfig = MarketRiskConfig(),
) -> pd.DataFrame:
    """Estimate a lagged multi-factor time-series risk model.

    Factor variance is attributed with Euler contributions
    ``beta_i * (covariance @ beta)_i``. Contributions may be negative when
    factors hedge one another, while their sum equals total factor variance.
    """

    required = {"date", "portfolio_return", *factor_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not factor_columns:
        raise ValueError("factor_columns must not be empty")

    columns = ["date", "portfolio_return", *factor_columns]
    data = frame.loc[:, columns].copy().sort_values("date").drop_duplicates("date")
    y_all = data["portfolio_return"].to_numpy(dtype=np.float64)
    factors_all = data[factor_columns].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []

    for index in range(len(data)):
        start = max(0, index - config.lookback)
        y = y_all[start:index]
        factors = factors_all[start:index]
        valid = np.isfinite(y) & np.isfinite(factors).all(axis=1)
        y = y[valid]
        factors = factors[valid]
        if len(y) < config.minimum_observations:
            continue

        weights = _ew_weights(len(y), config.half_life)
        design = np.column_stack([np.ones(len(y), dtype=np.float64), factors])
        sqrt_weights = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * sqrt_weights[:, None], y * sqrt_weights, rcond=None
        )
        alpha = float(coefficients[0])
        betas = coefficients[1:]
        residual = y - design @ coefficients
        centered_factors = factors - np.average(factors, axis=0, weights=weights)
        factor_covariance = (centered_factors * weights[:, None]).T @ centered_factors
        covariance_times_beta = factor_covariance @ betas
        contributions = betas * covariance_times_beta
        factor_variance = max(float(betas @ covariance_times_beta), 0.0)
        specific_variance = max(_weighted_variance(residual, weights), 0.0)
        predicted_variance = max(factor_variance + specific_variance, 1e-12)
        expected_return = _weighted_mean(y, weights)
        total_variance = _weighted_variance(y, weights)
        realised = y_all[index]
        future = y_all[index : index + config.forward_window]
        future = future[np.isfinite(future)]

        row: dict[str, object] = {
            "date": data.iloc[index]["date"],
            "portfolio_return": realised,
            "expected_return": expected_return,
            "alpha_daily": alpha,
            "r_squared": (
                1.0 - specific_variance / total_variance if total_variance > 0.0 else math.nan
            ),
            "factor_variance": factor_variance,
            "specific_variance": specific_variance,
            "predicted_variance": predicted_variance,
            "historical_variance": max(total_variance, 1e-12),
            "predicted_volatility_ann": math.sqrt(predicted_variance * TRADING_DAYS),
            "forward_20d_volatility_ann": (
                float(np.std(future, ddof=1) * math.sqrt(TRADING_DAYS))
                if len(future) == config.forward_window
                else math.nan
            ),
            "factor_risk_share": factor_variance / predicted_variance,
            "specific_risk_share": specific_variance / predicted_variance,
            "surprise_squared": (
                float((realised - expected_return) ** 2)
                if np.isfinite(realised)
                else math.nan
            ),
            "within_95": (
                abs(realised - expected_return) <= 1.96 * math.sqrt(predicted_variance)
                if np.isfinite(realised)
                else False
            ),
        }
        for factor_index, factor_name in enumerate(factor_columns):
            row[f"beta_{factor_name}"] = float(betas[factor_index])
            row[f"variance_contribution_{factor_name}"] = float(contributions[factor_index])
            row[f"risk_share_{factor_name}"] = float(
                contributions[factor_index] / predicted_variance
            )
            row[factor_name] = float(factors_all[index, factor_index])
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_cross_sectional_factor_returns(
    frame: pd.DataFrame,
    exposure_columns: list[str],
    *,
    return_column: str = "future_excess_return",
    weight_column: str = "regression_weight",
    minimum_observations: int = 100,
) -> pd.DataFrame:
    """Estimate daily style-factor returns with independent WLS regressions."""

    required = {"date", return_column, weight_column, *exposure_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date", sort=True):
        columns = [return_column, weight_column, *exposure_columns]
        values = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < minimum_observations:
            continue
        y = values[return_column].to_numpy(dtype=np.float64)
        exposures = values[exposure_columns].to_numpy(dtype=np.float64)
        observation_weights = values[weight_column].to_numpy(dtype=np.float64)
        valid_weights = np.isfinite(observation_weights) & (observation_weights > 0.0)
        y = y[valid_weights]
        exposures = exposures[valid_weights]
        observation_weights = observation_weights[valid_weights]
        if len(y) < minimum_observations:
            continue
        design = np.column_stack([np.ones(len(y), dtype=np.float64), exposures])
        sqrt_weights = np.sqrt(observation_weights / np.mean(observation_weights))
        coefficients, *_ = np.linalg.lstsq(
            design * sqrt_weights[:, None], y * sqrt_weights, rcond=None
        )
        residual = y - design @ coefficients
        row: dict[str, object] = {
            "date": date,
            "cross_sectional_intercept": float(coefficients[0]),
            "cross_sectional_observations": int(len(y)),
            "cross_sectional_residual_std": float(np.std(residual, ddof=len(coefficients))),
        }
        for index, name in enumerate(exposure_columns):
            row[f"{name}_factor_return"] = float(coefficients[index + 1])
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_attribution(daily: pd.DataFrame) -> dict[str, float | int | str]:
    valid = daily.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["predicted_variance", "surprise_squared"]
    )
    forward = valid.dropna(subset=["forward_20d_volatility_ann"])
    variance_ratio = float(valid["surprise_squared"].sum() / valid["predicted_variance"].sum())
    qlike = np.log(valid["predicted_variance"]) + (
        valid["surprise_squared"] / valid["predicted_variance"]
    )
    beta_column = "beta_market" if "beta_market" in valid.columns else "beta"
    return {
        "start_date": str(valid["date"].min()),
        "end_date": str(valid["date"].max()),
        "observations": int(len(valid)),
        "mean_beta": float(valid[beta_column].mean()),
        "mean_r_squared": float(valid["r_squared"].mean()),
        "mean_predicted_volatility_ann": float(valid["predicted_volatility_ann"].mean()),
        "mean_forward_20d_volatility_ann": float(forward["forward_20d_volatility_ann"].mean()),
        "volatility_calibration_ratio": math.sqrt(variance_ratio),
        "predicted_vs_forward_volatility_corr": float(
            forward["predicted_volatility_ann"].corr(forward["forward_20d_volatility_ann"])
        ),
        "coverage_95": float(valid["within_95"].mean()),
        "mean_factor_risk_share": float(valid["factor_risk_share"].mean()),
        "mean_specific_risk_share": float(valid["specific_risk_share"].mean()),
        "mean_qlike": float(qlike.mean()),
    }
