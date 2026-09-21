"""Point-in-time, stock-level factor risk model.

The model keeps the covariance in factor form::

    Sigma = X F X' + D

This avoids materialising a dense stock covariance matrix and makes portfolio
risk attribution auditable.  All estimators use observations strictly before
``as_of`` so a snapshot can safely be consumed by a close-to-next-open process.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class FactorRiskConfig:
    lookback: int = 252
    half_life: float = 63.0
    minimum_factor_observations: int = 126
    covariance_shrinkage: float = 0.20
    specific_half_life: float = 63.0
    minimum_specific_observations: int = 20
    specific_prior_observations: float = 60.0
    specific_floor_multiplier: float = 0.25
    specific_cap_multiplier: float = 4.0

    def validate(self) -> None:
        if self.lookback <= 1:
            raise ValueError("lookback must be greater than one")
        if self.half_life <= 0.0 or self.specific_half_life <= 0.0:
            raise ValueError("half lives must be positive")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if self.minimum_factor_observations <= 1:
            raise ValueError("minimum_factor_observations must be greater than one")
        if self.minimum_specific_observations < 1:
            raise ValueError("minimum_specific_observations must be positive")
        if self.specific_prior_observations < 0.0:
            raise ValueError("specific_prior_observations must be non-negative")


@dataclass(frozen=True)
class FactorRiskSnapshot:
    as_of: pd.Timestamp
    securities: tuple[str, ...]
    factor_names: tuple[str, ...]
    exposures: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray
    diagnostics: Mapping[str, float | int | str]

    def covariance(self) -> np.ndarray:
        """Materialise the stock covariance matrix for diagnostics only."""

        result = self.exposures @ self.factor_covariance @ self.exposures.T
        result[np.diag_indices_from(result)] += self.specific_variance
        return result


@dataclass(frozen=True)
class PortfolioRiskForecast:
    daily_variance: float
    annualized_volatility: float
    factor_variance: float
    specific_variance: float
    factor_exposures: Mapping[str, float]
    factor_contributions: Mapping[str, float]


def _ew_weights(length: int, half_life: float) -> np.ndarray:
    age = np.arange(length - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-math.log(2.0) * age / half_life)
    return weights / weights.sum()


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.diag(symmetric))), 1e-12)
    clipped = np.maximum(eigenvalues, scale * 1e-10)
    return (eigenvectors * clipped) @ eigenvectors.T


def estimate_cross_sectional_factor_model(
    frame: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    return_column: str = "future_excess_return",
    weight_column: str = "regression_weight",
    security_column: str = "ts_code",
    minimum_observations: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate daily market/style returns and stock-specific residuals.

    Each input date represents an exposure date.  The dependent return must be
    the subsequent holding-period return aligned back to that exposure date.
    """

    required = {"date", security_column, return_column, weight_column, *exposure_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not exposure_columns:
        raise ValueError("exposure_columns must not be empty")

    factor_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    columns = [security_column, return_column, weight_column, *exposure_columns]
    for date, group in frame.groupby("date", sort=True):
        values = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
        weights = values[weight_column].to_numpy(dtype=np.float64)
        valid = np.isfinite(weights) & (weights > 0.0)
        values = values.loc[valid]
        weights = weights[valid]
        if len(values) < minimum_observations:
            continue

        y = values[return_column].to_numpy(dtype=np.float64)
        x = values[list(exposure_columns)].to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(values), dtype=np.float64), x])
        sqrt_weights = np.sqrt(weights / np.mean(weights))
        weighted_design = design * sqrt_weights[:, None]
        coefficients, *_ = np.linalg.lstsq(
            weighted_design, y * sqrt_weights, rcond=None
        )
        fitted = design @ coefficients
        residual = y - fitted
        weighted_sse = float(np.dot(weights, residual * residual))
        centered = y - float(np.average(y, weights=weights))
        weighted_sst = float(np.dot(weights, centered * centered))
        factor_row: dict[str, object] = {
            "date": date,
            "market_factor_return": float(coefficients[0]),
            "cross_sectional_observations": int(len(values)),
            "cross_sectional_r_squared": (
                1.0 - weighted_sse / weighted_sst if weighted_sst > 0.0 else math.nan
            ),
            "design_condition_number": float(np.linalg.cond(weighted_design)),
        }
        factor_row.update(
            {
                f"{name}_factor_return": float(coefficients[index + 1])
                for index, name in enumerate(exposure_columns)
            }
        )
        factor_rows.append(factor_row)
        residual_rows.extend(
            {
                "date": date,
                "ts_code": code,
                "specific_return": float(value),
            }
            for code, value in zip(values[security_column], residual, strict=True)
        )
    return pd.DataFrame(factor_rows), pd.DataFrame(residual_rows)


def estimate_factor_covariance(
    factor_returns: pd.DataFrame,
    factor_return_columns: Sequence[str],
    as_of: object,
    config: FactorRiskConfig = FactorRiskConfig(),
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Estimate a shrunk EWMA factor covariance using dates before ``as_of``."""

    config.validate()
    required = {"date", *factor_return_columns}
    missing = required.difference(factor_returns.columns)
    if missing:
        raise ValueError(f"missing factor-return columns: {sorted(missing)}")
    cutoff = pd.Timestamp(as_of)
    history = factor_returns.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.loc[history["date"] < cutoff].sort_values("date").tail(config.lookback)
    values = history[list(factor_return_columns)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < config.minimum_factor_observations:
        raise ValueError(
            f"only {len(values)} factor-return observations before {cutoff.date()}; "
            f"need {config.minimum_factor_observations}"
        )
    array = values.to_numpy(dtype=np.float64)
    weights = _ew_weights(len(array), config.half_life)
    centered = array - np.average(array, axis=0, weights=weights)
    raw = (centered * weights[:, None]).T @ centered
    target = np.diag(np.diag(raw))
    shrunk = (1.0 - config.covariance_shrinkage) * raw + config.covariance_shrinkage * target
    covariance = _nearest_psd(shrunk)
    diagnostics: dict[str, float | int] = {
        "factor_observations": int(len(array)),
        "factor_covariance_min_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        "factor_covariance_condition_number": float(np.linalg.cond(covariance)),
    }
    return covariance, diagnostics


def estimate_specific_variances(
    specific_returns: pd.DataFrame,
    securities: Sequence[str],
    as_of: object,
    config: FactorRiskConfig = FactorRiskConfig(),
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float | int]]:
    """Estimate EWMA specific variances shrunk toward the cross-sectional median."""

    config.validate()
    required = {"date", "ts_code", "specific_return"}
    missing = required.difference(specific_returns.columns)
    if missing:
        raise ValueError(f"missing specific-return columns: {sorted(missing)}")
    cutoff = pd.Timestamp(as_of)
    history = specific_returns.copy()
    history["date"] = pd.to_datetime(history["date"])
    eligible_dates = (
        history.loc[history["date"] < cutoff, "date"].drop_duplicates().sort_values().tail(config.lookback)
    )
    history = history.loc[
        history["date"].isin(eligible_dates) & history["ts_code"].isin(securities)
    ].sort_values(["ts_code", "date"])

    raw_by_security: dict[str, tuple[float, int]] = {}
    for security, group in history.groupby("ts_code", sort=False):
        residual = group["specific_return"].to_numpy(dtype=np.float64)
        residual = residual[np.isfinite(residual)]
        if len(residual) == 0:
            continue
        weights = _ew_weights(len(residual), config.specific_half_life)
        raw_by_security[str(security)] = (float(np.dot(weights, residual * residual)), len(residual))

    eligible_raw = np.array(
        [variance for variance, count in raw_by_security.values() if count >= config.minimum_specific_observations],
        dtype=np.float64,
    )
    if len(eligible_raw) == 0:
        raise ValueError(f"no securities have {config.minimum_specific_observations} specific-return observations")
    prior = float(np.median(eligible_raw))
    lower = max(float(np.quantile(eligible_raw, 0.10)) * config.specific_floor_multiplier, 1e-12)
    upper = max(float(np.quantile(eligible_raw, 0.90)) * config.specific_cap_multiplier, lower)

    rows: list[dict[str, object]] = []
    result: list[float] = []
    for security in securities:
        raw, observations = raw_by_security.get(str(security), (prior, 0))
        confidence = observations / (observations + config.specific_prior_observations)
        shrunk = float(np.clip(confidence * raw + (1.0 - confidence) * prior, lower, upper))
        result.append(shrunk)
        rows.append(
            {
                "ts_code": str(security),
                "specific_variance": shrunk,
                "raw_specific_variance": raw,
                "specific_observations": observations,
                "shrinkage_to_prior": 1.0 - confidence,
            }
        )
    diagnostics: dict[str, float | int] = {
        "specific_prior_variance": prior,
        "specific_variance_floor": lower,
        "specific_variance_cap": upper,
        "specific_missing_histories": int(sum(row["specific_observations"] == 0 for row in rows)),
    }
    return np.asarray(result, dtype=np.float64), pd.DataFrame(rows), diagnostics


def build_factor_risk_snapshot(
    as_of: object,
    exposures: pd.DataFrame,
    exposure_columns: Sequence[str],
    factor_returns: pd.DataFrame,
    specific_returns: pd.DataFrame,
    config: FactorRiskConfig = FactorRiskConfig(),
) -> tuple[FactorRiskSnapshot, pd.DataFrame]:
    """Build one point-in-time stock risk snapshot."""

    required = {"ts_code", *exposure_columns}
    missing = required.difference(exposures.columns)
    if missing:
        raise ValueError(f"missing exposure columns: {sorted(missing)}")
    current = exposures.loc[:, ["ts_code", *exposure_columns]].copy()
    current = current.replace([np.inf, -np.inf], np.nan).dropna()
    current["ts_code"] = current["ts_code"].astype(str)
    current = current.drop_duplicates("ts_code", keep="last").sort_values("ts_code")
    if current.empty:
        raise ValueError("no complete current exposures")

    factor_return_columns = ["market_factor_return", *[f"{name}_factor_return" for name in exposure_columns]]
    covariance, factor_diagnostics = estimate_factor_covariance(
        factor_returns, factor_return_columns, as_of, config
    )
    securities = tuple(current["ts_code"].tolist())
    specific, specific_frame, specific_diagnostics = estimate_specific_variances(
        specific_returns, securities, as_of, config
    )
    style = current[list(exposure_columns)].to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(current), dtype=np.float64), style])
    diagnostics: dict[str, float | int | str] = {
        "as_of": str(pd.Timestamp(as_of).date()),
        "securities": len(securities),
        "factors": design.shape[1],
        **factor_diagnostics,
        **specific_diagnostics,
    }
    snapshot = FactorRiskSnapshot(
        as_of=pd.Timestamp(as_of),
        securities=securities,
        factor_names=("market", *tuple(exposure_columns)),
        exposures=design,
        factor_covariance=covariance,
        specific_variance=specific,
        diagnostics=diagnostics,
    )
    return snapshot, specific_frame


def forecast_portfolio_risk(
    snapshot: FactorRiskSnapshot,
    active_weights: Mapping[str, float] | pd.Series,
) -> PortfolioRiskForecast:
    """Forecast active portfolio risk and factor contributions."""

    weights = np.asarray([float(active_weights.get(code, 0.0)) for code in snapshot.securities])
    factor_exposure = snapshot.exposures.T @ weights
    covariance_times_exposure = snapshot.factor_covariance @ factor_exposure
    contributions = factor_exposure * covariance_times_exposure
    factor_variance = max(float(factor_exposure @ covariance_times_exposure), 0.0)
    specific_variance = max(float(np.dot(weights * weights, snapshot.specific_variance)), 0.0)
    total = max(factor_variance + specific_variance, 0.0)
    return PortfolioRiskForecast(
        daily_variance=total,
        annualized_volatility=math.sqrt(total * TRADING_DAYS),
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        factor_exposures=dict(zip(snapshot.factor_names, factor_exposure, strict=True)),
        factor_contributions=dict(zip(snapshot.factor_names, contributions, strict=True)),
    )


def write_factor_risk_snapshot(
    snapshot: FactorRiskSnapshot,
    specific_frame: pd.DataFrame,
    output: Path,
    config: FactorRiskConfig,
) -> None:
    """Write compact, portable snapshot artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    exposure_frame = pd.DataFrame(snapshot.exposures, columns=snapshot.factor_names)
    exposure_frame.insert(0, "ts_code", snapshot.securities)
    exposure_frame.to_parquet(output / "exposures.parquet", index=False)
    pd.DataFrame(
        snapshot.factor_covariance,
        index=snapshot.factor_names,
        columns=snapshot.factor_names,
    ).rename_axis("factor").reset_index().to_parquet(output / "factor_covariance.parquet", index=False)
    specific_frame.to_parquet(output / "specific_variance.parquet", index=False)
    payload = {"config": asdict(config), "diagnostics": dict(snapshot.diagnostics)}
    (output / "diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "_SUCCESS").write_text("", encoding="utf-8")
