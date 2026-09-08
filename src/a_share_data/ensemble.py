"""Leakage-safe rolling feature selection and compact neural regressors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SelectionSettings:
    min_coverage: float = 0.85
    min_abs_icir: float = 0.50
    max_features: int = 40
    max_abs_correlation: float = 0.90
    correlation_lookback_days: int = 60
    use_all_features: bool = False


def select_features(train: pl.DataFrame, features: tuple[str, ...], target: str, settings: SelectionSettings = SelectionSettings()) -> tuple[tuple[str, ...], list[dict[str, float | str]]]:
    """Select only from the rolling training panel, never from the test month."""
    data = train.filter(pl.col(target).is_not_null())
    if data.is_empty():
        raise ValueError("no labelled observations available for feature selection")
    coverage = data.select([pl.col(name).is_not_null().mean().alias(name) for name in features]).row(0, named=True)
    viable = [name for name in features if float(coverage[name]) >= settings.min_coverage and data.get_column(name).drop_nulls().std() > 1e-12]
    if not viable:
        raise ValueError("no features meet coverage and variance requirements")
    # Candidate factor evaluation and downstream diagnostics use Rank IC.  Use
    # the same monotone statistic for every rolling training-window decision.
    daily = data.group_by("trade_date").agg([
        pl.corr(pl.col(name).rank(), pl.col(target).rank()).alias(name)
        for name in viable
    ])
    stats: list[dict[str, float | str]] = []
    for name in viable:
        values = daily.get_column(name).drop_nulls()
        mean, std = float(values.mean()), float(values.std())
        icir = mean / std * np.sqrt(252) if std > 1e-12 else 0.0
        stats.append({"feature": name, "coverage": float(coverage[name]), "mean_ic": mean, "icir": icir, "abs_icir": abs(icir)})
    if settings.use_all_features:
        return tuple(viable), stats
    ranked = sorted((row for row in stats if row["abs_icir"] >= settings.min_abs_icir), key=lambda row: float(row["abs_icir"]), reverse=True)
    if not ranked:
        ranked = sorted(stats, key=lambda row: float(row["abs_icir"]), reverse=True)
    recent_dates = data.get_column("trade_date").unique().sort().tail(settings.correlation_lookback_days)
    recent = data.filter(pl.col("trade_date").is_in(recent_dates)).select(viable).fill_nan(None).fill_null(0.0)
    matrix = recent.to_numpy().astype(float, copy=False)
    correlations = np.corrcoef(matrix, rowvar=False) if matrix.shape[0] > 1 else np.eye(len(viable))
    positions = {name: index for index, name in enumerate(viable)}
    selected: list[str] = []
    for row in ranked:
        name = str(row["feature"])
        if all(abs(correlations[positions[name], positions[other]]) < settings.max_abs_correlation for other in selected):
            selected.append(name)
        if len(selected) >= settings.max_features:
            break
    if not selected:
        selected.append(str(ranked[0]["feature"]))
    return tuple(selected), stats


def fit_predict_neural(
    kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    epochs: int = 20,
    max_samples: int = 60_000,
    train_dates: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a two-horizon neural model with a time-ordered validation split.

    The MLP intentionally uses residual blocks: factor selection may change the
    input width at each refit, while the 256-wide residual trunk stays stable.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra
        raise RuntimeError("MLP/Transformer require the optional 'deep-learning' dependency") from exc
    if kind not in {"mlp", "transformer"}:
        raise ValueError(f"unsupported neural model: {kind}")
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    if train_dates is not None:
        dates = np.asarray(train_dates).astype("datetime64[D]")
        validation_dates = np.unique(dates)[-63:]
        validation_mask = np.isin(dates, validation_dates)
    else:
        validation_mask = np.zeros(len(train_x), dtype=bool)
        validation_mask[-max(1, len(train_x) // 10):] = True
    fit_mask = ~validation_mask
    if not fit_mask.any() or not validation_mask.any():
        raise ValueError("neural model needs both training and validation observations")
    fit_indices = np.flatnonzero(fit_mask)
    if max_samples > 0 and len(fit_indices) > max_samples:
        fit_indices = rng.choice(fit_indices, size=max_samples, replace=False)
    x_fit = np.nan_to_num(train_x[fit_indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    y_fit = train_y[fit_indices].astype(np.float32, copy=False)
    x_val = np.nan_to_num(train_x[validation_mask], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    y_val = train_y[validation_mask].astype(np.float32, copy=False)
    target_mean = y_fit.mean(axis=0, keepdims=True)
    target_std = y_fit.std(axis=0, keepdims=True)
    target_std[target_std < 1e-7] = 1.0
    y_fit = (y_fit - target_mean) / target_std
    y_val = (y_val - target_mean) / target_std
    width = train_x.shape[1]
    if kind == "mlp":
        class ResidualBlock(nn.Module):
            def __init__(self, dimension: int = 256) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(dimension), nn.Linear(dimension, dimension * 2),
                    nn.GELU(), nn.Dropout(.10), nn.Linear(dimension * 2, dimension),
                )

            def forward(self, value: Any) -> Any:
                return value + self.net(value)

        class ResidualMlp(nn.Module):
            def __init__(self, count: int) -> None:
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(count, 256), nn.LayerNorm(256), nn.GELU(),
                    ResidualBlock(), ResidualBlock(), ResidualBlock(), nn.LayerNorm(256),
                )
                self.head_h1 = nn.Linear(256, 1)
                self.head_h5 = nn.Linear(256, 1)

            def forward(self, value: Any) -> Any:
                encoded = self.trunk(value)
                return torch.cat((self.head_h1(encoded), self.head_h5(encoded)), dim=1)
        model = ResidualMlp(width)
    else:
        class FeatureTransformer(nn.Module):
            def __init__(self, count: int) -> None:
                super().__init__()
                self.value = nn.Linear(1, 32)
                self.position = nn.Parameter(torch.zeros(1, count, 32))
                layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=.10, batch_first=True, norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 2))
            def forward(self, value: object) -> object:
                token = self.value(value.unsqueeze(-1)) + self.position
                return self.head(self.encoder(token).mean(dim=1))
        model = FeatureTransformer(width)
    model.to(device)
    batch_size = 8192 if device.type in {"mps", "cuda"} else 2048
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    loss_fn = nn.MSELoss()
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale = 0
    loss_history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(epochs):
        train_loss_total = 0.0
        train_observations = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss_total += float(loss.item()) * len(xb)
            train_observations += len(xb)
        model.eval()
        with torch.no_grad():
            value = loss_fn(model(torch.from_numpy(x_val).to(device)), torch.from_numpy(y_val).to(device)).item()
        loss_history.append({"epoch": epoch + 1, "train_mse": train_loss_total / train_observations, "validation_mse": float(value)})
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
        model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)).to(device)).cpu().numpy()
    return prediction * target_std + target_mean, {
        "backend": device.type, "fit_samples": int(len(x_fit)), "validation_samples": int(len(x_val)),
        "loss": "mse", "best_validation_mse": float(best_loss), "epochs_completed": epoch + 1,
        "loss_history": loss_history,
    }
