"""Leakage-safe rolling feature selection and compact neural regressors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SelectionSettings:
    min_coverage: float = 0.85
    min_abs_icir: float = 0.50
    max_features: int = 40
    max_abs_correlation: float = 0.90
    correlation_lookback_days: int = 60


def select_features(train: pl.DataFrame, features: tuple[str, ...], target: str, settings: SelectionSettings = SelectionSettings()) -> tuple[tuple[str, ...], list[dict[str, float | str]]]:
    """Select only from the rolling training panel, never from the test month."""
    data = train.filter(pl.col(target).is_not_null())
    if data.is_empty():
        raise ValueError("no labelled observations available for feature selection")
    coverage = data.select([pl.col(name).is_not_null().mean().alias(name) for name in features]).row(0, named=True)
    viable = [name for name in features if float(coverage[name]) >= settings.min_coverage and data.get_column(name).drop_nulls().std() > 1e-12]
    if not viable:
        raise ValueError("no features meet coverage and variance requirements")
    # Features are already cross-sectionally standardized.  The daily Pearson IC
    # is a fast monotone-quality proxy used only on the current training window.
    daily = data.group_by("trade_date").agg([pl.corr(name, target).alias(name) for name in viable])
    stats: list[dict[str, float | str]] = []
    for name in viable:
        values = daily.get_column(name).drop_nulls()
        mean, std = float(values.mean()), float(values.std())
        icir = mean / std * np.sqrt(252) if std > 1e-12 else 0.0
        stats.append({"feature": name, "coverage": float(coverage[name]), "mean_ic": mean, "icir": icir, "abs_icir": abs(icir)})
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


def fit_predict_neural(kind: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int, epochs: int = 8, max_samples: int = 60_000) -> np.ndarray:
    """Fit a small two-target MLP or feature-token Transformer on CPU/GPU."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra
        raise RuntimeError("MLP/Transformer require the optional 'deep-learning' dependency") from exc
    if kind not in {"mlp", "transformer"}:
        raise ValueError(f"unsupported neural model: {kind}")
    rng = np.random.default_rng(seed)
    if len(train_x) > max_samples:
        index = rng.choice(len(train_x), size=max_samples, replace=False)
        train_x, train_y = train_x[index], train_y[index]
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = train_x.shape[1]
    if kind == "mlp":
        model = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(.10), nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 2))
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
    loader = DataLoader(TensorDataset(torch.tensor(np.nan_to_num(train_x), dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)), batch_size=2048, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=.02)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.nan_to_num(test_x), dtype=torch.float32, device=device)).cpu().numpy()
