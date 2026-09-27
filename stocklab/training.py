"""Shared training and evaluation for all model families."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Some Windows Conda installations cannot query physical core count; a small
# explicit cap avoids an unrelated joblib warning for these tiny experiments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 4, 8)))

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedData
from .models import CLASSICAL_MODELS, NEURAL_MODELS, build_model


@dataclass
class TrainResult:
    name: str
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    history: list[dict[str, float | int]]
    seconds: float
    model: Any
    device: str


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and match target shape")
    error = actual - prediction
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape_pct": float(100 * np.mean(np.abs(error / actual))),
    }


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=5.0))


def _tree(seed: int) -> Any:
    return HistGradientBoostingRegressor(max_iter=80, max_depth=3,
                                         learning_rate=0.05, random_state=seed)


class ExpandingStacker:
    """Fit a meta-model on forward-only out-of-fold base predictions."""

    def __init__(self, seed: int):
        self.bases = (_ridge(), _tree(seed))
        self.meta = Ridge(alpha=5.0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ExpandingStacker":
        oof = np.full((len(y), len(self.bases)), np.nan)
        for train_idx, holdout_idx in TimeSeriesSplit(n_splits=4).split(x):
            for column, base in enumerate(self.bases):
                fold_model = clone(base).fit(x[train_idx], y[train_idx])
                oof[holdout_idx, column] = fold_model.predict(x[holdout_idx])
        available = np.isfinite(oof).all(axis=1)
        if available.sum() < 20:
            raise ValueError("Not enough forward validation predictions for stacking")
        self.meta.fit(oof[available], y[available])
        self.fitted_bases = [clone(base).fit(x, y) for base in self.bases]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        base_predictions = np.column_stack([base.predict(x) for base in self.fitted_bases])
        return self.meta.predict(base_predictions)


def fit_classical(name: str, data: PreparedData, *, seed: int) -> TrainResult:
    if name not in CLASSICAL_MODELS:
        raise ValueError(name)
    start = time.perf_counter()
    if name == "naive":
        return TrainResult(name, data.validation.last_close.copy(), data.test.last_close.copy(),
                           [], time.perf_counter() - start, None, "cpu")
    if name == "ridge":
        model = _ridge()
    elif name == "hist-gbdt":
        model = _tree(seed)
    else:
        model = ExpandingStacker(seed)
    # Learn a change from the last observed close. This lets linear and tree
    # baselines handle price levels outside the range seen during training.
    model.fit(data.train.x_raw, data.train.y_raw - data.train.last_close)
    return TrainResult(
        name=name,
        validation_prediction=data.validation.last_close + np.asarray(model.predict(data.validation.x_raw)),
        test_prediction=data.test.last_close + np.asarray(model.predict(data.test.x_raw)),
        history=[],
        seconds=time.perf_counter() - start,
        model=model,
        device="cpu",
    )


def _torch_predict(model: nn.Module, x: np.ndarray, device: torch.device,
                   batch_size: int) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


def fit_neural(
    name: str,
    data: PreparedData,
    *,
    device: str,
    seed: int,
    hidden_size: int = 32,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 5,
) -> TrainResult:
    if name not in NEURAL_MODELS:
        raise ValueError(name)
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    model = build_model(name, lookback=data.lookback, hidden_size=hidden_size).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    training = TensorDataset(torch.from_numpy(data.train.x_scaled),
                             torch.from_numpy(data.train.y_scaled))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float | int]] = []
    best_state = None
    best_mae = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        squared_error_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            mse = torch.mean((prediction - batch_y).square())
            loss = mse + 0.001 * model.aux_loss
            loss.backward()
            optimizer.step()
            squared_error_sum += float(mse.detach().cpu()) * len(batch_y)

        validation_scaled = _torch_predict(model, data.validation.x_scaled,
                                           torch_device, batch_size)
        validation_raw = data.inverse(validation_scaled)
        validation_scores = regression_metrics(data.validation.y_raw, validation_raw)
        history.append({
            "epoch": epoch,
            "train_mse_scaled": squared_error_sum / len(data.train),
            "validation_mae": validation_scores["mae"],
            "validation_rmse": validation_scores["rmse"],
        })
        if validation_scores["mae"] < best_mae - 1e-6:
            best_mae = validation_scores["mae"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_raw = data.inverse(_torch_predict(model, data.validation.x_scaled,
                                                  torch_device, batch_size))
    test_raw = data.inverse(_torch_predict(model, data.test.x_scaled,
                                            torch_device, batch_size))
    return TrainResult(name, validation_raw, test_raw, history,
                       time.perf_counter() - start, model, str(torch_device))
