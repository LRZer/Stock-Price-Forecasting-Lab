"""Chronological one-step forecasting data preparation without test leakage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    x_raw: np.ndarray
    x_scaled: np.ndarray
    y_raw: np.ndarray
    y_scaled: np.ndarray
    last_close: np.ndarray
    dates: np.ndarray

    def __len__(self) -> int:
        return len(self.y_raw)

    @property
    def date_range(self) -> tuple[str, str]:
        return str(self.dates[0]), str(self.dates[-1])


@dataclass(frozen=True)
class PreparedData:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    scaler_mean: float
    scaler_scale: float
    source: Path
    rows: int
    lookback: int

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scaler_scale + self.scaler_mean


def load_close(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if not {"Date", "Close"}.issubset(frame.columns):
        raise ValueError(f"{path}: expected Date and Close columns")
    dates = pd.to_datetime(frame["Date"], errors="raise")
    close = pd.to_numeric(frame["Close"], errors="raise").to_numpy(dtype=np.float64)
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"{path}: dates must be unique and increasing")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"{path}: close prices must be finite and positive")
    return dates.dt.strftime("%Y-%m-%d").to_numpy(), close


def _prepare_bounds(path: Path, dates: np.ndarray, close: np.ndarray,
                    lookback: int, train_end: int, validation_end: int,
                    test_end: int) -> PreparedData:
    if min(train_end, validation_end - train_end, test_end - validation_end) < 20:
        raise ValueError("each chronological split must contain at least 20 targets")
    training_prices = close[: lookback + train_end]
    scaler_mean = float(training_prices.mean())
    scaler_scale = float(training_prices.std()) or 1.0
    scaled_close = ((close - scaler_mean) / scaler_scale).astype(np.float32)

    x_raw = np.stack([close[i : i + lookback] for i in range(test_end)]).astype(np.float32)
    x_scaled = np.stack(
        [scaled_close[i : i + lookback] for i in range(test_end)]
    ).astype(np.float32)
    y_raw = close[lookback : lookback + test_end].astype(np.float64)
    y_scaled = scaled_close[lookback : lookback + test_end].astype(np.float32)
    last_close = close[lookback - 1 : lookback + test_end - 1].astype(np.float64)
    target_dates = dates[lookback : lookback + test_end]

    def split(start: int, end: int) -> DataSplit:
        return DataSplit(
            x_raw=x_raw[start:end],
            x_scaled=x_scaled[start:end, :, None],
            y_raw=y_raw[start:end],
            y_scaled=y_scaled[start:end],
            last_close=last_close[start:end],
            dates=target_dates[start:end],
        )

    return PreparedData(
        train=split(0, train_end),
        validation=split(train_end, validation_end),
        test=split(validation_end, test_end),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        source=path,
        rows=len(close),
        lookback=lookback,
    )


def prepare_data(
    path: str | Path,
    *,
    lookback: int = 20,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PreparedData:
    """Build history windows; each target is the next observed trading day.

    Validation and test windows may use prices observed before their target date.
    The scaler is fitted only through the last training target.
    """
    path = Path(path).resolve()
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be less than 1")

    dates, close = load_close(path)
    samples = len(close) - lookback
    if samples < 100:
        raise ValueError(f"{path}: need at least 100 windows; got {samples}")
    train_end = int(samples * train_fraction)
    validation_end = int(samples * (train_fraction + validation_fraction))
    return _prepare_bounds(path, dates, close, lookback, train_end, validation_end, samples)


def prepare_rolling_folds(
    path: str | Path,
    *,
    lookback: int = 20,
    folds: int = 3,
    validation_size: int = 100,
    test_size: int = 100,
) -> list[PreparedData]:
    """Expanding training windows with disjoint, forward test periods."""
    path = Path(path).resolve()
    if lookback < 2 or folds < 1 or validation_size < 20 or test_size < 20:
        raise ValueError("invalid rolling-window sizes")
    dates, close = load_close(path)
    total_samples = len(close) - lookback
    first_test_start = total_samples - folds * test_size
    if first_test_start - validation_size < 100:
        raise ValueError("not enough history for these rolling folds")
    prepared = []
    for fold in range(folds):
        test_start = first_test_start + fold * test_size
        train_end = test_start - validation_size
        prepared.append(_prepare_bounds(path, dates, close, lookback, train_end,
                                        test_start, test_start + test_size))
    return prepared
