"""Daily OHLCV sequences with strictly forward next-day direction labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = ("close_log_return", "overnight_gap", "intraday_return",
                 "high_extension", "low_extension", "log_volume_change")


@dataclass(frozen=True)
class DirectionSeries:
    dates: np.ndarray
    close: np.ndarray
    daily_features: np.ndarray
    labels: np.ndarray
    lookback: int
    sources: tuple[Path, ...]

    @property
    def sample_dates(self) -> np.ndarray:
        return self.dates[self.lookback:]

    @property
    def sample_count(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class DirectionFold:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class ScaledSequences:
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    fold: DirectionFold
    mean: np.ndarray
    scale: np.ndarray


def load_direction_series(*paths: Path, lookback: int = 20) -> DirectionSeries:
    if not paths or lookback < 5:
        raise ValueError("Need OHLCV source files and lookback >= 5")
    sources = tuple(Path(path).resolve() for path in paths)
    frame = pd.concat((pd.read_csv(path) for path in sources), ignore_index=True)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(required - set(frame.columns))}")
    if len(frame) - lookback < 200:
        raise ValueError("Need at least 200 training targets")
    dates = pd.to_datetime(frame["Date"], errors="raise")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Dates must be unique and increasing")
    values = {}
    for column in ("Open", "High", "Low", "Close", "Volume"):
        values[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    op, high, low, close, volume = (values[name] for name in
                                    ("Open", "High", "Low", "Close", "Volume"))
    if (not all(np.isfinite(value).all() for value in values.values()) or
            min(op.min(), high.min(), low.min(), close.min()) <= 0 or volume.min() < 0):
        raise ValueError("OHLCV values must be finite and valid")
    if ((high < np.maximum(op, close)) | (low > np.minimum(op, close)) |
            (high < low)).any():
        raise ValueError("OHLC bounds are inconsistent")

    prior_close = np.r_[close[0], close[:-1]]
    prior_volume = np.r_[volume[0], volume[:-1]]
    daily = np.column_stack((
        np.log(close / prior_close),
        np.log(op / prior_close),
        np.log(close / op),
        np.log(high / np.maximum(op, close)),
        np.log(np.minimum(op, close) / low),
        np.log1p(volume) - np.log1p(prior_volume),
    ))
    daily[0, 0] = daily[0, 1] = daily[0, 5] = 0
    if not np.isfinite(daily).all():
        raise ValueError("Engineered features must be finite")
    labels = (close[lookback:] > close[lookback - 1:-1]).astype(np.float32)
    return DirectionSeries(dates.dt.strftime("%Y-%m-%d").to_numpy(), close, daily,
                           labels, lookback, sources)


def fixed_2025_fold(series: DirectionSeries) -> DirectionFold:
    n = series.sample_count
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    return DirectionFold(np.arange(train_end),
                         np.arange(train_end, validation_end),
                         np.arange(validation_end, n))


def rolling_folds(series: DirectionSeries, *, folds: int = 3,
                  validation_size: int = 100, test_size: int = 100) -> list[DirectionFold]:
    if min(folds, validation_size, test_size) < 1:
        raise ValueError("Fold counts and sizes must be positive")
    first_test = series.sample_count - folds * test_size
    if first_test - validation_size < 100:
        raise ValueError("Insufficient train history")
    result = []
    for number in range(folds):
        test_start = first_test + number * test_size
        train_end = test_start - validation_size
        result.append(DirectionFold(np.arange(train_end),
                                    np.arange(train_end, test_start),
                                    np.arange(test_start, test_start + test_size)))
    return result


def scale_for_fold(series: DirectionSeries, fold: DirectionFold) -> ScaledSequences:
    if not (len(fold.train) and len(fold.validation) and len(fold.test) and
            fold.train[-1] < fold.validation[0] and
            fold.validation[-1] < fold.test[0]):
        raise ValueError("Expected nonempty chronological train, validation, test")
    # The last training label is observed on row lookback + fold.train[-1].
    # No validation/test observations enter the mean or scale.
    training_daily = series.daily_features[:series.lookback + fold.train[-1] + 1]
    mean = training_daily.mean(axis=0)
    scale = training_daily.std(axis=0)
    scale[scale == 0] = 1
    normalized = (series.daily_features - mean) / scale
    end = fold.test[-1] + 1
    x = np.stack([normalized[i:i + series.lookback] for i in range(end)]).astype(np.float32)
    return ScaledSequences(x, series.labels[:end].copy(), series.sample_dates[:end].copy(),
                           fold, mean, scale)
