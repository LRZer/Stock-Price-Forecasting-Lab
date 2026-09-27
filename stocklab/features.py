"""Point-in-time OHLCV features for next-trading-day return experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CLOSE_FEATURES = ("return_1d", "return_5d", "return_20d", "volatility_5d")
OHLCV_FEATURES = CLOSE_FEATURES + (
    "overnight_gap", "intraday_return", "daily_range", "close_location", "log_volume_vs_20d"
)


@dataclass(frozen=True)
class FeatureSamples:
    dates: np.ndarray
    previous_close: np.ndarray
    actual_close: np.ndarray
    actual_return: np.ndarray
    close_features: np.ndarray
    ohlcv_features: np.ndarray
    sources: tuple[Path, ...]

    def __len__(self) -> int:
        return len(self.dates)

    def features_for(self, name: str) -> np.ndarray:
        return self.close_features if name.endswith("-close") else self.ohlcv_features


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_samples(*paths: Path) -> FeatureSamples:
    """Features at the close of day t; target is the close of trading day t+1."""
    if not paths:
        raise ValueError("At least one OHLCV CSV is required")
    sources = tuple(Path(path).resolve() for path in paths)
    frame = pd.concat((pd.read_csv(path) for path in sources), ignore_index=True)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(required - set(frame.columns))}")
    if len(frame) < 200:
        raise ValueError("Need at least 200 daily rows")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].isna().any() or not frame["Date"].is_monotonic_increasing:
        raise ValueError("Dates must be increasing")
    if frame["Date"].duplicated().any():
        raise ValueError("Dates must be unique")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    prices = frame[["Open", "High", "Low", "Close"]]
    if (prices <= 0).any().any() or (frame["Volume"] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((frame["High"] < prices[["Open", "Close"]].max(axis=1)) |
            (frame["Low"] > prices[["Open", "Close"]].min(axis=1)) |
            (frame["High"] < frame["Low"])).any():
        raise ValueError("OHLC price bounds are inconsistent")

    close = frame["Close"]
    prior_close = close.shift(1)
    return_1d = close.pct_change(fill_method=None)
    day_span = frame["High"] - frame["Low"]
    log_volume = np.log1p(frame["Volume"])
    features = pd.DataFrame({
        "return_1d": return_1d,
        "return_5d": close / close.shift(5) - 1,
        "return_20d": close / close.shift(20) - 1,
        "volatility_5d": return_1d.rolling(5).std(),
        "overnight_gap": frame["Open"] / prior_close - 1,
        "intraday_return": close / frame["Open"] - 1,
        "daily_range": day_span / prior_close,
        "close_location": ((close - frame["Low"]) / day_span.replace(0, np.nan)).fillna(0.5) - 0.5,
        "log_volume_vs_20d": log_volume - log_volume.rolling(20).mean(),
    })
    # The earliest usable feature row has 20 completed prior closes. The last
    # row has no following observed close and is never used as a training label.
    feature_rows = features.iloc[20:-1]
    if feature_rows.isna().any().any() or not np.isfinite(feature_rows.to_numpy()).all():
        raise ValueError("Engineered features contain missing or non-finite values")
    previous = close.iloc[20:-1].to_numpy(dtype=np.float64)
    actual = close.iloc[21:].to_numpy(dtype=np.float64)
    return FeatureSamples(
        dates=frame["Date"].iloc[21:].dt.strftime("%Y-%m-%d").to_numpy(),
        previous_close=previous,
        actual_close=actual,
        actual_return=actual / previous - 1,
        close_features=feature_rows.loc[:, CLOSE_FEATURES].to_numpy(dtype=np.float64),
        ohlcv_features=feature_rows.loc[:, OHLCV_FEATURES].to_numpy(dtype=np.float64),
        sources=sources,
    )


def rolling_splits(samples: FeatureSamples, *, folds: int = 3,
                   validation_size: int = 100, test_size: int = 100) -> list[TimeSplit]:
    if min(folds, validation_size, test_size) < 1:
        raise ValueError("Fold counts and sizes must be positive")
    first_test = len(samples) - folds * test_size
    if first_test - validation_size < 100:
        raise ValueError("Insufficient training history for rolling evaluation")
    splits = []
    for fold in range(folds):
        test_start = first_test + fold * test_size
        train_end = test_start - validation_size
        splits.append(TimeSplit(
            train=np.arange(train_end),
            validation=np.arange(train_end, test_start),
            test=np.arange(test_start, test_start + test_size),
        ))
    return splits


def holdout_split(samples: FeatureSamples, *, first_holdout_date: str = "2026-01-01",
                  validation_size: int = 150) -> TimeSplit:
    test_start = int(np.searchsorted(samples.dates, first_holdout_date))
    train_end = test_start - validation_size
    if train_end < 100 or len(samples) - test_start < 50:
        raise ValueError("Need at least 100 training and 50 holdout targets")
    return TimeSplit(
        train=np.arange(train_end),
        validation=np.arange(train_end, test_start),
        test=np.arange(test_start, len(samples)),
    )
