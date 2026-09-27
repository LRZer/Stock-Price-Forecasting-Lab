"""Time alignment and training-only normalization for neural classifiers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab.neural_direction import fit_classifier
from stocklab.ablation_study import variant
from stocklab.neural_direction_data import (DirectionFold, ScaledSequences,
                                            fixed_2025_fold,
                                            load_direction_series,
                                            rolling_folds, scale_for_fold)


class NeuralDirectionDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "prices.csv"
        n = 260
        close = 100 + np.cumsum(np.where(np.arange(n) % 3 == 0, 2., -0.5))
        self.frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1000 + np.arange(n) * 10,
        })
        self.frame.to_csv(self.path, index=False)

    def test_next_day_label_and_window_end_before_target(self) -> None:
        series = load_direction_series(self.path, lookback=20)
        fold = fixed_2025_fold(series)
        data = scale_for_fold(series, fold)
        self.assertEqual(len(series.labels), 240)
        self.assertEqual(series.labels[0], float(self.frame.Close.iloc[20] >
                                                  self.frame.Close.iloc[19]))
        self.assertEqual(data.dates[0], "2024-01-29")
        np.testing.assert_allclose(data.x[0, -1],
                                   (series.daily_features[19] - data.mean) / data.scale,
                                   rtol=1e-6)
        self.assertLess(fold.train[-1], fold.validation[0])
        self.assertLess(fold.validation[-1], fold.test[0])

    def test_future_prices_cannot_change_training_or_scaler(self) -> None:
        before = load_direction_series(self.path)
        fold = fixed_2025_fold(before)
        original = scale_for_fold(before, fold)
        future_row = before.lookback + fold.validation[0]
        changed = self.frame.copy()
        changed.loc[future_row:, ["Open", "High", "Low", "Close"]] *= 1.4
        changed.loc[future_row:, "Volume"] *= 4
        changed.to_csv(self.path, index=False)
        after = scale_for_fold(load_direction_series(self.path), fold)
        np.testing.assert_allclose(original.mean, after.mean)
        np.testing.assert_allclose(original.scale, after.scale)
        np.testing.assert_allclose(original.x[fold.train], after.x[fold.train])
        np.testing.assert_array_equal(original.y[fold.train], after.y[fold.train])

    def test_rolling_test_windows_are_disjoint(self) -> None:
        series = load_direction_series(self.path)
        folds = rolling_folds(series, folds=2, validation_size=30, test_size=40)
        self.assertEqual(len(set(folds[0].test) & set(folds[1].test)), 0)
        for fold in folds:
            self.assertLess(fold.train[-1], fold.validation[0])
            self.assertLess(fold.validation[-1], fold.test[0])

    def test_ablation_variants_keep_identical_target_dates(self) -> None:
        series = load_direction_series(self.path, lookback=20)
        base = scale_for_fold(series, fixed_2025_fold(series))
        short_close = variant(base, 5, (0,))
        self.assertEqual(short_close.x.shape[1:], (5, 1))
        np.testing.assert_array_equal(short_close.dates, base.dates)
        np.testing.assert_array_equal(short_close.y, base.y)
        np.testing.assert_allclose(short_close.x[:, :, 0], base.x[:, -5:, 0])


class NeuralDirectionLearningTests(unittest.TestCase):
    def test_gru_learns_an_obvious_direction_rule(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(260, 20, 6)).astype(np.float32)
        y = (x[:, -1, 0] > 0).astype(np.float32)
        fold = DirectionFold(np.arange(180), np.arange(180, 220),
                             np.arange(220, 260))
        data = ScaledSequences(x, y, np.arange(260), fold, np.zeros(6), np.ones(6))
        result = fit_classifier("gru", data, device="cpu", epochs=20)
        predicted = result.test_probability >= 0.5
        self.assertGreater(float((predicted == y[fold.test]).mean()), 0.70)


if __name__ == "__main__":
    unittest.main()
