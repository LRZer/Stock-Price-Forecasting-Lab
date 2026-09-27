"""Checks for chronology and preprocessing leakage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab.data import prepare_data, prepare_rolling_folds
from stocklab.training import fit_classical, regression_metrics


class DataPreparationTests(unittest.TestCase):
    def make_csv(self, directory: Path, *, shock_test: bool = False) -> Path:
        close = np.linspace(100, 140, 250) + np.sin(np.arange(250) / 5)
        if shock_test:
            close[230:] += 1000
        path = directory / "prices.csv"
        pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=250),
                      "Close": close}).to_csv(path, index=False)
        return path

    def test_split_is_chronological_and_windows_exclude_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = prepare_data(self.make_csv(Path(temporary)), lookback=20)
        self.assertLess(data.train.date_range[1], data.validation.date_range[0])
        self.assertLess(data.validation.date_range[1], data.test.date_range[0])
        np.testing.assert_allclose(data.test.x_raw[:, -1], data.test.last_close)
        self.assertTrue((data.train.dates < data.validation.dates[0]).all())

    def test_test_prices_do_not_change_training_scaler_or_training_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_csv(Path(temporary))
            original = prepare_data(path, lookback=20)
            self.make_csv(Path(temporary), shock_test=True)
            changed = prepare_data(path, lookback=20)
        self.assertEqual(original.scaler_mean, changed.scaler_mean)
        self.assertEqual(original.scaler_scale, changed.scaler_scale)
        np.testing.assert_array_equal(original.train.x_scaled, changed.train.x_scaled)
        np.testing.assert_array_equal(original.train.y_scaled, changed.train.y_scaled)

    def test_forward_stacking_produces_finite_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = prepare_data(self.make_csv(Path(temporary)), lookback=20)
        result = fit_classical("stacked", data, seed=42)
        self.assertEqual(len(result.test_prediction), len(data.test))
        self.assertTrue(np.isfinite(result.test_prediction).all())
        self.assertGreaterEqual(regression_metrics(data.test.y_raw, result.test_prediction)["mae"], 0)

    def test_rolling_test_periods_do_not_overlap_and_refit_scaler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_csv(Path(temporary))
            folds = prepare_rolling_folds(path, lookback=20, folds=3,
                                          validation_size=30, test_size=30)
            self.make_csv(Path(temporary), shock_test=True)
            changed = prepare_rolling_folds(path, lookback=20, folds=3,
                                            validation_size=30, test_size=30)
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(set(np.concatenate([fold.test.dates for fold in folds]))), 90)
        for original, modified in zip(folds, changed):
            self.assertLess(original.train.date_range[1], original.validation.date_range[0])
            self.assertLess(original.validation.date_range[1], original.test.date_range[0])
            self.assertEqual(original.scaler_mean, modified.scaler_mean)
            self.assertEqual(original.scaler_scale, modified.scaler_scale)


if __name__ == "__main__":
    unittest.main()
