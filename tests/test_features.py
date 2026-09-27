"""Checks that OHLCV features are available before their forecast target."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab.feature_study import fit_model, predict_return, score
from stocklab.features import holdout_split, make_samples, rolling_splits


class FeatureStudyTests(unittest.TestCase):
    def make_csv(self, directory: Path, *, future_shock: bool = False) -> Path:
        time = np.arange(450)
        close = 100 + time * 0.04 + 4 * np.sin(time / 8)
        if future_shock:
            close[400:] += 30
        open_price = close + 0.2 * np.cos(time / 3)
        path = directory / "prices.csv"
        pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=len(time)),
            "Open": open_price,
            "High": np.maximum(open_price, close) + 0.5,
            "Low": np.minimum(open_price, close) - 0.5,
            "Close": close,
            "Volume": 1_000_000 + time * 100,
        }).to_csv(path, index=False)
        return path

    def test_next_day_alignment_and_future_changes_do_not_rewrite_past_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_csv(Path(directory))
            original = make_samples(path)
            self.make_csv(Path(directory), future_shock=True)
            changed = make_samples(path)
        self.assertEqual(original.dates[0], "2024-01-30")
        self.assertEqual(len(original.close_features[0]), 4)
        self.assertEqual(len(original.ohlcv_features[0]), 9)
        # The first modified price is row 400. All targets up through row 399
        # must retain identical features and labels.
        np.testing.assert_array_equal(original.ohlcv_features[:379], changed.ohlcv_features[:379])
        np.testing.assert_array_equal(original.actual_return[:379], changed.actual_return[:379])
        np.testing.assert_array_equal(original.previous_close[:379], changed.previous_close[:379])

    def test_fit_and_rolling_splits_do_not_use_later_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_csv(Path(directory))
            original = make_samples(path)
            folds = rolling_splits(original, folds=2, validation_size=50, test_size=50)
            model = fit_model("ridge-ohlcv", original, folds[0].train)
            prediction = predict_return("ridge-ohlcv", model, original, folds[0].validation)
            self.make_csv(Path(directory), future_shock=True)
            changed = make_samples(path)
            changed_model = fit_model("ridge-ohlcv", changed, folds[0].train)
            changed_prediction = predict_return("ridge-ohlcv", changed_model, changed,
                                                folds[0].validation)
        np.testing.assert_allclose(prediction, changed_prediction, atol=0, rtol=0)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(folds[0].train[-1], folds[0].validation[0])
        self.assertLess(folds[0].validation[-1], folds[0].test[0])
        self.assertEqual(len(set(np.concatenate([fold.test for fold in folds]))), 100)

    def test_pipeline_learns_an_obvious_repeating_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = make_samples(self.make_csv(Path(directory)))
        split = rolling_splits(samples, folds=2, validation_size=50, test_size=50)[0]
        model = fit_model("ridge-close", samples, split.train)
        learned = predict_return("ridge-close", model, samples, split.test)
        naive = np.zeros(len(split.test))
        self.assertLess(score(samples, split.test, learned)["mae"],
                        0.9 * score(samples, split.test, naive)["mae"])

    def test_2026_holdout_boundary_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_csv(Path(directory))
            frame = pd.read_csv(path)
            frame["Date"] = pd.bdate_range("2025-01-01", periods=len(frame))
            frame.to_csv(path, index=False)
            samples = make_samples(path)
        split = holdout_split(samples, validation_size=100)
        self.assertTrue((samples.dates[split.train] < "2026-01-01").all())
        self.assertTrue((samples.dates[split.validation] < "2026-01-01").all())
        self.assertTrue((samples.dates[split.test] >= "2026-01-01").all())
        self.assertEqual(split.validation[-1] + 1, split.test[0])


if __name__ == "__main__":
    unittest.main()
