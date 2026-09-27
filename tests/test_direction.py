"""Accuracy and probability checks for the next-day direction task."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab.direction_study import (actual_up, classification_scores,
                                      fit_direction, probability_up)
from stocklab.features import make_samples, rolling_splits


class DirectionStudyTests(unittest.TestCase):
    def test_accuracy_and_brier_are_distinct(self):
        truth = np.array([1, 0, 1, 0])
        probability = np.array([0.9, 0.7, 0.8, 0.1])
        result = classification_scores(truth, probability)
        self.assertEqual(result["correct_days"], 3)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["brier"], 0.1375)

    def test_classifier_learns_an_obvious_periodic_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            time = np.arange(450)
            close = 100 + 0.04 * time + 4 * np.sin(time / 8)
            open_price = close + 0.1
            path = Path(directory) / "prices.csv"
            pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=len(time)),
                          "Open": open_price,
                          "High": np.maximum(open_price, close) + 0.5,
                          "Low": np.minimum(open_price, close) - 0.5,
                          "Close": close,
                          "Volume": 1_000_000 + time * 100}).to_csv(path, index=False)
            samples = make_samples(path)
        split = rolling_splits(samples, folds=2, validation_size=50, test_size=50)[0]
        truth = actual_up(samples, split.test)
        baseline = fit_direction("train-majority", samples, split.train)
        learned = fit_direction("logit-close", samples, split.train)
        baseline_scores = classification_scores(
            truth, probability_up("train-majority", baseline, samples, split.test))
        learned_probability = probability_up("logit-close", learned, samples, split.test)
        learned_scores = classification_scores(truth, learned_probability)
        self.assertTrue(((learned_probability >= 0) & (learned_probability <= 1)).all())
        self.assertGreater(learned_scores["accuracy"], baseline_scores["accuracy"] + 0.1)


if __name__ == "__main__":
    unittest.main()
