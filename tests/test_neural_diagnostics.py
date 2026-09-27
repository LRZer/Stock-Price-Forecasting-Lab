"""Diagnostics must reflect the actual daily binary decisions."""

from __future__ import annotations

import unittest

import numpy as np

from stocklab.neural_diagnostics import calibration_rows, confusion_row


class DiagnosticsTests(unittest.TestCase):
    def test_confusion_counts_and_probability_bins(self) -> None:
        actual = np.array([0, 0, 1, 1, 1], dtype=np.int8)
        probability = np.array([0.2, 0.7, 0.4, 0.8, 1.0])
        row = confusion_row(actual, probability)
        self.assertEqual((row["true_non_up"], row["false_up"],
                          row["missed_up"], row["true_up"]), (1, 1, 1, 2))
        self.assertEqual(row["predicted_up_days"], 3)
        bins = calibration_rows(actual, probability)
        self.assertEqual(sum(item["days"] for item in bins), len(actual))
        self.assertEqual(bins[9]["days"], 1)  # p=1 belongs to the last bin.
        self.assertEqual(bins[9]["actual_up_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
