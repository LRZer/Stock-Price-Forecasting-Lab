"""The direction interval compares the same dated predictions pairwise."""

from __future__ import annotations

import unittest

import numpy as np

from stocklab.uncertainty import moving_block_interval, paired_accuracy_difference


class PairedUncertaintyTests(unittest.TestCase):
    def test_exact_extremes_and_reproducibility(self) -> None:
        actual = np.tile([0, 1], 50).astype(np.int8)
        selected = np.where(actual == 1, 0.9, 0.1)
        baseline = 1 - selected
        result = paired_accuracy_difference(actual, selected, baseline)
        self.assertEqual(result["selected_correct_days"], 100)
        self.assertEqual(result["baseline_correct_days"], 0)
        self.assertEqual(result["accuracy_gain"], 1)
        self.assertEqual(result["block_bootstrap_95pct_low"], 1)
        self.assertEqual(result["block_bootstrap_95pct_high"], 1)
        np.testing.assert_equal(result, paired_accuracy_difference(actual, selected, baseline))

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            moving_block_interval(np.array([0.0, np.nan]))
        with self.assertRaises(ValueError):
            paired_accuracy_difference(np.array([0, 1]), np.array([0.6]), np.array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
