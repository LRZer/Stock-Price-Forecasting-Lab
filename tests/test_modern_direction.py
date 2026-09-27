"""New direction models accept the existing feature windows and train end to end."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from stocklab.modern_direction import fit_modern_classifier
from stocklab.modern_direction_models import (MODERN_DIRECTION_MODELS,
                                               build_modern_direction_model)
from stocklab.neural_direction_data import DirectionFold, ScaledSequences


class ModernDirectionTests(unittest.TestCase):
    def test_shapes_and_gradients_on_short_and_full_windows(self) -> None:
        torch.manual_seed(7)
        for lookback, channels in ((5, 1), (20, 6)):
            x = torch.randn(4, lookback, channels)
            for name in MODERN_DIRECTION_MODELS:
                with self.subTest(model=name, lookback=lookback, channels=channels):
                    model = build_modern_direction_model(
                        name, lookback=lookback, input_size=channels)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (4,))
                    self.assertTrue(torch.isfinite(logits).all())
                    logits.sum().backward()
                    self.assertTrue(any(parameter.grad is not None
                                        for parameter in model.parameters()))
                    if name == "patchtst-direction" and lookback == 5:
                        self.assertEqual(model.right_pad, 1)
                        self.assertEqual(model.position.shape[1], 2)

    def test_training_returns_probabilities_for_each_model(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(90, 20, 6)).astype(np.float32)
        y = (x[:, -1, 0] > 0).astype(np.float32)
        fold = DirectionFold(np.arange(60), np.arange(60, 75), np.arange(75, 90))
        data = ScaledSequences(x, y, np.arange(90), fold, np.zeros(6), np.ones(6))
        for name in MODERN_DIRECTION_MODELS:
            with self.subTest(model=name):
                result = fit_modern_classifier(name, data, device="cpu", epochs=2)
                self.assertEqual(result.test_probability.shape, (15,))
                self.assertTrue(np.isfinite(result.test_probability).all())
                self.assertTrue(((result.test_probability >= 0) &
                                 (result.test_probability <= 1)).all())

    def test_timesnet_prediction_is_independent_of_batch_neighbors(self) -> None:
        model = build_modern_direction_model("timesnet-direction", lookback=20,
                                            input_size=6).eval()
        # The final head starts at zero; make its input observable for this test.
        with torch.no_grad():
            model.head[-1].weight.fill_(0.1)
        x = torch.randn(3, 20, 6)
        with torch.no_grad():
            single = model(x[:1])
            grouped = model(x)[:1]
        torch.testing.assert_close(single, grouped)


if __name__ == "__main__":
    unittest.main()
