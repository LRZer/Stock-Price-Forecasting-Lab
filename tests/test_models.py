"""All migrated model names produce a one-step forecast."""

from __future__ import annotations

import unittest

import torch

from stocklab.models import DIRECTION_NEURAL_MODELS, NEURAL_MODELS, build_model


class ModelShapeTests(unittest.TestCase):
    def test_every_model_has_finite_output_and_naive_initialization(self) -> None:
        torch.manual_seed(42)
        history = torch.randn(3, 20, 1)
        for name in NEURAL_MODELS:
            with self.subTest(model=name):
                model = build_model(name, lookback=20, hidden_size=32).eval()
                with torch.no_grad():
                    prediction = model(history)
                self.assertEqual(tuple(prediction.shape), (3,))
                self.assertTrue(torch.isfinite(prediction).all())
                torch.testing.assert_close(prediction, history[:, -1, 0])

    def test_every_model_can_output_binary_logits_from_ohlcv_features(self) -> None:
        torch.manual_seed(42)
        history = torch.randn(3, 20, 6)
        for name in DIRECTION_NEURAL_MODELS:
            with self.subTest(model=name):
                model = build_model(name, lookback=20, hidden_size=32,
                                    input_size=6, task="direction").eval()
                with torch.no_grad():
                    logits = model(history)
                self.assertEqual(tuple(logits.shape), (3,))
                self.assertTrue(torch.isfinite(logits).all())
                torch.testing.assert_close(logits, torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
