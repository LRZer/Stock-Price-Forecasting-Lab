"""Descriptive paired uncertainty for dated daily outcomes."""

from __future__ import annotations

import numpy as np


def moving_block_interval(differences: np.ndarray, *, block_size: int = 5,
                          repetitions: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Resample adjacent-day blocks; this is descriptive, not a market guarantee."""
    values = np.asarray(differences, dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all() or
            not 1 <= block_size <= len(values) or repetitions < 100):
        raise ValueError("Expected finite daily differences and valid resampling settings")
    rng = np.random.default_rng(seed)
    n = len(values)
    blocks_needed = (n + block_size - 1) // block_size
    starts = rng.integers(0, n - block_size + 1, size=(repetitions, blocks_needed))
    indices = (starts[:, :, None] + np.arange(block_size)).reshape(repetitions, -1)[:, :n]
    low, high = np.quantile(values[indices].mean(axis=1), (0.025, 0.975))
    return float(low), float(high)


def paired_accuracy_difference(actual_up: np.ndarray, selected_probability: np.ndarray,
                               baseline_probability: np.ndarray) -> dict[str, float | int]:
    """Selected minus baseline accuracy, with a five-day block interval."""
    actual = np.asarray(actual_up, dtype=np.int8)
    selected = np.asarray(selected_probability, dtype=np.float64)
    baseline = np.asarray(baseline_probability, dtype=np.float64)
    if (actual.ndim != 1 or len(actual) == 0 or len(selected) != len(actual) or
            len(baseline) != len(actual) or not np.isin(actual, [0, 1]).all() or
            not np.isfinite(selected).all() or not np.isfinite(baseline).all() or
            (selected < 0).any() or (selected > 1).any() or
            (baseline < 0).any() or (baseline > 1).any()):
        raise ValueError("Expected matching binary labels and valid up probabilities")
    selected_correct = (selected >= 0.5) == actual
    baseline_correct = (baseline >= 0.5) == actual
    differences = selected_correct.astype(float) - baseline_correct.astype(float)
    low, high = moving_block_interval(differences)
    return {"test_days": len(actual),
            "selected_correct_days": int(selected_correct.sum()),
            "baseline_correct_days": int(baseline_correct.sum()),
            "accuracy_gain": float(differences.mean()),
            "block_bootstrap_95pct_low": low,
            "block_bootstrap_95pct_high": high}
