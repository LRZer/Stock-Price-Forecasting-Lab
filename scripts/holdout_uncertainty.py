"""Paired moving-block intervals for the already saved 2026 holdout forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def block_interval(differences: np.ndarray, *, block_size: int = 5,
                   repetitions: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Resample adjacent days; report a descriptive interval, not a trading guarantee."""
    if not 1 <= block_size <= len(differences):
        raise ValueError("Block size must fit the test period")
    rng = np.random.default_rng(seed)
    n = len(differences)
    block_count = (n + block_size - 1) // block_size
    starts = rng.integers(0, n - block_size + 1, size=(repetitions, block_count))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets).reshape(repetitions, -1)[:, :n]
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path,
                        default=ROOT / "runs" / "feature-study" / "holdout-2026")
    args = parser.parse_args()
    rows = []
    for ticker in ("GOOG", "AAPL", "TSLA"):
        folder = args.holdout_dir / ticker.lower()
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        selected = summary["selected_by_validation"]
        predictions = pd.read_csv(folder / "predictions.csv")
        naive_abs = np.abs(predictions["Actual"].to_numpy() - predictions["naive"].to_numpy())
        selected_abs = np.abs(predictions["Actual"].to_numpy() - predictions[selected].to_numpy())
        gain = naive_abs - selected_abs
        low, high = block_interval(gain)
        best_day = int(np.argmax(gain))
        actual_return_pct = predictions["ActualReturnPct"].to_numpy()
        predicted_return_pct = 100 * (
            predictions[selected].to_numpy() / predictions["PreviousClose"].to_numpy() - 1)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(predictions["Date"], actual_return_pct, color="black",
                     linewidth=0.9, label="Actual next-day return")
        axes[0].plot(predictions["Date"], predicted_return_pct, color="tab:orange",
                     linewidth=1, label=f"Selected: {selected}")
        axes[0].axhline(0, color="gray", linewidth=0.7)
        axes[0].set_ylabel("Return (%)")
        axes[0].legend()
        axes[1].plot(predictions["Date"], gain.cumsum(), color="tab:blue")
        axes[1].axhline(0, color="gray", linewidth=0.7)
        axes[1].set_ylabel("Cumulative MAE gain")
        axes[1].set_title("Positive means fewer absolute price errors than previous close")
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(8))
        axes[1].tick_params(axis="x", rotation=45)
        fig.suptitle(f"{ticker} 2026 holdout diagnostic")
        fig.tight_layout()
        fig.savefig(folder / "diagnostics.png", dpi=150)
        plt.close(fig)
        rows.append({"ticker": ticker, "selected_model": selected,
                     "test_days": len(gain), "mean_daily_mae_gain": float(gain.mean()),
                     "block_bootstrap_95pct_low": low, "block_bootstrap_95pct_high": high,
                     "largest_gain_date": str(predictions["Date"].iloc[best_day]),
                     "largest_single_day_gain": float(gain[best_day]),
                     "mean_gain_without_best_day": float((gain.sum() - gain[best_day]) / (len(gain) - 1))})
    report = pd.DataFrame(rows)
    report.to_csv(args.holdout_dir / "uncertainty.csv", index=False, float_format="%.6f")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
