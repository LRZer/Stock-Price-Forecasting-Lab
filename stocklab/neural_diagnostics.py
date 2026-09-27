"""Readable diagnostics for next-day up probabilities and hard decisions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def confusion_row(y: np.ndarray, probability: np.ndarray) -> dict:
    actual = np.asarray(y, dtype=np.int8)
    predicted = np.asarray(probability) >= 0.5
    if len(actual) != len(predicted) or not np.isin(actual, [0, 1]).all():
        raise ValueError("Expected binary labels and one probability per day")
    tn = int(((actual == 0) & ~predicted).sum())
    fp = int(((actual == 0) & predicted).sum())
    fn = int(((actual == 1) & ~predicted).sum())
    tp = int(((actual == 1) & predicted).sum())
    return {"true_non_up": tn, "false_up": fp,
            "missed_up": fn, "true_up": tp,
            "actual_up_days": fn + tp, "predicted_up_days": fp + tp,
            "up_recall": tp / (fn + tp) if fn + tp else float("nan"),
            "non_up_recall": tn / (tn + fp) if tn + fp else float("nan")}


def calibration_rows(y: np.ndarray, probability: np.ndarray,
                     *, bins: int = 10) -> list[dict]:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if (len(actual) != len(p) or bins < 2 or not np.isfinite(p).all() or
            (p < 0).any() or (p > 1).any() or not np.isin(actual, [0, 1]).all()):
        raise ValueError("Expected valid binary labels and probabilities")
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for number in range(bins):
        mask = index == number
        rows.append({"bin": number, "probability_from": number / bins,
                     "probability_to": (number + 1) / bins,
                     "days": int(mask.sum()),
                     "mean_predicted_probability": float(p[mask].mean()) if mask.any() else np.nan,
                     "actual_up_rate": float(actual[mask].mean()) if mask.any() else np.nan})
    return rows


def plot_selected_diagnostics(predictions: pd.DataFrame, confusion: pd.DataFrame,
                              calibration: pd.DataFrame, selected: str,
                              destination: Path) -> None:
    actual = predictions["ActualUp"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes[0], ("train-majority", selected)):
        row = confusion.set_index("model").loc[name]
        matrix = np.array([[row.true_non_up, row.false_up],
                           [row.missed_up, row.true_up]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=14,
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set_xticks((0, 1), ("Predict non-up", "Predict up"))
        ax.set_yticks((0, 1), ("Actual non-up", "Actual up"))
        ax.set_title(name)

    ax = axes[1, 0]
    ax.plot((0, 1), (0, 1), color="black", linestyle="--", label="Ideal calibration")
    for name, color in (("train-majority", "#888888"), (selected, "#ed8c2b")):
        rows = calibration.loc[(calibration.model == name) & (calibration.days > 0)]
        ax.plot(rows.mean_predicted_probability, rows.actual_up_rate, "o-",
                color=color, label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(up)",
           ylabel="Observed up frequency", title="Calibration by probability bin")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(predictions[f"{selected}_p_up"], bins=np.linspace(0, 1, 21),
            color="#ed8c2b", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set(xlim=(0, 1), xlabel="Predicted P(up)", ylabel="Test days",
           title=f"{selected} probability distribution; actual up {actual.mean():.1%}")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_model_diagnostics(predictions: pd.DataFrame, confusion: pd.DataFrame,
                           calibration: pd.DataFrame, name: str,
                           destination: Path) -> None:
    """One readable confusion/rate/calibration/probability page per candidate."""
    row = confusion.set_index("model").loc[name]
    bins = calibration.loc[(calibration.model == name) & (calibration.days > 0)]
    probability = predictions[f"{name}_p_up"].to_numpy(dtype=np.float64)
    total = len(predictions)
    if len(probability) != int(row.actual_up_days + row.true_non_up + row.false_up):
        raise ValueError(f"Confusion counts do not match predictions for {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    fig.suptitle(f"{name}: {int(row.true_non_up + row.true_up)}/{total} correct", fontsize=12)

    ax = axes[0, 0]
    matrix = np.array([[row.true_non_up, row.false_up],
                       [row.missed_up, row.true_up]], dtype=int)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=12,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks((0, 1), ("Predict non-up", "Predict up"))
    ax.set_yticks((0, 1), ("Actual non-up", "Actual up"))
    ax.set_title("Confusion matrix")

    ax = axes[0, 1]
    actual_rate = row.actual_up_days / total
    predicted_rate = row.predicted_up_days / total
    bars = ax.bar(("Actual up", "Predicted up"), (actual_rate, predicted_rate),
                  color=("#777777", "#3377bb"))
    for bar, rate in zip(bars, (actual_rate, predicted_rate)):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.1%}",
                ha="center", va="bottom")
    ax.set(ylim=(0, 1.1), ylabel="Share of test days", title="How often does it guess up?")

    ax = axes[1, 0]
    ax.plot((0, 1), (0, 1), "k--", linewidth=1, label="Ideal")
    ax.plot(bins.mean_predicted_probability, bins.actual_up_rate,
            "o-", color="#ed8c2b", label=name)
    for point in bins.itertuples():
        ax.annotate(f"n={point.days}",
                    (point.mean_predicted_probability, point.actual_up_rate),
                    xytext=(3, 5), textcoords="offset points", fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(up)",
           ylabel="Observed up frequency", title="Calibration by 10 bins")

    ax = axes[1, 1]
    ax.hist(probability, bins=np.linspace(0, 1, 21), color="#3377bb", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), xlabel="Predicted P(up)", ylabel="Test days",
           title="Probability distribution")
    fig.tight_layout()
    fig.savefig(destination, dpi=110)
    plt.close(fig)


def render_all_model_diagnostics(predictions: pd.DataFrame, confusion: pd.DataFrame,
                                 calibration: pd.DataFrame, destination_dir: Path) -> int:
    names = tuple(confusion.model)
    if set(names) != set(calibration.model):
        raise ValueError("Confusion and calibration candidates differ")
    for name in names:
        plot_model_diagnostics(predictions, confusion, calibration, name,
                               destination_dir / f"{name}.png")
    return len(names)
