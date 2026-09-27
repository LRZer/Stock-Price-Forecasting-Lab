"""Save reproducible metrics, predictions, checkpoints and plots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import PreparedData
from .models import NEURAL_MODELS
from .training import TrainResult, regression_metrics


def save_report(data: PreparedData, results: list[TrainResult], output_dir: Path,
                settings: dict) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    naive = next(result for result in results if result.name == "naive")
    naive_mae = regression_metrics(data.test.y_raw, naive.test_prediction)["mae"]

    rows = []
    predictions = pd.DataFrame({
        "Date": data.test.dates,
        "Actual": data.test.y_raw,
        "PreviousClose": data.test.last_close,
    })
    history_rows = []
    for result in results:
        validation = regression_metrics(data.validation.y_raw, result.validation_prediction)
        test = regression_metrics(data.test.y_raw, result.test_prediction)
        rows.append({
            "model": result.name,
            "validation_mae": validation["mae"],
            "validation_rmse": validation["rmse"],
            "test_mae": test["mae"],
            "test_rmse": test["rmse"],
            "test_mape_pct": test["mape_pct"],
            "test_mae_improvement_vs_naive_pct": 100 * (naive_mae - test["mae"]) / naive_mae,
            "epochs_run": len(result.history),
            "seconds": result.seconds,
            "device": result.device,
        })
        predictions[result.name] = result.test_prediction
        history_rows.extend({"model": result.name, **row} for row in result.history)
        if result.name in NEURAL_MODELS:
            state = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
            torch.save(state, output_dir / "models" / f"{result.name}.pt")
        elif result.model is not None:
            joblib.dump(result.model, output_dir / "models" / f"{result.name}.joblib")

    metrics = pd.DataFrame(rows).sort_values("validation_mae").reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(output_dir / "predictions.csv", index=False, float_format="%.6f")
    pd.DataFrame(history_rows, columns=("model", "epoch", "train_mse_scaled",
                                        "validation_mae", "validation_rmse")).to_csv(
        output_dir / "training_history.csv", index=False, float_format="%.6f"
    )

    selected = str(metrics.iloc[0]["model"])
    summary = {
        "task": "next observed trading day close from the preceding history window",
        "selected_by_validation_mae": selected,
        "data_file": str(data.source),
        "data_sha256": hashlib.sha256(data.source.read_bytes()).hexdigest(),
        "rows": data.rows,
        "lookback": data.lookback,
        "splits": {
            "train": {"targets": len(data.train), "dates": data.train.date_range},
            "validation": {"targets": len(data.validation), "dates": data.validation.date_range},
            "test": {"targets": len(data.test), "dates": data.test.date_range},
        },
        "scaler": {"mean_from_train_only": data.scaler_mean,
                   "std_from_train_only": data.scaler_scale},
        "settings": settings,
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data.test.dates, data.test.y_raw, label="Actual", color="black", linewidth=1.5)
    ax.plot(data.test.dates, predictions[selected], label=f"Selected: {selected}", linewidth=1.3)
    if selected != "naive":
        ax.plot(data.test.dates, naive.test_prediction, label="Previous close baseline",
                color="gray", alpha=0.8, linewidth=1)
    ax.set_title("Held-out test: next-day close")
    ax.set_ylabel("Close price")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    fig.tight_layout()
    fig.savefig(output_dir / "forecast.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, max(3, len(metrics) * 0.38)))
    ordered = metrics.sort_values("test_mae", ascending=False)
    ax.barh(ordered["model"], ordered["test_mae"])
    ax.set_xlabel("Test MAE (price units)")
    ax.set_title("Same test period for every model; selection used validation only")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)

    if history_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, group in pd.DataFrame(history_rows).groupby("model"):
            ax.plot(group["epoch"], group["validation_mae"], label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training.png", dpi=150)
        plt.close(fig)

    return metrics
