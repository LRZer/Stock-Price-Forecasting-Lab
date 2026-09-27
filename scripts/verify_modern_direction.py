"""Check the saved exploratory model scores against their daily probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab.direction_study import classification_scores
from stocklab.modern_direction_models import MODERN_DIRECTION_MODELS
from stocklab.neural_direction import ROOT, TICKERS


NAMES = {"train-majority", *MODERN_DIRECTION_MODELS}
RUNS = ROOT / "runs" / "modern-direction"


def check_folder(folder) -> None:
    metrics = pd.read_csv(folder / "metrics.csv")
    predictions = pd.read_csv(folder / "predictions.csv")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert set(metrics.model) == NAMES == set(summary["models"])
    assert len(metrics) == len(NAMES) and len(predictions) == metrics.test_days.iloc[0]
    assert predictions.Date.is_unique and predictions.Date.is_monotonic_increasing
    assert summary["splits"]["test"]["targets"] == len(predictions)
    assert summary["selected_by_validation"] == metrics.iloc[0].model
    for source in summary["sources"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    truth = predictions.ActualUp.to_numpy()
    for row in metrics.itertuples(index=False):
        probability = predictions[f"{row.model}_p_up"].to_numpy()
        predicted = predictions[f"{row.model}_pred_up"].to_numpy()
        assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
        assert np.array_equal(predicted, (probability >= 0.5).astype(int))
        score = classification_scores(truth, probability)
        assert int(row.test_correct_days) == score["correct_days"]
        assert abs(float(row.test_accuracy) - score["accuracy"]) < 2e-6
        assert abs(float(row.test_brier) - score["brier"]) < 2e-6
        chart = folder / "diagnostics" / "models" / f"{row.model}.png"
        assert chart.is_file() and chart.stat().st_size > 1000


def main() -> None:
    folders = []
    for ticker in TICKERS:
        fixed = RUNS / "fixed-2025" / ticker.lower()
        folds = sorted((RUNS / "rolling-2025" / ticker.lower()).glob("fold-*"))
        assert len(folds) == 3
        folders.extend([fixed, *folds])
        rollup = pd.read_csv(RUNS / "rolling-2025" / ticker.lower() /
                             "rolling_metrics.csv").set_index("model")
        assert set(rollup.index) == NAMES
        for name in NAMES:
            total = sum(int(pd.read_csv(folder / "metrics.csv").set_index("model")
                            .loc[name, "test_correct_days"]) for folder in folds)
            assert total == int(rollup.loc[name, "total_correct_days"])
            assert int(rollup.loc[name, "total_test_days"]) == 300
    for folder in folders:
        check_folder(folder)
    print(f"Verified {len(folders)} exploratory reports and "
          f"{len(folders) * len(NAMES)} per-model charts")


if __name__ == "__main__":
    main()
