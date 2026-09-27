"""Next-trading-day up-probability study with simple train-only baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Tiny Windows experiments do not need joblib to inspect physical core counts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 4, 8)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (CLOSE_FEATURES, OHLCV_FEATURES, FeatureSamples, TimeSplit,
                       holdout_split, make_samples, rolling_splits)


ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("GOOG", "AAPL", "TSLA")
MODELS = ("train-majority", "logit-close", "logit-ohlcv", "gbdt-close", "gbdt-ohlcv")


def actual_up(samples: FeatureSamples, rows: np.ndarray) -> np.ndarray:
    # A rare unchanged close belongs to the non-up class.
    return (samples.actual_return[rows] > 0).astype(np.int8)


def fit_direction(name: str, samples: FeatureSamples, train: np.ndarray):
    y = actual_up(samples, train)
    if name == "train-majority":
        return float(y.mean())
    if len(np.unique(y)) != 2:
        raise ValueError("Classifier training period needs both directions")
    if name.startswith("logit-"):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    elif name.startswith("gbdt-"):
        model = HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=7, min_samples_leaf=20,
            learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )
    else:
        raise ValueError(name)
    model.fit(samples.features_for(name)[train], y)
    return model


def probability_up(name: str, model, samples: FeatureSamples,
                   rows: np.ndarray) -> np.ndarray:
    if name == "train-majority":
        return np.full(len(rows), model, dtype=np.float64)
    return np.asarray(model.predict_proba(samples.features_for(name)[rows])[:, 1],
                      dtype=np.float64)


def classification_scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float64)
    if (not np.isfinite(probability).all() or (probability < 0).any() or
            (probability > 1).any() or len(probability) != len(y)):
        raise ValueError("Expected one valid up probability per target day")
    predicted = probability >= 0.5
    correct = int((predicted == y).sum())
    up_mask = y == 1
    down_mask = ~up_mask
    up_recall = float(predicted[up_mask].mean()) if up_mask.any() else 0.0
    down_recall = float((~predicted[down_mask]).mean()) if down_mask.any() else 0.0
    return {
        "correct_days": correct,
        "accuracy": correct / len(y),
        "balanced_accuracy": (up_recall + down_recall) / 2 if up_mask.any() and down_mask.any()
        else float("nan"),
        "brier": float(np.square(probability - y).mean()),
        "predicted_up_rate": float(predicted.mean()),
        "actual_up_rate": float(y.mean()),
    }


def evaluate(samples: FeatureSamples, split: TimeSplit, *, refit_for_test: bool):
    y_validation = actual_up(samples, split.validation)
    validation_scores = {}
    trained = {}
    for name in MODELS:
        model = fit_direction(name, samples, split.train)
        trained[name] = model
        probability = probability_up(name, model, samples, split.validation)
        validation_scores[name] = classification_scores(y_validation, probability)
    # Primary selection: percentage of correctly classified validation days.
    # Brier resolves ties without inspecting the test period.
    selected = min(MODELS, key=lambda name: (-validation_scores[name]["accuracy"],
                                              validation_scores[name]["brier"]))

    y_test = actual_up(samples, split.test)
    results = []
    predictions = pd.DataFrame({"Date": samples.dates[split.test],
                                "ActualUp": y_test,
                                "ActualReturnPct": samples.actual_return[split.test] * 100})
    for name in MODELS:
        model = (fit_direction(name, samples, np.concatenate((split.train, split.validation)))
                 if refit_for_test else trained[name])
        probability = probability_up(name, model, samples, split.test)
        test = classification_scores(y_test, probability)
        predictions[f"{name}_p_up"] = probability
        predictions[f"{name}_pred_up"] = (probability >= 0.5).astype(np.int8)
        results.append({"model": name,
                        "validation_accuracy": validation_scores[name]["accuracy"],
                        "validation_brier": validation_scores[name]["brier"],
                        "test_correct_days": test["correct_days"],
                        "test_days": len(y_test),
                        "test_accuracy": test["accuracy"],
                        "test_balanced_accuracy": test["balanced_accuracy"],
                        "test_brier": test["brier"],
                        "test_predicted_up_rate": test["predicted_up_rate"],
                        "test_actual_up_rate": test["actual_up_rate"]})
    return selected, pd.DataFrame(results), predictions


def _metadata(samples: FeatureSamples, split: TimeSplit) -> dict:
    return {
        "sources": [{"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in samples.sources],
        "target": "P(next trading close > previous trading close)",
        "unchanged_close_rule": "non-up class",
        "decision_threshold": 0.5,
        "close_features": CLOSE_FEATURES,
        "ohlcv_features": OHLCV_FEATURES,
        "train": [str(samples.dates[split.train[0]]), str(samples.dates[split.train[-1]])],
        "validation": [str(samples.dates[split.validation[0]]),
                       str(samples.dates[split.validation[-1]])],
        "test": [str(samples.dates[split.test[0]]), str(samples.dates[split.test[-1]])],
        "selection": "highest validation accuracy; lowest validation Brier breaks ties",
    }


def run_rolling(ticker: str, samples: FeatureSamples, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    prediction_frames = []
    selected_rows = []
    split_notes = []
    for fold, split in enumerate(rolling_splits(samples), start=1):
        selected, metrics, predictions = evaluate(samples, split, refit_for_test=False)
        metrics.insert(0, "fold", fold)
        predictions.insert(0, "fold", fold)
        frames.append(metrics)
        prediction_frames.append(predictions)
        split_notes.append(_metadata(samples, split))
        selected_row = metrics.loc[metrics["model"] == selected].iloc[0]
        baseline_row = metrics.loc[metrics["model"] == "train-majority"].iloc[0]
        selected_rows.append({"fold": fold, "selected_by_validation": selected,
                              "selected_correct_days": selected_row["test_correct_days"],
                              "selected_test_accuracy": selected_row["test_accuracy"],
                              "baseline_test_accuracy": baseline_row["test_accuracy"]})
    metrics = pd.concat(frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selected_frame = pd.DataFrame(selected_rows)
    summary = metrics.groupby("model", sort=False).agg(
        total_correct_days=("test_correct_days", "sum"),
        total_test_days=("test_days", "sum"),
        mean_test_brier=("test_brier", "mean"),
        mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
    ).reset_index()
    summary["overall_test_accuracy"] = summary["total_correct_days"] / summary["total_test_days"]
    metrics.to_csv(destination / "fold_metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(destination / "predictions.csv", index=False, float_format="%.6f")
    selected_frame.to_csv(destination / "selected_by_fold.csv", index=False, float_format="%.6f")
    summary.to_csv(destination / "rolling_metrics.csv", index=False, float_format="%.6f")
    (destination / "summary.json").write_text(json.dumps({
        "ticker": ticker, "mode": "retrospective_2025_rolling",
        "folds": split_notes, "models": MODELS,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{ticker} rolling: selected strategy "
          f"{selected_frame['selected_correct_days'].sum()}/{len(predictions)} "
          f"= {selected_frame['selected_correct_days'].sum() / len(predictions):.1%}; "
          f"baseline {int(summary.loc[summary['model'] == 'train-majority', 'total_correct_days'].iloc[0])}"
          f"/{len(predictions)}", flush=True)
    return {"ticker": ticker, "selected_correct_days": int(selected_frame["selected_correct_days"].sum()),
            "test_days": len(predictions),
            "selected_accuracy": float(selected_frame["selected_correct_days"].sum() / len(predictions)),
            "baseline_accuracy": float(summary.loc[summary["model"] == "train-majority",
                                               "overall_test_accuracy"].iloc[0])}


def run_2026(ticker: str, samples: FeatureSamples, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    split = holdout_split(samples)
    selected, metrics, predictions = evaluate(samples, split, refit_for_test=True)
    metrics.to_csv(destination / "metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(destination / "predictions.csv", index=False, float_format="%.6f")
    note = _metadata(samples, split)
    note.update({"ticker": ticker, "mode": "retrospective_2026_evaluation",
                 "selected_by_validation": selected,
                 "important": "2026 labels were inspected in an earlier price study; this is not a fresh holdout"})
    (destination / "summary.json").write_text(json.dumps(note, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    chosen = metrics.loc[metrics["model"] == selected].iloc[0]
    baseline = metrics.loc[metrics["model"] == "train-majority"].iloc[0]
    print(f"{ticker} 2026 retrospective: selected {selected}, "
          f"{chosen['test_correct_days']}/{chosen['test_days']} = {chosen['test_accuracy']:.1%}; "
          f"baseline {baseline['test_accuracy']:.1%}", flush=True)
    return {"ticker": ticker, "selected_by_validation": selected,
            "selected_correct_days": int(chosen["test_correct_days"]),
            "test_days": int(chosen["test_days"]),
            "selected_accuracy": float(chosen["test_accuracy"]),
            "baseline_accuracy": float(baseline["test_accuracy"]),
            "selected_brier": float(chosen["test_brier"]),
            "baseline_brier": float(baseline["test_brier"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rolling", "retrospective-2026"), default="rolling")
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=list(TICKERS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "direction-study")
    args = parser.parse_args()
    destination = args.output_dir / args.mode
    rows = []
    for ticker in args.tickers:
        paths = [ROOT / "data" / "raw" / f"{ticker}.csv"]
        if args.mode == "retrospective-2026":
            paths.append(ROOT / "data" / "holdout-2026" / f"{ticker}.csv")
        samples = make_samples(*paths)
        run_dir = destination / ticker.lower()
        rows.append(run_rolling(ticker, samples, run_dir) if args.mode == "rolling"
                    else run_2026(ticker, samples, run_dir))
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "all_tickers.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
