"""Small, dated GRU ablation: history length, feature set, random seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .direction_study import classification_scores
from .neural_direction import ROOT, TICKERS, fit_classifier
from .neural_direction_data import (FEATURE_NAMES, ScaledSequences,
                                    load_direction_series, rolling_folds,
                                    scale_for_fold)


SEEDS = (42, 123, 2026)
WINDOWS = (5, 20)
FEATURE_SETS = {"close-return": (0,), "ohlcv": tuple(range(len(FEATURE_NAMES)))}


def variant(base: ScaledSequences, lookback: int,
            channels: tuple[int, ...]) -> ScaledSequences:
    if lookback > base.x.shape[1] or not channels:
        raise ValueError("Invalid ablation window or channels")
    x = base.x[:, -lookback:, :][:, :, channels].copy()
    return ScaledSequences(x, base.y, base.dates, base.fold,
                           base.mean[list(channels)], base.scale[list(channels)])


def run(output_dir: Path, *, device: str = "auto", epochs: int = 20) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    predictions = []
    source_info = []
    for ticker in TICKERS:
        path = ROOT / "data" / "raw" / f"{ticker}.csv"
        series = load_direction_series(path, lookback=20)
        source_info.append({"ticker": ticker, "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for number, fold in enumerate(rolling_folds(series), 1):
            base = scale_for_fold(series, fold)
            truth = base.y[fold.test]
            majority = fit_classifier("train-majority", base)
            majority_score = classification_scores(truth, majority.test_probability)
            for feature_set, channels in FEATURE_SETS.items():
                for window in WINDOWS:
                    data = variant(base, window, channels)
                    for seed in SEEDS:
                        print(f"{ticker} fold {number}: {feature_set}, {window} days, seed {seed}",
                              flush=True)
                        trained = fit_classifier("gru", data, seed=seed,
                                                 device=device, epochs=epochs)
                        validation = classification_scores(data.y[fold.validation],
                                                           trained.validation_probability)
                        test = classification_scores(truth, trained.test_probability)
                        metrics.append({
                            "ticker": ticker, "fold": number, "feature_set": feature_set,
                            "lookback": window, "seed": seed,
                            "validation_correct_days": validation["correct_days"],
                            "validation_days": len(fold.validation),
                            "validation_accuracy": validation["accuracy"],
                            "validation_brier": validation["brier"],
                            "test_correct_days": test["correct_days"],
                            "test_days": len(fold.test),
                            "test_accuracy": test["accuracy"],
                            "test_brier": test["brier"],
                            "test_predicted_up_rate": test["predicted_up_rate"],
                            "baseline_correct_days": majority_score["correct_days"],
                            "baseline_accuracy": majority_score["accuracy"],
                            "best_epoch": trained.best_epoch,
                        })
                        predictions.extend({
                            "ticker": ticker, "fold": number, "feature_set": feature_set,
                            "lookback": window, "seed": seed, "Date": date,
                            "ActualUp": int(actual), "p_up": float(probability),
                            "pred_up": int(probability >= 0.5),
                        } for date, actual, probability in
                            zip(data.dates[fold.test], truth, trained.test_probability))
    frame = pd.DataFrame(metrics)
    frame.to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False,
                                      float_format="%.6f")
    seed_summary = frame.groupby(["ticker", "feature_set", "lookback", "seed"],
                                 as_index=False).agg(
        correct_days=("test_correct_days", "sum"),
        test_days=("test_days", "sum"),
        baseline_correct_days=("baseline_correct_days", "sum"),
        mean_brier=("test_brier", "mean"),
        mean_predicted_up_rate=("test_predicted_up_rate", "mean"),
    )
    seed_summary["accuracy"] = seed_summary.correct_days / seed_summary.test_days
    seed_summary["baseline_accuracy"] = (seed_summary.baseline_correct_days /
                                         seed_summary.test_days)
    seed_summary.to_csv(output_dir / "seed_summary.csv", index=False, float_format="%.6f")
    summary = seed_summary.groupby(["ticker", "feature_set", "lookback"],
                                   as_index=False).agg(
        mean_correct_days=("correct_days", "mean"),
        min_correct_days=("correct_days", "min"),
        max_correct_days=("correct_days", "max"),
        mean_accuracy=("accuracy", "mean"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        mean_brier=("mean_brier", "mean"),
        baseline_accuracy=("baseline_accuracy", "first"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    metadata = {
        "purpose": "exploratory sensitivity, not model selection on test data",
        "architecture": "gru", "hidden_size": 32, "epochs_max": epochs,
        "seeds": SEEDS, "windows": WINDOWS, "feature_sets": FEATURE_SETS,
        "common_target_dates": "all variants start with a 20-day series; 5-day inputs slice its final 5 days",
        "folds": "three expanding chronological folds, each with 100 validation and 100 test days",
        "device": device, "sources": source_info,
        "test_note": "2025 data have been inspected; do not select a winner by these scores",
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False,
                                                          indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs" / "ablation-study" / "gru")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run(args.output_dir, device=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
