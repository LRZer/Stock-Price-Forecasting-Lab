"""Small, leakage-aware close-only versus OHLCV return study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .features import (CLOSE_FEATURES, OHLCV_FEATURES, FeatureSamples,
                       TimeSplit, holdout_split, make_samples, rolling_splits)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("naive", "ridge-close", "ridge-ohlcv", "gbdt-close", "gbdt-ohlcv")
TICKERS = ("GOOG", "AAPL", "TSLA")


def _estimator(name: str):
    if name.startswith("ridge-"):
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name.startswith("gbdt-"):
        return HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=7, min_samples_leaf=20,
            learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, samples: FeatureSamples, train: np.ndarray):
    if name == "naive":
        return None
    model = _estimator(name)
    model.fit(samples.features_for(name)[train], samples.actual_return[train] * 100)
    return model


def predict_return(name: str, model, samples: FeatureSamples, rows: np.ndarray) -> np.ndarray:
    if name == "naive":
        return np.zeros(len(rows), dtype=np.float64)
    return np.asarray(model.predict(samples.features_for(name)[rows]), dtype=np.float64) / 100


def score(samples: FeatureSamples, rows: np.ndarray,
          predicted_return: np.ndarray) -> dict[str, float | None]:
    previous = samples.previous_close[rows]
    actual = samples.actual_close[rows]
    actual_return = samples.actual_return[rows]
    predicted = previous * (1 + predicted_return)
    error = actual - predicted
    denominator = np.square(actual_return).sum()
    nonzero = predicted_return != 0
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "return_r2_vs_zero": float(1 - np.square(actual_return - predicted_return).sum() / denominator)
        if denominator else None,
        "direction_accuracy": float(np.mean(np.sign(actual_return[nonzero]) ==
                                            np.sign(predicted_return[nonzero])))
        if nonzero.any() else None,
    }


def _evaluate(samples: FeatureSamples, split: TimeSplit, *, refit_for_test: bool):
    validation_predictions = {}
    validation_scores = {}
    for name in MODELS:
        model = fit_model(name, samples, split.train)
        prediction = predict_return(name, model, samples, split.validation)
        validation_predictions[name] = prediction
        validation_scores[name] = score(samples, split.validation, prediction)
    selected = min(MODELS, key=lambda name: validation_scores[name]["mae"])

    test_predictions = {}
    fitted_models = {}
    for name in MODELS:
        if refit_for_test:
            train_rows = np.concatenate((split.train, split.validation))
            model = fit_model(name, samples, train_rows)
        else:
            model = fit_model(name, samples, split.train)
        fitted_models[name] = model
        test_predictions[name] = predict_return(name, model, samples, split.test)

    naive_mae = score(samples, split.test, test_predictions["naive"])["mae"]
    metrics = []
    for name in MODELS:
        test = score(samples, split.test, test_predictions[name])
        metrics.append({
            "model": name,
            "validation_mae": validation_scores[name]["mae"],
            "test_mae": test["mae"],
            "test_rmse": test["rmse"],
            "test_mae_improvement_vs_naive_pct": 100 * (naive_mae - test["mae"]) / naive_mae,
            "test_return_r2_vs_zero": test["return_r2_vs_zero"],
            "test_direction_accuracy": test["direction_accuracy"],
        })
    predictions = pd.DataFrame({
        "Date": samples.dates[split.test],
        "Actual": samples.actual_close[split.test],
        "PreviousClose": samples.previous_close[split.test],
        "ActualReturnPct": samples.actual_return[split.test] * 100,
    })
    for name, prediction in test_predictions.items():
        predictions[name] = samples.previous_close[split.test] * (1 + prediction)
    return selected, pd.DataFrame(metrics), predictions, fitted_models


def _source_metadata(samples: FeatureSamples) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in samples.sources]


def _split_metadata(samples: FeatureSamples, split: TimeSplit) -> dict:
    return {part: {"targets": len(getattr(split, part)),
                   "first_date": str(samples.dates[getattr(split, part)[0]]),
                   "last_date": str(samples.dates[getattr(split, part)[-1]])}
            for part in ("train", "validation", "test")}


def run_rolling(ticker: str, samples: FeatureSamples, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    prediction_frames = []
    selection_rows = []
    splits = rolling_splits(samples)
    for fold, split in enumerate(splits, start=1):
        selected, metrics, predictions, _ = _evaluate(samples, split, refit_for_test=False)
        metrics.insert(0, "fold", fold)
        predictions.insert(0, "fold", fold)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        chosen = metrics.loc[metrics["model"] == selected].iloc[0]
        naive = metrics.loc[metrics["model"] == "naive"].iloc[0]
        selection_rows.append({"fold": fold, "test_start": samples.dates[split.test[0]],
                               "test_end": samples.dates[split.test[-1]],
                               "selected_by_validation": selected,
                               "selected_test_mae": chosen["test_mae"],
                               "naive_test_mae": naive["test_mae"]})
        print(f"{ticker} fold {fold}: {selected}; test MAE {chosen['test_mae']:.3f}, "
              f"naive {naive['test_mae']:.3f}", flush=True)
    fold_metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    aggregate = fold_metrics.groupby("model", sort=False).agg(
        mean_validation_mae=("validation_mae", "mean"),
        mean_test_mae=("test_mae", "mean"),
        folds_beating_naive=("test_mae_improvement_vs_naive_pct", lambda x: int((x > 0).sum())),
    ).reset_index()
    baseline = float(aggregate.loc[aggregate["model"] == "naive", "mean_test_mae"].iloc[0])
    aggregate["mean_test_mae_improvement_vs_naive_pct"] = 100 * (
        baseline - aggregate["mean_test_mae"]) / baseline
    fold_metrics.to_csv(destination / "fold_metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(destination / "predictions.csv", index=False, float_format="%.6f")
    selections.to_csv(destination / "selected_by_fold.csv", index=False, float_format="%.6f")
    aggregate.to_csv(destination / "rolling_metrics.csv", index=False, float_format="%.6f")
    summary = {"ticker": ticker, "mode": "exploratory_rolling_2025",
               "sources": _source_metadata(samples),
               "close_features": CLOSE_FEATURES, "ohlcv_features": OHLCV_FEATURES,
               "splits": [_split_metadata(samples, split) for split in splits],
               "models": MODELS, "selection_rule": "lowest validation price MAE per fold",
               "mean_selected_test_mae": float(selections["selected_test_mae"].mean()),
               "mean_naive_test_mae": float(selections["naive_test_mae"].mean())}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    return {"ticker": ticker, "selected_strategy_mae": summary["mean_selected_test_mae"],
            "naive_mae": summary["mean_naive_test_mae"]}


def run_holdout(ticker: str, samples: FeatureSamples, destination: Path,
                *, force: bool = False) -> dict:
    if (destination / "metrics.csv").exists() and not force:
        raise FileExistsError(f"Holdout already evaluated: {destination}. Use --force only deliberately.")
    destination.mkdir(parents=True, exist_ok=True)
    split = holdout_split(samples)
    selected, metrics, predictions, models = _evaluate(samples, split, refit_for_test=True)
    metrics.to_csv(destination / "metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(destination / "predictions.csv", index=False, float_format="%.6f")
    if models[selected] is not None:
        joblib.dump(models[selected], destination / "selected_model.joblib")
    summary = {"ticker": ticker, "mode": "single_predeclared_2026_holdout",
               "sources": _source_metadata(samples),
               "close_features": CLOSE_FEATURES, "ohlcv_features": OHLCV_FEATURES,
               "split": _split_metadata(samples, split),
               "models": MODELS, "selected_by_validation": selected,
               "final_fit": "refit each model on all pre-2026 observations; no 2026 labels used",
               "selection_rule": "lowest pre-2026 validation price MAE"}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(predictions["Date"], predictions["Actual"], label="Actual", color="black", linewidth=1.5)
    ax.plot(predictions["Date"], predictions["naive"], label="Previous close", alpha=0.8)
    if selected != "naive":
        ax.plot(predictions["Date"], predictions[selected], label=f"Selected: {selected}", alpha=0.8)
    ax.set_title(f"{ticker} 2026 holdout; selected on pre-2026 validation")
    ax.set_ylabel("Close price")
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "forecast.png", dpi=150)
    plt.close(fig)
    chosen = metrics.loc[metrics["model"] == selected].iloc[0]
    naive = metrics.loc[metrics["model"] == "naive"].iloc[0]
    print(f"{ticker} 2026 holdout: {selected}; test MAE {chosen['test_mae']:.3f}, "
          f"naive {naive['test_mae']:.3f}", flush=True)
    return {"ticker": ticker, "selected_by_validation": selected,
            "selected_test_mae": float(chosen["test_mae"]),
            "naive_test_mae": float(naive["test_mae"]),
            "selected_improvement_pct": float(chosen["test_mae_improvement_vs_naive_pct"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rolling", "holdout"), default="rolling")
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=list(TICKERS))
    parser.add_argument("--force", action="store_true", help="Allow deliberate holdout re-evaluation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "feature-study")
    args = parser.parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 4, 8)))
    destination = args.output_dir / ("rolling" if args.mode == "rolling" else "holdout-2026")
    rows = []
    for ticker in args.tickers:
        paths = [ROOT / "data" / "raw" / f"{ticker}.csv"]
        if args.mode == "holdout":
            paths.append(ROOT / "data" / "holdout-2026" / f"{ticker}.csv")
        samples = make_samples(*paths)
        run_dir = destination / ticker.lower()
        rows.append(run_rolling(ticker, samples, run_dir) if args.mode == "rolling" else
                    run_holdout(ticker, samples, run_dir, force=args.force))
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "all_tickers.csv", index=False, float_format="%.6f")
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
