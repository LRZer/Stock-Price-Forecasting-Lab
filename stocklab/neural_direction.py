"""Compare direct next-day up classifiers on identical dated sequences."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .direction_study import classification_scores
from .models import DIRECTION_NEURAL_MODELS, build_model
from .neural_diagnostics import (calibration_rows, confusion_row,
                                 plot_selected_diagnostics,
                                 render_all_model_diagnostics)
from .neural_direction_data import (FEATURE_NAMES, DirectionFold, DirectionSeries,
                                    ScaledSequences, fixed_2025_fold,
                                    load_direction_series, rolling_folds, scale_for_fold)


ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("GOOG", "AAPL", "TSLA")
SIMPLE_NAMES = ("logit-window", "gbdt-window")
ALL_NAMES = ("train-majority",) + SIMPLE_NAMES + DIRECTION_NEURAL_MODELS
# Fixed by structural diversity, before any new-symbol evaluation.
PRIMARY_NAMES = ("train-majority",) + SIMPLE_NAMES + (
    "gru", "tcn-residual", "gru-attention")


@dataclass
class DirectionResult:
    name: str
    validation_probability: np.ndarray
    test_probability: np.ndarray
    history: list[dict]
    model: nn.Module | None
    best_epoch: int
    seconds: float


def _predict_probability(model: nn.Module, x: np.ndarray,
                         device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def _summarize_window(x: np.ndarray) -> np.ndarray:
    # Each candidate sees the same dated history; compact summaries keep the
    # classical baselines from fitting 120 unrelated lag coefficients.
    return np.concatenate((x[:, -1], x[:, -5:].mean(axis=1),
                           x.mean(axis=1), x.std(axis=1)), axis=1)


def fit_classifier(name: str, data: ScaledSequences, *, device: str = "auto",
                   seed: int = 42, hidden_size: int = 32, epochs: int = 20,
                   batch_size: int = 64, patience: int = 5,
                   learning_rate: float = 0.001) -> DirectionResult:
    fold = data.fold
    if name == "train-majority":
        prevalence = float(data.y[fold.train].mean())
        return DirectionResult(name,
                               np.full(len(fold.validation), prevalence),
                               np.full(len(fold.test), prevalence),
                               [], None, 0, 0.0)
    if name in SIMPLE_NAMES:
        started = time.perf_counter()
        x = _summarize_window(data.x)
        y = data.y[fold.train].astype(np.int8)
        if len(np.unique(y)) != 2:
            raise ValueError("Training period needs both directions")
        if name == "logit-window":
            estimator = make_pipeline(StandardScaler(),
                                      LogisticRegression(C=0.1, max_iter=1000))
        else:
            estimator = HistGradientBoostingClassifier(
                max_iter=80, max_leaf_nodes=7, min_samples_leaf=20,
                learning_rate=0.05, l2_regularization=1.0,
                early_stopping=False, random_state=seed)
        estimator.fit(x[fold.train], y)
        probability = lambda rows: estimator.predict_proba(x[rows])[:, 1].astype(np.float64)
        return DirectionResult(name, probability(fold.validation), probability(fold.test),
                               [], None, 0, time.perf_counter() - started)
    if name not in DIRECTION_NEURAL_MODELS or min(epochs, batch_size, patience) < 1:
        raise ValueError("Unknown model or invalid training settings")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch_device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(name, lookback=data.x.shape[1], hidden_size=hidden_size,
                        input_size=data.x.shape[2], task="direction").to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    training = TensorDataset(torch.from_numpy(data.x[fold.train]),
                             torch.from_numpy(data.y[fold.train]))
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_brier = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            binary_loss = loss_function(logits, batch_y)
            loss = binary_loss + 0.001 * model.aux_loss
            loss.backward()
            optimizer.step()
            train_loss_sum += float(binary_loss.detach().cpu()) * len(batch_y)
        validation_probability = _predict_probability(model, data.x[fold.validation],
                                                       torch_device, batch_size)
        validation = classification_scores(data.y[fold.validation], validation_probability)
        history.append({"epoch": epoch,
                        "train_binary_cross_entropy": train_loss_sum / len(fold.train),
                        "validation_accuracy": validation["accuracy"],
                        "validation_brier": validation["brier"]})
        if validation["brier"] < best_brier - 1e-6:
            best_brier = validation["brier"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    validation_probability = _predict_probability(model, data.x[fold.validation],
                                                   torch_device, batch_size)
    test_probability = _predict_probability(model, data.x[fold.test],
                                             torch_device, batch_size)
    return DirectionResult(name, validation_probability, test_probability,
                           history, model, best_epoch, time.perf_counter() - start_time)


def _split_dates(data: ScaledSequences, rows: np.ndarray) -> dict:
    return {"targets": len(rows), "first_date": str(data.dates[rows[0]]),
            "last_date": str(data.dates[rows[-1]])}


def save_fold_report(series: DirectionSeries, data: ScaledSequences,
                     results: list[DirectionResult], output_dir: Path,
                     settings: dict, *, save_weights: bool) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_weights:
        (output_dir / "models").mkdir(exist_ok=True)
    fold = data.fold
    validation_truth = data.y[fold.validation].astype(np.int8)
    test_truth = data.y[fold.test].astype(np.int8)
    rows = []
    histories = []
    confusion_rows = []
    calibration = []
    predictions = pd.DataFrame({"Date": data.dates[fold.test], "ActualUp": test_truth})
    for result in results:
        validation = classification_scores(validation_truth, result.validation_probability)
        test = classification_scores(test_truth, result.test_probability)
        rows.append({"model": result.name,
                     "validation_correct_days": validation["correct_days"],
                     "validation_days": len(fold.validation),
                     "validation_accuracy": validation["accuracy"],
                     "validation_brier": validation["brier"],
                     "test_correct_days": test["correct_days"],
                     "test_days": len(fold.test),
                     "test_accuracy": test["accuracy"],
                     "test_balanced_accuracy": test["balanced_accuracy"],
                     "test_brier": test["brier"],
                     "test_predicted_up_rate": test["predicted_up_rate"],
                     "test_actual_up_rate": test["actual_up_rate"],
                     "best_epoch": result.best_epoch,
                     "epochs_run": len(result.history),
                     "seconds": result.seconds})
        predictions[f"{result.name}_p_up"] = result.test_probability
        predictions[f"{result.name}_pred_up"] = (result.test_probability >= 0.5).astype(np.int8)
        confusion_rows.append({"model": result.name,
                               **confusion_row(test_truth, result.test_probability)})
        calibration.extend({"model": result.name, **item}
                           for item in calibration_rows(test_truth, result.test_probability))
        histories.extend({"model": result.name, **item} for item in result.history)
        if save_weights and result.model is not None:
            weights = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
            torch.save(weights, output_dir / "models" / f"{result.name}.pt")

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["validation_accuracy", "validation_brier"],
                                  ascending=[False, True], kind="stable").reset_index(drop=True)
    selected = str(metrics.iloc[0]["model"])
    primary = str(metrics.loc[metrics.model.isin(PRIMARY_NAMES)].iloc[0]["model"])
    metrics.to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(output_dir / "predictions.csv", index=False, float_format="%.6f")
    confusion_frame = pd.DataFrame(confusion_rows)
    calibration_frame = pd.DataFrame(calibration)
    confusion_frame.to_csv(output_dir / "confusion.csv", index=False, float_format="%.6f")
    calibration_frame.to_csv(output_dir / "calibration.csv", index=False, float_format="%.6f")
    plot_selected_diagnostics(predictions, confusion_frame, calibration_frame, primary,
                              output_dir / "diagnostics.png")
    render_all_model_diagnostics(predictions, confusion_frame, calibration_frame,
                                 output_dir / "diagnostics" / "models")
    pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False,
                                   float_format="%.6f")
    summary = {
        "task": "P(next trading close > current close)",
        "unchanged_close_rule": "non-up class",
        "decision_threshold": 0.5,
        "feature_names": FEATURE_NAMES,
        "lookback": series.lookback,
        "models": [result.name for result in results],
        "selected_by_validation": selected,
        "primary_candidates": [name for name in PRIMARY_NAMES if name in metrics.model.values],
        "primary_selected_by_validation": primary,
        "selection_rule": "highest validation accuracy; lowest validation Brier breaks ties",
        "epoch_selection": "lowest validation Brier within each neural model",
        "splits": {"train": _split_dates(data, fold.train),
                   "validation": _split_dates(data, fold.validation),
                   "test": _split_dates(data, fold.test)},
        "train_only_feature_mean": data.mean.tolist(),
        "train_only_feature_scale": data.scale.tolist(),
        "sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in series.sources],
        "settings": settings,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "note": settings.get("evaluation_note",
                             "The 2025 period has been inspected before; results are retrospective learning evidence"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    ordered = metrics.sort_values("test_accuracy")
    colors = ["#999999" if name == "train-majority" else
              ("#ed8c2b" if name == selected else "#3377bb") for name in ordered["model"]]
    fig, ax = plt.subplots(figsize=(9, max(5, len(ordered) * 0.36)))
    ax.barh(ordered["model"], ordered["test_accuracy"] * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Correct next-day directions (%)")
    ax.set_title(f"Selected by validation: {selected}; gray = train-majority")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy.png", dpi=150)
    plt.close(fig)
    return metrics, summary


def run_one(ticker: str, *, mode: str, output_dir: Path,
            device: str, epochs: int, seed: int,
            names: tuple[str, ...] = ALL_NAMES,
            source_path: Path | None = None) -> dict:
    series = load_direction_series(source_path or ROOT / "data" / "raw" / f"{ticker}.csv")
    folds = [fixed_2025_fold(series)] if mode == "fixed-2025" else rolling_folds(series)
    fold_metrics = []
    selected_rows = []
    for number, fold in enumerate(folds, 1):
        data = scale_for_fold(series, fold)
        results = []
        for name in names:
            print(f"{ticker} fold {number}/{len(folds)}: {name}", flush=True)
            results.append(fit_classifier(name, data, device=device, epochs=epochs, seed=seed))
        folder = output_dir / (f"fold-{number}" if mode == "rolling-2025" else "")
        metrics, summary = save_fold_report(series, data, results, folder,
                                            {"mode": mode, "fold": number,
                                             "seed": seed, "epochs_max": epochs,
                                             "device": device, "hidden_size": 32,
                                            "batch_size": 64, "learning_rate": 0.001,
                                             "patience": 5,
                                             "evaluation_note": (
                                                 "Frozen external-symbol check on ACN and RMD; historical-date evidence"
                                                 if source_path is not None else
                                                 "The 2025 period has been inspected before; retrospective learning evidence")},
                                            save_weights=(mode == "fixed-2025"))
        chosen = metrics.iloc[0]
        primary = metrics.loc[metrics.model == summary["primary_selected_by_validation"]].iloc[0]
        majority = metrics.loc[metrics["model"] == "train-majority"].iloc[0]
        selected_rows.append({"fold": number, "selected": summary["selected_by_validation"],
                              "selected_correct_days": int(chosen["test_correct_days"]),
                              "selected_test_accuracy": float(chosen["test_accuracy"]),
                              "primary_selected": summary["primary_selected_by_validation"],
                              "primary_correct_days": int(primary["test_correct_days"]),
                              "primary_test_accuracy": float(primary["test_accuracy"]),
                              "baseline_correct_days": int(majority["test_correct_days"]),
                              "baseline_test_accuracy": float(majority["test_accuracy"]),
                              "test_days": len(fold.test)})
        metrics.insert(0, "fold", number)
        fold_metrics.append(metrics)
        print(f"Primary {summary['primary_selected_by_validation']}: "
              f"{int(primary['test_correct_days'])}/{len(fold.test)} = {primary['test_accuracy']:.1%}; "
              f"baseline {majority['test_accuracy']:.1%}", flush=True)

    selections = pd.DataFrame(selected_rows)
    if mode == "rolling-2025":
        output_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = pd.concat(fold_metrics, ignore_index=True)
        all_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, float_format="%.6f")
        selections.to_csv(output_dir / "selected_by_fold.csv", index=False, float_format="%.6f")
        rollup = all_metrics.groupby("model", sort=False).agg(
            total_correct_days=("test_correct_days", "sum"),
            total_test_days=("test_days", "sum"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
            mean_test_brier=("test_brier", "mean"),
        ).reset_index()
        rollup["overall_test_accuracy"] = rollup["total_correct_days"] / rollup["total_test_days"]
        rollup.to_csv(output_dir / "rolling_metrics.csv", index=False, float_format="%.6f")
    total_days = int(selections["test_days"].sum())
    return {"ticker": ticker, "mode": mode,
            "primary_correct_days": int(selections["primary_correct_days"].sum()),
            "primary_accuracy": float(selections["primary_correct_days"].sum() / total_days),
            "primary_selected_models": ",".join(selections["primary_selected"]),
            "selected_correct_days": int(selections["selected_correct_days"].sum()),
            "baseline_correct_days": int(selections["baseline_correct_days"].sum()),
            "test_days": total_days,
            "selected_accuracy": float(selections["selected_correct_days"].sum() / total_days),
            "baseline_accuracy": float(selections["baseline_correct_days"].sum() / total_days),
            "selected_models": ",".join(selections["selected"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed-2025", "rolling-2025"), default="fixed-2025")
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=list(TICKERS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "neural-classifiers")
    args = parser.parse_args()
    destination = args.output_dir / args.mode
    rows = [run_one(ticker, mode=args.mode, output_dir=destination / ticker.lower(),
                    device=args.device, epochs=args.epochs, seed=args.seed)
            for ticker in args.tickers]
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "all_tickers.csv", index=False, float_format="%.6f")
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
