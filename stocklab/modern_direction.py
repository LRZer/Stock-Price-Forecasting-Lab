"""Exploratory modern classifiers; keeps the frozen six-candidate study intact."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .direction_study import classification_scores
from .modern_direction_models import MODERN_DIRECTION_MODELS, build_modern_direction_model
from .neural_direction import (DirectionResult, ROOT, TICKERS, _predict_probability,
                               fit_classifier, save_fold_report)
from .neural_direction_data import (ScaledSequences, fixed_2025_fold,
                                    load_direction_series, rolling_folds, scale_for_fold)


def fit_modern_classifier(name: str, data: ScaledSequences, *, device: str = "auto",
                          seed: int = 42, hidden_size: int = 32, epochs: int = 20,
                          batch_size: int = 64, patience: int = 5,
                          learning_rate: float = 0.001) -> DirectionResult:
    if name not in MODERN_DIRECTION_MODELS or min(epochs, batch_size, patience) < 1:
        raise ValueError("Unknown modern model or invalid training settings")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch_device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_modern_direction_model(name, lookback=data.x.shape[1],
                                        input_size=data.x.shape[2],
                                        hidden_size=hidden_size).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    fold = data.fold
    training = TensorDataset(torch.from_numpy(data.x[fold.train]),
                             torch.from_numpy(data.y[fold.train]))
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_brier = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            binary_loss = loss_function(model(batch_x), batch_y)
            binary_loss.backward()
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
    return DirectionResult(name, validation_probability, test_probability, history,
                           model, best_epoch, time.perf_counter() - started)


def run_modern_one(ticker: str, *, mode: str, output_dir: Path, device: str,
                   epochs: int, seed: int) -> None:
    series = load_direction_series(ROOT / "data" / "raw" / f"{ticker}.csv")
    folds = [fixed_2025_fold(series)] if mode == "fixed-2025" else rolling_folds(series)
    metrics_by_fold = []
    for number, fold in enumerate(folds, 1):
        data = scale_for_fold(series, fold)
        results = [fit_classifier("train-majority", data)]
        for name in MODERN_DIRECTION_MODELS:
            print(f"{ticker} fold {number}/{len(folds)}: {name}", flush=True)
            results.append(fit_modern_classifier(name, data, device=device,
                                                 epochs=epochs, seed=seed))
        folder = output_dir / (f"fold-{number}" if mode == "rolling-2025" else "")
        metrics, _ = save_fold_report(
            series, data, results, folder,
            {"mode": mode, "fold": number, "seed": seed, "epochs_max": epochs,
             "device": device, "hidden_size": 32, "batch_size": 64,
             "learning_rate": 0.001, "patience": 5,
             "evaluation_note": "Exploratory 2025 comparison after historical results were inspected; "
                                "not a candidate in the frozen future evaluation"},
            save_weights=(mode == "fixed-2025"))
        # Keep enough digits to reconstruct the 0.5 decision on borderline days.
        daily = pd.DataFrame({"Date": data.dates[fold.test],
                              "ActualUp": data.y[fold.test].astype(np.int8)})
        for result in results:
            daily[f"{result.name}_p_up"] = result.test_probability
            daily[f"{result.name}_pred_up"] = (result.test_probability >= 0.5).astype(np.int8)
        daily.to_csv(folder / "predictions.csv", index=False, float_format="%.17g")
        metrics.insert(0, "fold", number)
        metrics_by_fold.append(metrics)
    if mode == "rolling-2025":
        output_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = pd.concat(metrics_by_fold, ignore_index=True)
        all_metrics.to_csv(output_dir / "fold_metrics.csv", index=False,
                           float_format="%.6f")
        summary = all_metrics.groupby("model", sort=False).agg(
            total_correct_days=("test_correct_days", "sum"),
            total_test_days=("test_days", "sum"),
            mean_test_brier=("test_brier", "mean")).reset_index()
        summary["overall_test_accuracy"] = (summary.total_correct_days /
                                            summary.total_test_days)
        summary.to_csv(output_dir / "rolling_metrics.csv", index=False,
                       float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed-2025", "rolling-2025"),
                        default="fixed-2025")
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=list(TICKERS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs" / "modern-direction")
    args = parser.parse_args()
    destination = args.output_dir / args.mode
    for ticker in args.tickers:
        run_modern_one(ticker, mode=args.mode,
                       output_dir=destination / ticker.lower(),
                       device=args.device, epochs=args.epochs, seed=args.seed)
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
