"""Evaluate models over several expanding, forward-only time windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from stocklab.data import prepare_rolling_folds
from stocklab.models import ALL_MODELS, CLASSICAL_MODELS
from stocklab.reporting import save_report
from stocklab.training import fit_classical, fit_neural


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", choices=("GOOG", "AAPL", "TSLA"), default="GOOG")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS,
                        default=["naive", "ridge", "gru", "attention-is-all-you-need"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    data_path = args.data or ROOT / "data" / "raw" / f"{args.ticker}.csv"
    if not data_path.is_file():
        parser.error(f"Data file not found: {data_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    names = list(dict.fromkeys(args.models))
    if "naive" not in names:
        names.insert(0, "naive")
    folds = prepare_rolling_folds(data_path, lookback=args.lookback, folds=args.folds,
                                  validation_size=args.validation_size, test_size=args.test_size)
    output_dir = args.output_dir or ROOT / "runs" / f"rolling-{args.ticker.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    selected_rows = []
    for number, data in enumerate(folds, 1):
        print(f"Fold {number}/{len(folds)}: train through {data.train.date_range[1]}, "
              f"test {data.test.date_range[0]} to {data.test.date_range[1]}", flush=True)
        results = []
        for name in names:
            if name in CLASSICAL_MODELS:
                result = fit_classical(name, data, seed=args.seed)
            else:
                result = fit_neural(name, data, device=device, seed=args.seed,
                                    epochs=args.epochs)
            results.append(result)
        settings = {"rolling_fold": number, "rolling_folds": args.folds,
                    "validation_size": args.validation_size, "test_size": args.test_size,
                    "models": names, "seed": args.seed, "device": device,
                    "epochs_max": args.epochs}
        metrics = save_report(data, results, output_dir / f"fold-{number}", settings)
        metrics.insert(0, "fold", number)
        fold_rows.extend(metrics.to_dict("records"))
        chosen = metrics.iloc[0]
        naive = metrics.loc[metrics["model"] == "naive"].iloc[0]
        selected_rows.append({
            "fold": number,
            "test_start": data.test.date_range[0],
            "test_end": data.test.date_range[1],
            "selected_by_validation": chosen["model"],
            "selected_test_mae": chosen["test_mae"],
            "naive_test_mae": naive["test_mae"],
        })

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, float_format="%.6f")
    summary = fold_metrics.groupby("model", as_index=False).agg(
        mean_test_mae=("test_mae", "mean"),
        std_test_mae=("test_mae", "std"),
        mean_validation_mae=("validation_mae", "mean"),
    ).sort_values("mean_validation_mae")
    summary.to_csv(output_dir / "rolling_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_by_fold.csv", index=False,
                                        float_format="%.6f")
    (output_dir / "rolling_summary.json").write_text(json.dumps({
        "data_file": str(data_path.resolve()), "folds": args.folds,
        "test_size": args.test_size, "validation_size": args.validation_size,
        "test_periods_are_disjoint": True,
        "later_folds_refit_on_all_previous_observed_prices": True,
        "models": names, "device": device,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
