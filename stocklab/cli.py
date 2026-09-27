"""Command-line entry point for small educational forecasting experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from .data import prepare_data
from .models import ALL_MODELS, CLASSICAL_MODELS, NEURAL_MODELS
from .reporting import save_report
from .training import fit_classical, fit_neural


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("naive", "ridge", "hist-gbdt", "stacked", "lstm", "gru",
                  "attention-is-all-you-need", "cnn-seq2seq")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-models", help="Show available forecasting models")
    compare = subparsers.add_parser("compare", help="Train and evaluate models on one stock")
    compare.add_argument("--ticker", default="GOOG", choices=("GOOG", "AAPL", "TSLA"))
    compare.add_argument("--data", type=Path, help="Override the CSV input path")
    compare.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                         choices=ALL_MODELS)
    compare.add_argument("--all-models", action="store_true",
                         help="Train all four baselines and 18 neural variants")
    compare.add_argument("--lookback", type=int, default=20)
    compare.add_argument("--train-fraction", type=float, default=0.70)
    compare.add_argument("--validation-fraction", type=float, default=0.15)
    compare.add_argument("--epochs", type=int, default=15)
    compare.add_argument("--batch-size", type=int, default=64)
    compare.add_argument("--hidden-size", type=int, default=32)
    compare.add_argument("--learning-rate", type=float, default=0.001)
    compare.add_argument("--patience", type=int, default=5)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    compare.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.command == "list-models":
        print("Classical:", ", ".join(CLASSICAL_MODELS))
        print("Neural:", ", ".join(NEURAL_MODELS))
        return

    data_path = args.data or ROOT / "data" / "raw" / f"{args.ticker}.csv"
    if not data_path.is_file():
        parser.error(f"Data file not found: {data_path}. Run scripts/download_data.py first.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested, but PyTorch cannot access a CUDA GPU")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    names = list(ALL_MODELS) if args.all_models else list(dict.fromkeys(args.models))
    if "naive" not in names:
        names.insert(0, "naive")
    if args.output_dir:
        output_dir = args.output_dir
    elif args.all_models:
        output_dir = ROOT / "runs" / f"{args.ticker.lower()}-all"
    elif tuple(names) == DEFAULT_MODELS:
        output_dir = ROOT / "runs" / args.ticker.lower()
    else:
        identifier = hashlib.sha256(",".join(names).encode()).hexdigest()[:8]
        output_dir = ROOT / "runs" / f"{args.ticker.lower()}-{identifier}"
    data = prepare_data(data_path, lookback=args.lookback,
                        train_fraction=args.train_fraction,
                        validation_fraction=args.validation_fraction)
    print(f"Data: {data.source.name}, {data.rows} rows")
    print(f"Targets: train={len(data.train)}, validation={len(data.validation)}, test={len(data.test)}")
    print(f"Neural device: {device}")
    results = []
    for name in names:
        print(f"Training {name}...", flush=True)
        if name in CLASSICAL_MODELS:
            result = fit_classical(name, data, seed=args.seed)
        else:
            result = fit_neural(name, data, device=device, seed=args.seed,
                                hidden_size=args.hidden_size, epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate, patience=args.patience)
        results.append(result)
        print(f"  {result.seconds:.1f}s, {len(result.history)} epochs", flush=True)

    settings = {"models": names, "seed": args.seed, "device": device,
                "epochs_max": args.epochs, "batch_size": args.batch_size,
                "hidden_size": args.hidden_size, "learning_rate": args.learning_rate,
                "patience": args.patience,
                "train_fraction": args.train_fraction,
                "validation_fraction": args.validation_fraction}
    metrics = save_report(data, results, output_dir, settings)
    print(metrics[["model", "validation_mae", "test_mae", "test_rmse",
                   "test_mae_improvement_vs_naive_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
