"""Backfill individual model charts from saved predictions without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stocklab.neural_diagnostics import render_all_model_diagnostics


ROOT = Path(__file__).resolve().parents[1]


def report_folders() -> list[Path]:
    runs = ROOT / "runs"
    folders = list((runs / "neural-classifiers" / "fixed-2025").glob("*/metrics.csv"))
    folders += list((runs / "neural-classifiers" / "rolling-2025").glob("*/fold-*/metrics.csv"))
    folders += list((runs / "external-symbol-check").glob("*/metrics.csv"))
    folders += list((runs / "prospective-100").glob("*/metrics.csv"))
    return sorted(path.parent for path in folders)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rerender existing model pages")
    args = parser.parse_args()
    total = 0
    for folder in report_folders():
        confusion = pd.read_csv(folder / "confusion.csv")
        destination = folder / "diagnostics" / "models"
        if not args.force and all((destination / f"{name}.png").exists()
                                  for name in confusion.model):
            print(f"Already rendered: {folder}")
            continue
        predictions = pd.read_csv(folder / "predictions.csv")
        calibration = pd.read_csv(folder / "calibration.csv")
        count = render_all_model_diagnostics(predictions, confusion, calibration, destination)
        total += count
        print(f"Rendered {count}: {folder}", flush=True)
    print(f"Rendered {total} model pages across {len(report_folders())} fold reports")


if __name__ == "__main__":
    main()
