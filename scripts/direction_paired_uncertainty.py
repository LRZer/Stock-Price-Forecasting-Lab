"""Describe selected-minus-baseline directional accuracy from saved daily forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab.uncertainty import paired_accuracy_difference


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _read_fold(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    chosen = summary["primary_selected_by_validation"]
    predictions = pd.read_csv(folder / "predictions.csv")
    return (predictions.ActualUp.to_numpy(dtype=np.int8),
            predictions[f"{chosen}_p_up"].to_numpy(dtype=float),
            predictions["train-majority_p_up"].to_numpy(dtype=float), chosen)


def one_period(period: str, ticker: str, folders: list[Path]) -> dict:
    parts = [_read_fold(folder) for folder in folders]
    actual = np.concatenate([item[0] for item in parts])
    selected = np.concatenate([item[1] for item in parts])
    baseline = np.concatenate([item[2] for item in parts])
    dates = [pd.read_csv(folder / "predictions.csv", usecols=["Date"]).Date
             for folder in folders]
    joined = pd.concat(dates, ignore_index=True)
    if not joined.is_monotonic_increasing or joined.duplicated().any():
        raise ValueError(f"{period} {ticker}: target dates overlap or are out of order")
    return {"period": period, "ticker": ticker,
            "selected_by_validation": ",".join(item[3] for item in parts),
            "first_date": joined.iloc[0], "last_date": joined.iloc[-1],
            **paired_accuracy_difference(actual, selected, baseline)}


def main() -> None:
    rows = []
    for ticker in ("GOOG", "AAPL", "TSLA"):
        slug = ticker.lower()
        rows.append(one_period("fixed-2025", ticker,
                               [RUNS / "neural-classifiers" / "fixed-2025" / slug]))
        rows.append(one_period("rolling-2025", ticker,
                               [RUNS / "neural-classifiers" / "rolling-2025" / slug /
                                f"fold-{number}" for number in (1, 2, 3)]))
    for ticker in ("ACN", "RMD"):
        rows.append(one_period("external-symbol-check", ticker,
                               [RUNS / "external-symbol-check" / ticker.lower()]))
    output = RUNS / "direction-paired-uncertainty.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")
    print(pd.DataFrame(rows)[["period", "ticker", "selected_correct_days",
                              "baseline_correct_days", "accuracy_gain",
                              "block_bootstrap_95pct_low",
                              "block_bootstrap_95pct_high"]].to_string(index=False,
                                                                float_format=lambda value: f"{value:.4f}"))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
