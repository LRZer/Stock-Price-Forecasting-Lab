"""Check readiness, then run the frozen first-100-new-days direction evaluation once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.download_data import SOURCE_BASE, download_one

from .neural_direction import (PRIMARY_NAMES, ROOT, fit_classifier,
                               save_fold_report)
from .neural_direction_data import DirectionFold, load_direction_series, scale_for_fold
from .uncertainty import paired_accuracy_difference


TICKERS = ("GOOG", "AAPL", "TSLA")
CUTOFF = "2026-09-25"
FUTURE_START = "2026-09-26"
TEST_DAYS = 100
VALIDATION_DAYS = 150
DATA_DIR = ROOT / "data" / "prospective-100"
RUN_DIR = ROOT / "runs" / "prospective-100"
STAGE_DIR = ROOT.parent.parent / "work" / "prospective-first-100-staging"


def check_frozen_sources() -> None:
    for manifest_name in ("FUTURE_BASE_HASHES.json", "FUTURE_CODE_HASHES.json"):
        manifest = json.loads((ROOT / manifest_name).read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source changed or disappeared: {relative}")


def available_new_days() -> dict[str, pd.DataFrame]:
    """Read source data, without saving labels or running a model."""
    return {ticker: download_one(ticker, FUTURE_START, "2100-01-01")
            for ticker in TICKERS}


def load_pinned_2026_context() -> dict[str, pd.DataFrame]:
    """Fetch deleted historical context transiently; reject any source revision."""
    manifest = json.loads((ROOT / "FUTURE_BASE_HASHES.json").read_text(encoding="utf-8"))
    expected = manifest["removed_2026_context_sha256"]
    context = {}
    for ticker in TICKERS:
        frame = download_one(ticker, "2026-01-01", "2026-09-26")
        if (len(frame) != 184 or frame.Date.iloc[0] != "2026-01-02" or
                frame.Date.iloc[-1] != CUTOFF):
            raise ValueError(f"{ticker}: removed 2026 context has changed dates")
        digest = hashlib.sha256(frame.to_csv(index=False, float_format="%.6f")
                                .encode("utf-8")).hexdigest()
        if digest != expected[ticker]:
            raise ValueError(f"{ticker}: removed 2026 context differs from pinned hash")
        context[ticker] = frame
    return context


def first_common_100(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame] | None:
    if any(len(data[ticker]) < TEST_DAYS for ticker in TICKERS):
        return None
    first = {ticker: data[ticker].iloc[:TEST_DAYS].copy() for ticker in TICKERS}
    reference = first[TICKERS[0]]["Date"].tolist()
    if any(first[ticker]["Date"].tolist() != reference for ticker in TICKERS[1:]):
        raise ValueError("The first 100 new dates differ across tickers; preserve data and review")
    if reference[0] <= CUTOFF or not pd.Index(reference).is_monotonic_increasing:
        raise ValueError("New target dates must be strictly after the frozen cutoff")
    return first


def frozen_fold(series) -> DirectionFold:
    pre_count = int((series.sample_dates <= CUTOFF).sum())
    if pre_count <= VALIDATION_DAYS + 100 or series.sample_count - pre_count != TEST_DAYS:
        raise ValueError("Unexpected frozen history or prospective test length")
    fold = DirectionFold(np.arange(pre_count - VALIDATION_DAYS),
                         np.arange(pre_count - VALIDATION_DAYS, pre_count),
                         np.arange(pre_count, pre_count + TEST_DAYS))
    if series.sample_dates[fold.validation[-1]] != CUTOFF:
        raise ValueError("Frozen validation period does not end at cutoff")
    return fold


def finish_staged_run() -> None:
    """Promote only a complete, hashed run; allow recovery after an interrupted move."""
    check_frozen_sources()
    marker = STAGE_DIR / "READY.json"
    if not marker.is_file():
        raise FileNotFoundError("No complete staged run; preserve any partial files for review")
    staged_data, staged_runs = STAGE_DIR / "data", STAGE_DIR / "runs"
    if (staged_data.exists() and DATA_DIR.exists()) or (staged_runs.exists() and RUN_DIR.exists()):
        raise FileExistsError("Both staged and final prospective directories exist; review before moving")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    required = {f"data/{ticker}.csv" for ticker in TICKERS}
    required.update({"runs/run_start.json", "runs/all_tickers.csv", "runs/paired_uncertainty.csv",
                     "runs/ARTIFACT_HASHES.json"})
    for ticker in TICKERS:
        required.update({f"runs/{ticker.lower()}/{name}" for name in
                         ("metrics.csv", "predictions.csv", "summary.json",
                          "paired_uncertainty.json")})
    if not required.issubset(manifest["files"]):
        raise ValueError("Staged run is missing required raw data or reports")
    for relative, expected in manifest["files"].items():
        part, _, remainder = relative.partition("/")
        if part not in ("data", "runs") or not remainder:
            raise ValueError(f"Invalid staged path: {relative}")
        base = (staged_data if staged_data.exists() else DATA_DIR) if part == "data" else (
            staged_runs if staged_runs.exists() else RUN_DIR)
        path = base / remainder
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Staged file changed or disappeared: {relative}")
    if not DATA_DIR.exists():
        staged_data.rename(DATA_DIR)
    if not RUN_DIR.exists():
        staged_runs.rename(RUN_DIR)
    marker.unlink()
    STAGE_DIR.rmdir()
    print(f"Saved frozen first-100-new-days result: {RUN_DIR.resolve()}")


def run_once(first: dict[str, pd.DataFrame]) -> None:
    if DATA_DIR.exists() or RUN_DIR.exists() or STAGE_DIR.exists():
        raise FileExistsError("Prospective data, report or staging already exists; never overwrite")
    # Preflight all combined histories before creating either output directory.
    check_frozen_sources()
    context = load_pinned_2026_context()
    scratch_root = ROOT.parent.parent / "work"
    scratch_root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="prospective-preflight-", dir=scratch_root) as temporary:
        context_paths = {}
        for ticker in TICKERS:
            frame = first[ticker]
            if len(frame) != TEST_DAYS or frame["Date"].iloc[0] <= CUTOFF:
                raise ValueError(f"{ticker}: invalid new test window")
            temp_path = Path(temporary) / f"{ticker}.csv"
            frame.to_csv(temp_path, index=False, float_format="%.6f")
            context_path = Path(temporary) / f"{ticker}-context.csv"
            context[ticker].to_csv(context_path, index=False, float_format="%.6f")
            context_paths[ticker] = context_path
            old = load_direction_series(ROOT / "data" / "raw" / f"{ticker}.csv",
                                        context_path)
            if old.dates[-1] != CUTOFF:
                raise ValueError(f"{ticker}: frozen local history does not end on {CUTOFF}")
            combined = load_direction_series(ROOT / "data" / "raw" / f"{ticker}.csv",
                                             context_path,
                                             temp_path)
            frozen_fold(combined)
        _run_once_with_context(first, context_paths)


def _run_once_with_context(first: dict[str, pd.DataFrame],
                           context_paths: dict[str, Path]) -> None:
    staged_data, staged_runs = STAGE_DIR / "data", STAGE_DIR / "runs"
    staged_data.mkdir(parents=True)
    staged_runs.mkdir()
    metadata = {
        "protocol_sha256": hashlib.sha256((ROOT / "FUTURE_EVAL_PROTOCOL.md").read_bytes()).hexdigest(),
        "code_manifest_sha256": hashlib.sha256((ROOT / "FUTURE_CODE_HASHES.json").read_bytes()).hexdigest(),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "torch": version("torch"),
            "matplotlib": version("matplotlib"),
        },
        "source_url": SOURCE_BASE, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": CUTOFF, "test_days": TEST_DAYS,
        "selected_dates": first[TICKERS[0]]["Date"].tolist(),
        "symbols": TICKERS,
    }
    (staged_runs / "run_start.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                 encoding="utf-8")
    rows = []
    intervals = []
    for ticker in TICKERS:
        new_path = staged_data / f"{ticker}.csv"
        first[ticker].to_csv(new_path, index=False, float_format="%.6f")
        series = load_direction_series(ROOT / "data" / "raw" / f"{ticker}.csv",
                                       context_paths[ticker],
                                       new_path)
        fold = frozen_fold(series)
        data = scale_for_fold(series, fold)
        results = [fit_classifier(name, data, device="auto", seed=42, epochs=20)
                   for name in PRIMARY_NAMES]
        folder = staged_runs / ticker.lower()
        metrics, summary = save_fold_report(
            series, data, results, folder,
            {"mode": "prospective-first-100", "seed": 42, "epochs_max": 20,
             "hidden_size": 32, "batch_size": 64, "learning_rate": 0.001,
             "patience": 5, "cutoff": CUTOFF,
             "evaluation_note": "First 100 new days under frozen 2026-09-27 protocol"},
            save_weights=True)
        if summary["sources"][-1]["sha256"] != hashlib.sha256(new_path.read_bytes()).hexdigest():
            raise ValueError(f"{ticker}: staged source hash mismatch")
        pinned = json.loads((ROOT / "FUTURE_BASE_HASHES.json").read_text(encoding="utf-8"))
        if summary["sources"][-2]["sha256"] != pinned["removed_2026_context_sha256"][ticker]:
            raise ValueError(f"{ticker}: temporary 2026 context hash mismatch")
        summary["sources"][-2]["path"] = f"removed-2026-context/{ticker}.csv"
        summary["sources"][-1]["path"] = str(DATA_DIR / f"{ticker}.csv")
        (folder / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        selected = metrics.loc[metrics.model == summary["primary_selected_by_validation"]].iloc[0]
        baseline = metrics.loc[metrics.model == "train-majority"].iloc[0]
        probabilities = {result.name: result.test_probability for result in results}
        paired = paired_accuracy_difference(data.y[fold.test],
                                            probabilities[selected.model],
                                            probabilities["train-majority"])
        interval_record = {"ticker": ticker, "selected_by_validation": selected.model,
                           "method": "paired moving-block bootstrap",
                           "block_size": 5, "repetitions": 10000, "seed": 42,
                           **paired}
        (folder / "paired_uncertainty.json").write_text(
            json.dumps(interval_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        intervals.append(interval_record)
        rows.append({"ticker": ticker, "selected_by_validation": selected.model,
                     "selected_correct_days": int(selected.test_correct_days),
                     "baseline_correct_days": int(baseline.test_correct_days),
                     "test_days": TEST_DAYS, "selected_brier": float(selected.test_brier),
                     "baseline_brier": float(baseline.test_brier),
                     "accuracy_gain": paired["accuracy_gain"],
                     "gain_95pct_low": paired["block_bootstrap_95pct_low"],
                     "gain_95pct_high": paired["block_bootstrap_95pct_high"]})
    pd.DataFrame(rows).to_csv(staged_runs / "all_tickers.csv", index=False, float_format="%.6f")
    pd.DataFrame(intervals).to_csv(staged_runs / "paired_uncertainty.csv", index=False,
                                   float_format="%.6f")
    files = {path.relative_to(STAGE_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for base in (staged_data, staged_runs) for path in base.rglob("*") if path.is_file()}
    artifact_manifest = staged_runs / "ARTIFACT_HASHES.json"
    artifact_manifest.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
    files["runs/ARTIFACT_HASHES.json"] = hashlib.sha256(artifact_manifest.read_bytes()).hexdigest()
    (STAGE_DIR / "READY.json").write_text(json.dumps({"files": files}, indent=2) + "\n",
                                          encoding="utf-8")
    finish_staged_run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true",
                        help="Run once only when all three stocks have 100 new days")
    action.add_argument("--finish-staged", action="store_true",
                        help="Finish moving a fully generated staged run without retraining")
    args = parser.parse_args()
    check_frozen_sources()
    if args.finish_staged:
        finish_staged_run()
        return
    data = available_new_days()
    counts = {ticker: len(frame) for ticker, frame in data.items()}
    print("New complete daily rows after cutoff:", counts)
    first = first_common_100(data)
    if first is None:
        print("Not ready: no model training or partial test report was run")
        return
    print(f"Ready: common first 100 new dates {first[TICKERS[0]].Date.iloc[0]} "
          f"to {first[TICKERS[0]].Date.iloc[-1]}")
    if args.run:
        run_once(first)


if __name__ == "__main__":
    main()
