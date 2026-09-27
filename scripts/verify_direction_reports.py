"""Check saved direction-study reports without retraining or selecting winners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.render_all_model_diagnostics import report_folders
from stocklab.direction_study import classification_scores
from stocklab.neural_direction import ALL_NAMES, PRIMARY_NAMES, ROOT
from stocklab.neural_diagnostics import calibration_rows, confusion_row


def check_fold(folder: Path) -> int:
    metrics = pd.read_csv(folder / "metrics.csv")
    predictions = pd.read_csv(folder / "predictions.csv")
    confusion = pd.read_csv(folder / "confusion.csv").set_index("model")
    calibration = pd.read_csv(folder / "calibration.csv")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    names = tuple(metrics.model)
    expected = (PRIMARY_NAMES if {"external-symbol-check", "prospective-100"}
                .intersection(folder.parts) else ALL_NAMES)
    assert set(names) == set(expected) == set(summary["models"])
    assert len(names) == len(expected) and set(confusion.index) == set(names)
    assert set(calibration.model) == set(names)
    validation_order = tuple(metrics.sort_values(["validation_accuracy", "validation_brier"],
                                                 ascending=[False, True], kind="stable").model)
    assert names == validation_order
    assert len(predictions) == int(metrics.test_days.iloc[0])
    assert predictions.Date.is_unique and predictions.Date.is_monotonic_increasing
    assert summary["splits"]["test"] == {
        "targets": len(predictions), "first_date": str(predictions.Date.iloc[0]),
        "last_date": str(predictions.Date.iloc[-1])}
    assert set(predictions.ActualUp.unique()) <= {0, 1}
    assert summary["selected_by_validation"] == names[0]
    assert summary["primary_selected_by_validation"] == (
        metrics.loc[metrics.model.isin(PRIMARY_NAMES)].iloc[0].model)
    for source in summary["sources"]:
        if source["path"].startswith("removed-2026-context/"):
            ticker = Path(source["path"]).stem
            pinned = json.loads((ROOT / "FUTURE_BASE_HASHES.json").read_text(encoding="utf-8"))
            assert source["sha256"] == pinned["removed_2026_context_sha256"][ticker]
            continue
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    for row in metrics.itertuples():
        name = row.model
        probability = predictions[f"{name}_p_up"].to_numpy()
        predicted = predictions[f"{name}_pred_up"].to_numpy()
        actual = predictions.ActualUp.to_numpy()
        assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
        assert np.array_equal(predicted, (probability >= 0.5).astype(int))
        correct = int((predicted == actual).sum())
        assert correct == row.test_correct_days
        scores = classification_scores(actual, probability)
        assert abs(float(row.test_accuracy) - scores["accuracy"]) < 2e-6
        assert abs(float(row.test_balanced_accuracy) - scores["balanced_accuracy"]) < 2e-6
        assert abs(float(row.test_brier) - scores["brier"]) < 2e-6
        assert abs(float(row.test_actual_up_rate) - scores["actual_up_rate"]) < 2e-6
        assert abs(float(predicted.mean()) - row.test_predicted_up_rate) < 1e-5
        counts = confusion.loc[name]
        assert int(counts.true_non_up + counts.false_up + counts.missed_up + counts.true_up) == len(actual)
        assert int(counts.true_non_up + counts.true_up) == correct
        expected_confusion = confusion_row(actual, probability)
        for key, value in expected_confusion.items():
            assert abs(float(counts[key]) - value) < 2e-6, (folder, name, key)
        bins = calibration.loc[calibration.model == name].sort_values("bin")
        assert len(bins) == 10 and int(bins.days.sum()) == len(actual)
        assert ((bins.loc[bins.days > 0, "mean_predicted_probability"].between(0, 1))).all()
        expected_bins = calibration_rows(actual, probability)
        for saved, expected in zip(bins.itertuples(index=False), expected_bins):
            assert saved.bin == expected["bin"] and saved.days == expected["days"], (folder, name)
            if saved.days:
                assert abs(saved.mean_predicted_probability - expected["mean_predicted_probability"]) < 2e-6
                assert abs(saved.actual_up_rate - expected["actual_up_rate"]) < 2e-6
        picture = folder / "diagnostics" / "models" / f"{name}.png"
        assert picture.is_file() and picture.stat().st_size > 1000
    return len(names)


def check_ablation() -> tuple[int, int]:
    path = ROOT / "runs" / "ablation-study" / "gru"
    metrics = pd.read_csv(path / "metrics.csv")
    predictions = pd.read_csv(path / "predictions.csv")
    assert len(metrics) == 108 and len(predictions) == 10800
    group = ["ticker", "fold", "feature_set", "lookback", "seed"]
    assert metrics.groupby(group).size().eq(1).all()
    assert predictions.groupby(group).size().eq(100).all()
    for (ticker, fold), dates in predictions.groupby(["ticker", "fold"]):
        assert dates.Date.nunique() == 100
        assert dates.groupby("Date").ActualUp.nunique().eq(1).all()
        assert dates.groupby(group).Date.apply(tuple).nunique() == 1
    for row in metrics.itertuples():
        subset = predictions.loc[(predictions.ticker == row.ticker) &
                                 (predictions.fold == row.fold) &
                                 (predictions.feature_set == row.feature_set) &
                                 (predictions.lookback == row.lookback) &
                                 (predictions.seed == row.seed)]
        assert int((subset.pred_up == subset.ActualUp).sum()) == row.test_correct_days
    return len(metrics), len(predictions)


def check_paired_uncertainty() -> int:
    frame = pd.read_csv(ROOT / "runs" / "direction-paired-uncertainty.csv")
    assert len(frame) == 8
    assert frame[["period", "ticker"]].drop_duplicates().shape[0] == 8
    exact = ((frame.selected_correct_days - frame.baseline_correct_days) /
             frame.test_days)
    assert np.allclose(frame.accuracy_gain, exact, atol=1e-6)
    assert (frame.block_bootstrap_95pct_low <= frame.block_bootstrap_95pct_high).all()
    assert frame[["block_bootstrap_95pct_low", "block_bootstrap_95pct_high"]].abs().le(1).all().all()
    return len(frame)


def check_period_rollups() -> int:
    periods = (
        (ROOT / "runs" / "neural-classifiers" / "fixed-2025", ("GOOG", "AAPL", "TSLA"), 1),
        (ROOT / "runs" / "neural-classifiers" / "rolling-2025", ("GOOG", "AAPL", "TSLA"), 3),
        (ROOT / "runs" / "external-symbol-check", ("ACN", "RMD"), 1),
    )
    checked = 0
    for parent, tickers, fold_count in periods:
        rollup = pd.read_csv(parent / "all_tickers.csv").set_index("ticker")
        assert set(rollup.index) == set(tickers)
        for ticker in tickers:
            folders = ([parent / ticker.lower() / f"fold-{number}"
                        for number in range(1, fold_count + 1)] if fold_count > 1
                       else [parent / ticker.lower()])
            primary_names = []
            all_names = []
            primary_correct = baseline_correct = all_correct = total_days = 0
            all_dates = []
            for folder in folders:
                summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                metrics = pd.read_csv(folder / "metrics.csv").set_index("model")
                predictions = pd.read_csv(folder / "predictions.csv", usecols=["Date"])
                all_dates.extend(predictions.Date.tolist())
                primary = summary["primary_selected_by_validation"]
                selected = summary["selected_by_validation"]
                primary_names.append(primary)
                all_names.append(selected)
                primary_correct += int(metrics.loc[primary].test_correct_days)
                baseline_correct += int(metrics.loc["train-majority"].test_correct_days)
                all_correct += int(metrics.loc[selected].test_correct_days)
                total_days += len(predictions)
            assert pd.Index(all_dates).is_monotonic_increasing and len(all_dates) == len(set(all_dates))
            saved = rollup.loc[ticker]
            assert saved.primary_selected_models == ",".join(primary_names)
            assert saved.selected_models == ",".join(all_names)
            assert int(saved.primary_correct_days) == primary_correct
            assert int(saved.selected_correct_days) == all_correct
            assert int(saved.baseline_correct_days) == baseline_correct
            assert int(saved.test_days) == total_days
            assert abs(float(saved.primary_accuracy) - primary_correct / total_days) < 2e-6
            assert abs(float(saved.baseline_accuracy) - baseline_correct / total_days) < 2e-6
            checked += 1
    return checked


def check_prospective_artifacts() -> int:
    data_dir = ROOT / "data" / "prospective-100"
    run_dir = ROOT / "runs" / "prospective-100"
    if not data_dir.exists() and not run_dir.exists():
        return 0
    assert data_dir.is_dir() and run_dir.is_dir(), "Prospective promotion is incomplete"
    manifest = json.loads((run_dir / "ARTIFACT_HASHES.json").read_text(encoding="utf-8"))
    listed = set(manifest["files"])
    actual = {f"data/{path.relative_to(data_dir).as_posix()}" for path in data_dir.rglob("*")
              if path.is_file()}
    actual.update(f"runs/{path.relative_to(run_dir).as_posix()}" for path in run_dir.rglob("*")
                  if path.is_file() and path.name != "ARTIFACT_HASHES.json")
    assert listed == actual, "Original prospective artifacts differ from the saved inventory"
    for relative, expected in manifest["files"].items():
        part, _, remainder = relative.partition("/")
        base = data_dir if part == "data" else run_dir if part == "runs" else None
        assert base is not None and remainder
        assert hashlib.sha256((base / remainder).read_bytes()).hexdigest() == expected
    start = json.loads((run_dir / "run_start.json").read_text(encoding="utf-8"))
    assert start["protocol_sha256"] == hashlib.sha256(
        (ROOT / "FUTURE_EVAL_PROTOCOL.md").read_bytes()).hexdigest()
    assert start["code_manifest_sha256"] == hashlib.sha256(
        (ROOT / "FUTURE_CODE_HASHES.json").read_bytes()).hexdigest()
    dates = start["selected_dates"]
    assert len(dates) == 100 and pd.Index(dates).is_monotonic_increasing
    assert len(set(dates)) == 100 and dates[0] > start["cutoff"]
    assert set(start["symbols"]) == {"GOOG", "AAPL", "TSLA"}
    totals = pd.read_csv(run_dir / "all_tickers.csv").set_index("ticker")
    intervals = pd.read_csv(run_dir / "paired_uncertainty.csv").set_index("ticker")
    assert set(totals.index) == set(intervals.index) == set(start["symbols"])
    for ticker in start["symbols"]:
        source = pd.read_csv(data_dir / f"{ticker}.csv")
        assert source.Date.tolist() == dates
        folder = run_dir / ticker.lower()
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(folder / "metrics.csv").set_index("model")
        chosen = summary["primary_selected_by_validation"]
        paired = json.loads((folder / "paired_uncertainty.json").read_text(encoding="utf-8"))
        top = totals.loc[ticker]
        interval = intervals.loc[ticker]
        assert summary["splits"]["test"]["targets"] == 100
        assert summary["splits"]["test"]["first_date"] == dates[0]
        assert summary["splits"]["test"]["last_date"] == dates[-1]
        assert top.selected_by_validation == paired["selected_by_validation"] == chosen
        assert int(top.selected_correct_days) == paired["selected_correct_days"] == int(
            metrics.loc[chosen].test_correct_days)
        assert int(top.baseline_correct_days) == paired["baseline_correct_days"] == int(
            metrics.loc["train-majority"].test_correct_days)
        assert int(top.test_days) == paired["test_days"] == 100
        assert abs(float(top.accuracy_gain) - paired["accuracy_gain"]) < 2e-6
        assert abs(float(interval.accuracy_gain) - paired["accuracy_gain"]) < 2e-6
        assert abs(float(top.gain_95pct_low) - paired["block_bootstrap_95pct_low"]) < 2e-6
        assert abs(float(top.gain_95pct_high) - paired["block_bootstrap_95pct_high"]) < 2e-6
    return len(start["symbols"])


def main() -> None:
    folders = report_folders()
    assert folders, "No saved direction reports found"
    pages = sum(check_fold(folder) for folder in folders)
    configurations, daily_rows = check_ablation()
    intervals = check_paired_uncertainty()
    rollups = check_period_rollups()
    prospective = check_prospective_artifacts()
    print(f"Verified {len(folders)} fold reports, {pages} candidate charts, "
          f"{configurations} ablation configurations, {daily_rows} daily ablation predictions, "
          f"{intervals} paired comparisons, {rollups} stock-period summaries, "
          f"{prospective} prospective reports")


if __name__ == "__main__":
    main()
