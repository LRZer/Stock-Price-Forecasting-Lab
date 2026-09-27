"""The future checkpoint stays behind a fixed cutoff and common-date gate."""

from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from stocklab import prospective_check as prospective
from stocklab.neural_direction import ROOT
from stocklab.neural_direction_data import load_direction_series, scale_for_fold
from stocklab.prospective_check import (CUTOFF, TEST_DAYS, TICKERS,
                                        check_frozen_sources, first_common_100,
                                        frozen_fold, load_pinned_2026_context)


class ProspectiveProtocolTests(unittest.TestCase):
    @staticmethod
    def synthetic_context(ticker: str) -> pd.DataFrame:
        history = pd.read_csv(ROOT / "data" / "raw" / f"{ticker}.csv")
        dates = pd.bdate_range("2026-01-02", periods=183).append(
            pd.DatetimeIndex([CUTOFF])).strftime("%Y-%m-%d")
        close = float(history.Close.iloc[-1]) + np.arange(184) * 0.1
        return pd.DataFrame({"Date": dates, "Open": close,
                             "High": close + 1, "Low": close - 1,
                             "Close": close, "Volume": 1_000_000})

    def test_complete_stage_can_finish_after_interrupted_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            data = root / "data" / "prospective-100"
            runs = root / "runs" / "prospective-100"
            staged_data, staged_runs = stage / "data", stage / "runs"
            staged_data.mkdir(parents=True)
            staged_runs.mkdir()
            files = {}
            names = [f"data/{ticker}.csv" for ticker in TICKERS]
            names.extend(("runs/run_start.json", "runs/all_tickers.csv",
                          "runs/paired_uncertainty.csv", "runs/ARTIFACT_HASHES.json"))
            for ticker in TICKERS:
                names.extend(f"runs/{ticker.lower()}/{name}" for name in
                             ("metrics.csv", "predictions.csv", "summary.json",
                              "paired_uncertainty.json"))
            for name in names:
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode("utf-8"))
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (stage / "READY.json").write_text(json.dumps({"files": files}), encoding="utf-8")
            data.parent.mkdir()
            runs.parent.mkdir()
            staged_data.rename(data)  # The first promotion completed, then the process stopped.
            with (mock.patch.object(prospective, "DATA_DIR", data),
                  mock.patch.object(prospective, "RUN_DIR", runs),
                  mock.patch.object(prospective, "STAGE_DIR", stage),
                  mock.patch.object(prospective, "check_frozen_sources")):
                prospective.finish_staged_run()
            self.assertTrue((runs / "all_tickers.csv").is_file())
            self.assertTrue((data / "GOOG.csv").is_file())
            self.assertFalse(stage.exists())

    def test_failed_generation_keeps_final_directories_absent(self) -> None:
        dates = pd.bdate_range("2026-09-28", periods=TEST_DAYS)
        first = {}
        for ticker in TICKERS:
            close = np.full(TEST_DAYS, self.synthetic_context(ticker).Close.iloc[-1])
            first[ticker] = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"),
                                          "Open": close, "High": close + 1,
                                          "Low": close - 1, "Close": close,
                                          "Volume": 1_000_000})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (mock.patch.object(prospective, "DATA_DIR", root / "data" / "prospective-100"),
                  mock.patch.object(prospective, "RUN_DIR", root / "runs" / "prospective-100"),
                  mock.patch.object(prospective, "STAGE_DIR", root / "stage"),
                  mock.patch.object(prospective, "check_frozen_sources"),
                  mock.patch.object(prospective, "load_pinned_2026_context",
                                    return_value={ticker: self.synthetic_context(ticker)
                                                  for ticker in TICKERS}),
                  mock.patch.object(prospective, "fit_classifier", side_effect=RuntimeError("simulated failure"))):
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    prospective.run_once(first)
            self.assertFalse((root / "data" / "prospective-100").exists())
            self.assertFalse((root / "runs" / "prospective-100").exists())
            self.assertTrue((root / "stage" / "data" / "GOOG.csv").is_file())
            self.assertFalse((root / "stage" / "READY.json").exists())

    def test_frozen_guard_rejects_changed_code_or_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for manifest_name, source_name in (("FUTURE_BASE_HASHES.json", "old.csv"),
                                               ("FUTURE_CODE_HASHES.json", "runner.py")):
                content = b"frozen\n"
                (root / source_name).write_bytes(content)
                (root / manifest_name).write_text(json.dumps({"files": {
                    source_name: hashlib.sha256(content).hexdigest()}}), encoding="utf-8")
            with mock.patch("stocklab.prospective_check.ROOT", root):
                check_frozen_sources()
                (root / "runner.py").write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "runner.py"):
                    check_frozen_sources()

    def test_new_days_gate_requires_100_matching_dates(self) -> None:
        dates = pd.bdate_range("2026-09-28", periods=TEST_DAYS).strftime("%Y-%m-%d")
        frame = pd.DataFrame({"Date": dates})
        self.assertIsNone(first_common_100({name: frame.iloc[:99] for name in TICKERS}))
        uneven = {name: frame.copy() for name in TICKERS}
        uneven["TSLA"] = uneven["TSLA"].copy()
        uneven["TSLA"].loc[0, "Date"] = "2026-09-29"
        with self.assertRaises(ValueError):
            first_common_100(uneven)

    def test_frozen_split_uses_no_new_target_for_training_or_validation(self) -> None:
        check_frozen_sources()
        dates = pd.bdate_range("2026-09-28", periods=TEST_DAYS)
        context = self.synthetic_context("GOOG")
        close = context.Close.iloc[-1] + np.cumsum(
            np.where(np.arange(TEST_DAYS) % 2, -0.5, 0.5))
        frame = pd.DataFrame({"Date": dates, "Open": close - 0.2,
                              "High": close + 0.5, "Low": close - 0.5,
                              "Close": close, "Volume": 1_000_000})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "future.csv"
            context_path = Path(temporary) / "context.csv"
            frame.to_csv(path, index=False)
            context.to_csv(context_path, index=False)
            series = load_direction_series(ROOT / "data" / "raw" / "GOOG.csv",
                                           context_path,
                                           path)
            fold = frozen_fold(series)
            data = scale_for_fold(series, fold)
        self.assertEqual(len(fold.validation), 150)
        self.assertEqual(len(fold.test), TEST_DAYS)
        self.assertEqual(data.dates[fold.validation[-1]], CUTOFF)
        self.assertGreater(data.dates[fold.test[0]], CUTOFF)
        self.assertLess(fold.train[-1], fold.validation[0])
        self.assertLess(fold.validation[-1], fold.test[0])

    def test_removed_context_is_fetched_only_if_hash_matches(self) -> None:
        frames = {ticker: self.synthetic_context(ticker) for ticker in TICKERS}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digests = {ticker: hashlib.sha256(
                frame.to_csv(index=False, float_format="%.6f").encode()).hexdigest()
                for ticker, frame in frames.items()}
            (root / "FUTURE_BASE_HASHES.json").write_text(
                json.dumps({"removed_2026_context_sha256": digests}), encoding="utf-8")
            with (mock.patch.object(prospective, "ROOT", root),
                  mock.patch.object(prospective, "download_one",
                                    side_effect=lambda ticker, *_: frames[ticker])):
                self.assertEqual(len(load_pinned_2026_context()), 3)
                frames["GOOG"] = frames["GOOG"].copy()
                frames["GOOG"].loc[0, "Close"] += 0.1
                with self.assertRaisesRegex(ValueError, "pinned hash"):
                    load_pinned_2026_context()


if __name__ == "__main__":
    unittest.main()
