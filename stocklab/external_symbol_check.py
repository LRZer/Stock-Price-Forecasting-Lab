"""Run the frozen six-candidate check on two previously untested symbols once."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .neural_direction import PRIMARY_NAMES, ROOT, run_one


SYMBOLS = ("ACN", "RMD")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs" / "external-symbol-check")
    args = parser.parse_args()
    if (args.output_dir / "all_tickers.csv").exists():
        raise FileExistsError("External check already exists; keep the original untouched")
    rows = []
    for ticker in SYMBOLS:
        source_path = ROOT / "data" / "external-check" / f"{ticker}.csv"
        rows.append(run_one(ticker, mode="fixed-2025",
                            output_dir=args.output_dir / ticker.lower(),
                            device=args.device, epochs=20, seed=42,
                            names=PRIMARY_NAMES, source_path=source_path))
    pd.DataFrame(rows).to_csv(args.output_dir / "all_tickers.csv", index=False,
                              float_format="%.6f")
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
