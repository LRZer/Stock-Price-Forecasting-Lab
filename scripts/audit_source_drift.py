"""Read only: compare frozen historical OHLCV against the current original source."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.download_data import download_one
from stocklab.neural_direction import ROOT
from stocklab.prospective_check import load_pinned_2026_context


TICKERS = ("GOOG", "AAPL", "TSLA")
COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def main() -> None:
    any_drift = False
    context = load_pinned_2026_context()
    print("Frozen history vs current source through 2026-09-25 (read only)")
    for ticker in TICKERS:
        frozen = pd.concat((pd.read_csv(ROOT / "data" / "raw" / f"{ticker}.csv"),
                            context[ticker]),
                           ignore_index=True).set_index("Date")
        current = download_one(ticker, "2022-01-01", "2026-09-26").set_index("Date")
        missing = frozen.index.difference(current.index)
        extra = current.index.difference(frozen.index)
        common = frozen.index.intersection(current.index)
        old_values = frozen.loc[common, list(COLUMNS)].to_numpy(dtype=np.float64)
        new_values = current.loc[common, list(COLUMNS)].to_numpy(dtype=np.float64)
        changed = ~np.isclose(old_values, new_values, rtol=0, atol=1e-6)
        count = int(changed.sum())
        any_drift |= bool(len(missing) or len(extra) or count)
        print(f"{ticker}: frozen {len(frozen)} rows, source {len(current)} rows; "
              f"missing {len(missing)}, extra {len(extra)}, changed cells {count}")
        if count:
            for row, col in list(zip(*np.where(changed)))[:5]:
                print(f"  {common[row]} {COLUMNS[col]}: frozen={old_values[row, col]:.6f}, "
                      f"source={new_values[row, col]:.6f}")
    if any_drift:
        raise SystemExit("Source history differs from the frozen files; review before future evaluation")
    print("No historical source drift detected; frozen files were not changed")


if __name__ == "__main__":
    main()
