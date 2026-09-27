"""Download a small, fixed educational daily-price dataset."""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKERS = ("GOOG", "AAPL", "TSLA")
COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
SOURCE_BASE = "https://pubmarks.github.io/datasets/stocks"


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{SOURCE_BASE}/{ticker}/ohlcv.csv"
    with urlopen(url, timeout=30) as response:
        frame = pd.read_csv(io.BytesIO(response.read()))
    if frame.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    frame.columns = [str(column).title() for column in frame.columns]
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{ticker}: missing columns {sorted(missing)}")
    frame = frame.loc[:, COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.loc[(frame["Date"] >= pd.Timestamp(start))
                      & (frame["Date"] < pd.Timestamp(end))]
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any():
        raise RuntimeError(f"{ticker}: duplicate dates")
    if frame.isna().any().any():
        raise RuntimeError(f"{ticker}: missing values")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise RuntimeError(f"{ticker}: non-positive prices")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--min-rows", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "Pubmarks public daily OHLCV CSV",
        "source_url": SOURCE_BASE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_inclusive": args.start,
        "end_exclusive": args.end,
        "interval": "1d",
        "adjustment_policy": "Upstream yfinance auto_adjust=False; uses Close, not Adj Close; Yahoo split/revision handling may change",
        "upstream_fetch_code": "https://github.com/Pubmarks/datasets/blob/main/scripts/python/yf/lib/ohlcv_fetch.py",
        "upstream_daily_workflow": "https://github.com/Pubmarks/datasets/blob/main/.github/workflows/stocks-ohlcv-daily.yaml",
        "tickers": {},
    }
    for ticker in args.tickers:
        ticker = ticker.upper().strip()
        if not ticker.isalnum():
            raise ValueError(f"Invalid ticker: {ticker!r}")
        frame = download_one(ticker, args.start, args.end)
        if len(frame) < args.min_rows:
            raise RuntimeError(f"{ticker}: only {len(frame)} rows")
        path = args.output_dir / f"{ticker}.csv"
        frame.to_csv(path, index=False, float_format="%.6f")
        metadata["tickers"][ticker] = {
            "rows": len(frame),
            "first_date": frame["Date"].iloc[0],
            "last_date": frame["Date"].iloc[-1],
            "file": path.name,
            "url": f"{SOURCE_BASE}/{ticker}/ohlcv.csv",
        }
        print(f"{ticker}: {len(frame)} rows, {frame['Date'].iloc[0]} to {frame['Date'].iloc[-1]}")

    (args.output_dir / "SOURCES.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
