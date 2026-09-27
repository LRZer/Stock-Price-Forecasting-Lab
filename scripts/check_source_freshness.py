"""Read only: show how recently the frozen source updated each stock."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.download_data import SOURCE_BASE, download_one
from stocklab.prospective_check import CUTOFF, TICKERS


def main() -> None:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    print(f"Source: {SOURCE_BASE} (checked {today}; no training or files saved)")
    print("Ticker  Latest row  Calendar age  New rows after cutoff  Status")
    for ticker in TICKERS:
        frame = download_one(ticker, "2026-01-01", "2100-01-01")
        if frame.empty:
            raise RuntimeError(f"{ticker}: source has no rows since 2026-01-01")
        latest = frame.Date.iloc[-1]
        age = (today - datetime.fromisoformat(latest).date()).days
        new_rows = int(frame.Date.gt(CUTOFF).sum())
        status = "CHECK SOURCE" if age < 0 or age > 14 else "current enough"
        print(f"{ticker:<7} {latest}  {age:>4} days      {new_rows:>4}                  {status}")


if __name__ == "__main__":
    main()
