"""Download raw market data from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yfinance as yf

from stock_direction.config import END_DATE, RAW_DATA_DIR, START_DATE


def fetch_ohlcv(
    tickers: Iterable[str],
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DATA_DIR,
) -> list[Path]:
    """Download OHLCV data and save one CSV per ticker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")
        path = output_dir / f"{ticker.replace('^', '')}_historical.csv"
        data.to_csv(path)
        saved_paths.append(path)

    return saved_paths
