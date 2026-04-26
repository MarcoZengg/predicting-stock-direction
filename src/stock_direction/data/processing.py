"""Load, clean, split, and save stock-direction datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stock_direction.config import (
    EXTERNAL_TICKERS,
    LABEL_THRESHOLD,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TEST_SIZE,
    TICKERS,
)
from stock_direction.features.build_features import build_feature_table, numeric_feature_columns

OHLCV_COLUMNS = ["Date", "Close", "High", "Low", "Open", "Volume"]


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Read raw OHLCV data from either normal CSV or yfinance multi-header CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    preview = pd.read_csv(path, nrows=1)
    if preview.columns[0] == "Price":
        df = pd.read_csv(path, skiprows=2)
        df.columns = OHLCV_COLUMNS[: len(df.columns)]
    else:
        df = pd.read_csv(path)
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})

    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[OHLCV_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=OHLCV_COLUMNS).sort_values("Date")
    df = df.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return df


def _raw_path_for_ticker(ticker: str, raw_dir: Path) -> Path:
    return raw_dir / f"{ticker.replace('^', '')}_historical.csv"


def _load_external_returns(raw_dir: Path) -> dict[str, pd.DataFrame]:
    external: dict[str, pd.DataFrame] = {}
    for name in EXTERNAL_TICKERS:
        path = _raw_path_for_ticker(name, raw_dir)
        if not path.exists():
            continue
        df = read_ohlcv_csv(path)[["Date", "Close"]].copy()
        lower = name.lower()
        df[f"{lower}_return_1d"] = df["Close"].pct_change()
        df[f"{lower}_return_5d"] = df["Close"].pct_change(5)
        df[f"{lower}_level"] = df["Close"]
        external[name] = df.drop(columns="Close")
    return external


def add_context_features(
    target: pd.DataFrame,
    ticker: str,
    base_tables: dict[str, pd.DataFrame],
    external_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge same-day cross-asset context available after market close."""
    df = target.copy()

    for ext in external_tables.values():
        df = df.merge(ext, on="Date", how="left")

    for other_ticker, other_df in base_tables.items():
        if other_ticker == ticker:
            continue
        other_return = other_df[["Date", "return"]].rename(
            columns={"return": f"{other_ticker.lower()}_return_1d"}
        )
        df = df.merge(other_return, on="Date", how="left")

    if ticker != "SPY" and "spy_return_1d" in df.columns:
        df["rel_return_vs_spy_1d"] = df["return"] - df["spy_return_1d"]
    if ticker == "SPY":
        for other in ["QQQ", "IWM"]:
            col = f"{other.lower()}_return_1d"
            if col in df.columns:
                df[f"{other.lower()}_minus_spy_return_1d"] = df[col] - df["return"]

    return df


def chronological_split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split without shuffling to avoid look-ahead bias."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def process_all_tickers(
    tickers: Iterable[str] = TICKERS,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    label_threshold: float = LABEL_THRESHOLD,
) -> pd.DataFrame:
    """Build processed datasets, chronological splits, and a data summary."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_tables: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        raw = read_ohlcv_csv(_raw_path_for_ticker(ticker, raw_dir))
        base_tables[ticker] = build_feature_table(raw, label_threshold=label_threshold)

    external_tables = _load_external_returns(raw_dir)
    summary_rows = []
    feature_manifest: dict[str, list[str]] = {}

    for ticker, base_df in base_tables.items():
        df = add_context_features(base_df, ticker, base_tables, external_tables)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        train, test = chronological_split(df)
        df.to_csv(processed_dir / f"{ticker}_processed.csv", index=False)
        train.to_csv(processed_dir / f"{ticker}_train.csv", index=False)
        test.to_csv(processed_dir / f"{ticker}_test.csv", index=False)

        features = numeric_feature_columns(df)
        feature_manifest[ticker] = features
        summary_rows.append(
            {
                "ticker": ticker,
                "rows": len(df),
                "train_rows": len(train),
                "test_rows": len(test),
                "start_date": df["Date"].min().date().isoformat(),
                "end_date": df["Date"].max().date().isoformat(),
                "positive_label_rate": round(float(df["label"].mean()), 4),
                "feature_count": len(features),
            }
        )

    with (RESULTS_DIR / "feature_columns.json").open("w", encoding="utf-8") as fp:
        json.dump(feature_manifest, fp, indent=2)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "data_summary.csv", index=False)
    return summary
