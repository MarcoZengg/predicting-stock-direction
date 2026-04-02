import os
import yfinance as yf

# Define tickers and date range
ETF_TICKERS = ["QQQ", "SPY", "IWM"]
EXTERNAL_SYMBOLS = ["^VIX", "^TNX", "TLT", "HYG", "GLD", "USO"]
START = "2010-01-01"
END = "2025-12-31"

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def symbol_to_filename(symbol: str) -> str:
    # Keep filenames simple and stable for downstream loading.
    return symbol.replace("^", "") + "_historical.csv"


# Save to CSV (one file per ETF)
for ticker in ETF_TICKERS:
    df = yf.download(ticker, start=START, end=END, progress=True)
    df.to_csv(os.path.join(RAW_DIR, f"{ticker}_historical.csv"))

# Save to CSV (one file per external risk/macro proxy)
for symbol in EXTERNAL_SYMBOLS:
    df = yf.download(symbol, start=START, end=END, progress=True)
    df.to_csv(os.path.join(RAW_DIR, symbol_to_filename(symbol)))

