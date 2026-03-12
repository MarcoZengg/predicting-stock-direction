# This is for Data Acquisition phase

import yfinance as yf

# Define tickers and date range
TICKERS = ["QQQ", "SPY", "IWM"]
START = "2010-01-01"
END = "2025-12-31"


# Save to CSV (one file per ticker)
for ticker in TICKERS:
    df = yf.download(ticker, start=START, end=END, progress=True)
    df.to_csv(f"data/raw/{ticker}_historical.csv")
