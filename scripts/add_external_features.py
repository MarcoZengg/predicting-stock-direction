import os
import pandas as pd
import yfinance as yf

# get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TICKERS = ["SPY", "QQQ", "IWM"]

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ENHANCED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_enhanced")
os.makedirs(ENHANCED_DIR, exist_ok=True)

# external market features
EXTERNAL_TICKERS = {
    "VIX": "^VIX",
    "TLT": "TLT",
    "GLD": "GLD"
}


def download_close_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # handle possible multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # keep only Date and Close
    if "Close" not in df.columns:
        raise ValueError(f"Close column not found for {ticker}")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# first load all processed ETF data
processed_data = {}

for ticker in TICKERS:
    processed_path = os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv")

    df = pd.read_csv(processed_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    processed_data[ticker] = df

# use global date range for external downloads
all_dates = pd.concat([processed_data[t]["Date"] for t in TICKERS])
start_date = all_dates.min().strftime("%Y-%m-%d")
end_date = (all_dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print("Downloading external features...")
print("Date range:", start_date, "to", end_date)

# download external market data
external_features = {}

for name, ticker_symbol in EXTERNAL_TICKERS.items():
    ext_df = download_close_data(ticker_symbol, start_date, end_date)

    # use returns instead of close price
    ext_df[f"{name.lower()}_return"] = ext_df["Close"].pct_change()

    # short rolling trends
    ext_df[f"{name.lower()}_return_ma_3"] = ext_df[f"{name.lower()}_return"].rolling(3).mean()
    ext_df[f"{name.lower()}_return_ma_5"] = ext_df[f"{name.lower()}_return"].rolling(5).mean()

    ext_df = ext_df[
        [
            "Date",
            f"{name.lower()}_return",
            f"{name.lower()}_return_ma_3",
            f"{name.lower()}_return_ma_5"
        ]
    ]

    external_features[name] = ext_df

# create ETF cross-return features
cross_return_features = {}

for ticker in TICKERS:
    df = processed_data[ticker][["Date", "return"]].copy()
    df = df.rename(columns={"return": f"{ticker.lower()}_return"})
    cross_return_features[ticker] = df

# merge features into each target ticker dataset
for target_ticker in TICKERS:

    print(f"\nAdding external features for {target_ticker}...")

    df = processed_data[target_ticker].copy()

    # merge VIX / TLT / GLD features
    for name in external_features:
        df = df.merge(external_features[name], on="Date", how="left")

    # merge returns from the other two ETFs
    for other_ticker in TICKERS:
        if other_ticker != target_ticker:
            df = df.merge(cross_return_features[other_ticker], on="Date", how="left")

    # shift external features by 1 day
    external_cols = [
        "vix_return",
        "vix_return_ma_3",
        "vix_return_ma_5",
        "tlt_return",
        "tlt_return_ma_3",
        "tlt_return_ma_5",
        "gld_return",
        "gld_return_ma_3",
        "gld_return_ma_5"
    ]

    for other_ticker in TICKERS:
        if other_ticker != target_ticker:
            external_cols.append(f"{other_ticker.lower()}_return")

    for col in external_cols:
        df[col] = df[col].shift(1)

    # drop rows with missing values after merge + shift
    df = df.dropna().reset_index(drop=True)

    # save full enhanced dataset
    enhanced_path = os.path.join(ENHANCED_DIR, f"{target_ticker}_processed_enhanced.csv")
    df.to_csv(enhanced_path, index=False)

    print("Saved:", enhanced_path)
    print("Columns added:")
    print(external_cols)

    # time split again
    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    train.to_csv(os.path.join(ENHANCED_DIR, f"{target_ticker}_train_enhanced.csv"), index=False)
    test.to_csv(os.path.join(ENHANCED_DIR, f"{target_ticker}_test_enhanced.csv"), index=False)

    print("Train size:", len(train))
    print("Test size:", len(test))
    print("Label distribution:")
    print(df["label"].value_counts(normalize=True))