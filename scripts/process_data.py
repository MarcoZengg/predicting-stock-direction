import os
import pandas as pd

# ====== Get project root ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TICKERS = ["SPY", "QQQ", "IWM"]

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

for TICKER in TICKERS:

    print(f"\nProcessing {TICKER}...")

    RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", f"{TICKER}_historical.csv")

    # ====== Load ======
    df = pd.read_csv(RAW_PATH, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # ====== Feature Engineering ======
    df["return"] = df["Close"].pct_change()
    df["label"] = (df["return"].shift(-1) > 0).astype(int)

    df["lag_return_1"] = df["return"]
    df["rolling_mean_5"] = df["return"].rolling(5).mean()
    df["rolling_std_5"] = df["return"].rolling(5).std()

    df = df.dropna().reset_index(drop=True)

    # ====== Save full dataset ======
    processed_path = os.path.join(PROCESSED_DIR, f"{TICKER}_processed.csv")
    df.to_csv(processed_path, index=False)

    print("Saved:", processed_path)
    print("Label distribution:")
    print(df["label"].value_counts(normalize=True))

    # ====== Time Split ======
    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    train.to_csv(os.path.join(PROCESSED_DIR, f"{TICKER}_train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, f"{TICKER}_test.csv"), index=False)

    print("Train size:", len(train))
    print("Test size:", len(test))