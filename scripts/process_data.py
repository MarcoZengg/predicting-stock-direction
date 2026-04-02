import os
import pandas as pd

# get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TICKERS = ["SPY", "QQQ", "IWM"]

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    # yfinance CSV after skiprows=2 -> Date,Close,High,Low,Open,Volume
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def build_vix_features() -> pd.DataFrame:
    vix_raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "VIX_historical.csv")
    vix_df = load_price_csv(vix_raw_path)
    vix_df["vix_level"] = vix_df["Close"]
    vix_df["vix_change_1d"] = vix_df["vix_level"].diff()
    vix_df["vix_return_1d"] = vix_df["vix_level"].pct_change()
    vix_df["vix_return_5d"] = vix_df["vix_level"].pct_change(5)
    vix_df["vix_rolling_mean_5"] = vix_df["vix_return_1d"].rolling(5).mean()
    vix_df["vix_rolling_std_5"] = vix_df["vix_return_1d"].rolling(5).std()
    vix_df["vix_rolling_mean_10"] = vix_df["vix_return_1d"].rolling(10).mean()
    vix_df["vix_rolling_std_10"] = vix_df["vix_return_1d"].rolling(10).std()
    vix_level_mean_20 = vix_df["vix_level"].rolling(20).mean()
    vix_level_std_20 = vix_df["vix_level"].rolling(20).std()
    vix_df["vix_zscore_20"] = (vix_df["vix_level"] - vix_level_mean_20) / vix_level_std_20
    cols = [
        "Date",
        "vix_level",
        "vix_change_1d",
        "vix_return_1d",
        "vix_return_5d",
        "vix_rolling_mean_5",
        "vix_rolling_std_5",
        "vix_rolling_mean_10",
        "vix_rolling_std_10",
        "vix_zscore_20",
    ]
    return vix_df[cols]


def build_proxy_features(symbol_name: str, prefix: str) -> pd.DataFrame:
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", f"{symbol_name}_historical.csv")
    df = load_price_csv(raw_path)
    df[f"{prefix}_level"] = df["Close"]
    df[f"{prefix}_change_1d"] = df["Close"].diff()
    df[f"{prefix}_return_1d"] = df["Close"].pct_change()
    df[f"{prefix}_return_5d"] = df["Close"].pct_change(5)
    return df[["Date", f"{prefix}_level", f"{prefix}_change_1d", f"{prefix}_return_1d", f"{prefix}_return_5d"]]


# External feature blocks (built once, merged into each ETF)
vix_features = build_vix_features()
tnx_features = build_proxy_features("TNX", "tnx")
tlt_features = build_proxy_features("TLT", "tlt")
hyg_features = build_proxy_features("HYG", "hyg")
gld_features = build_proxy_features("GLD", "gld")
uso_features = build_proxy_features("USO", "uso")

# ETF relative-strength helper data
ret_df = None
mom5_df = None
for t in TICKERS:
    cur = load_price_csv(os.path.join(PROJECT_ROOT, "data", "raw", f"{t}_historical.csv"))[["Date", "Close"]].copy()
    cur[f"{t.lower()}_return_1d"] = cur["Close"].pct_change()
    cur[f"{t.lower()}_mom5"] = cur["Close"] / cur["Close"].shift(5) - 1
    cur = cur[["Date", f"{t.lower()}_return_1d", f"{t.lower()}_mom5"]]
    if ret_df is None:
        ret_df = cur[["Date", f"{t.lower()}_return_1d"]]
        mom5_df = cur[["Date", f"{t.lower()}_mom5"]]
    else:
        ret_df = ret_df.merge(cur[["Date", f"{t.lower()}_return_1d"]], on="Date", how="outer")
        mom5_df = mom5_df.merge(cur[["Date", f"{t.lower()}_mom5"]], on="Date", how="outer")

for TICKER in TICKERS:
    print(f"\nProcessing {TICKER}...")

    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", f"{TICKER}_historical.csv")
    df = load_price_csv(raw_path)

    # ====== Base Feature Engineering ======
    df["return"] = df["Close"].pct_change()
    df["label"] = (df["return"].shift(-1) > 0).astype(int)

    # Existing features
    df["lag_return_1"] = df["return"]
    df["rolling_mean_5"] = df["return"].rolling(5).mean()
    df["rolling_std_5"] = df["return"].rolling(5).std()
    df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_5"] = df["Volume"].rolling(5).mean()

    # Additional OHLC-derived features
    prev_close = df["Close"].shift(1)
    df["gap_open"] = (df["Open"] - prev_close) / prev_close
    df["hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["co_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["lag_return_2"] = df["return"].shift(2)
    df["lag_return_3"] = df["return"].shift(3)
    df["rolling_std_20"] = df["return"].rolling(20).std()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["close_to_ma20"] = (df["Close"] - df["ma_20"]) / df["ma_20"]
    df["ma5_to_ma20"] = df["ma_5"] / df["ma_20"] - 1

    # Merge external blocks
    df = df.merge(vix_features, on="Date", how="left")
    df = df.merge(tnx_features, on="Date", how="left")
    df = df.merge(tlt_features, on="Date", how="left")
    df = df.merge(hyg_features, on="Date", how="left")
    df = df.merge(gld_features, on="Date", how="left")
    df = df.merge(uso_features, on="Date", how="left")
    df = df.merge(ret_df, on="Date", how="left")
    df = df.merge(mom5_df, on="Date", how="left")

    # Relative-strength features
    if TICKER == "QQQ":
        df["rel_return_vs_spy_1d"] = df["qqq_return_1d"] - df["spy_return_1d"]
        df["rel_momentum_vs_spy_5d"] = df["qqq_mom5"] - df["spy_mom5"]
    elif TICKER == "IWM":
        df["rel_return_vs_spy_1d"] = df["iwm_return_1d"] - df["spy_return_1d"]
        df["rel_momentum_vs_spy_5d"] = df["iwm_mom5"] - df["spy_mom5"]
    else:
        df["rel_return_vs_spy_1d"] = 0.0
        df["rel_momentum_vs_spy_5d"] = 0.0
    df["qqq_minus_spy_return_1d"] = df["qqq_return_1d"] - df["spy_return_1d"]
    df["iwm_minus_spy_return_1d"] = df["iwm_return_1d"] - df["spy_return_1d"]

    # Calendar / regime features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    vol_med_60 = df["volatility_10"].rolling(60).median()
    df["high_vol_regime"] = (df["volatility_10"] > vol_med_60).astype(int)
    df["risk_off_regime"] = ((df["vix_change_1d"] > 0) & (df["tlt_return_1d"] > 0)).astype(int)

    # Keep clean rows only after all features are present
    df = df.dropna().reset_index(drop=True)

    # Save full dataset
    processed_path = os.path.join(PROCESSED_DIR, f"{TICKER}_processed.csv")
    df.to_csv(processed_path, index=False)
    print("Saved:", processed_path)
    print("Label distribution:")
    print(df["label"].value_counts(normalize=True))

    # Time split
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    train.to_csv(os.path.join(PROCESSED_DIR, f"{TICKER}_train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, f"{TICKER}_test.csv"), index=False)
    print("Train size:", len(train))
    print("Test size:", len(test))
