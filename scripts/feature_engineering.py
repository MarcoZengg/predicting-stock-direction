import os
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TICKERS = ["SPY", "QQQ", "IWM"]

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def add_technical_indicators(df):
    """
    Add comprehensive technical indicators
    """
    # ===== Momentum Indicators =====
    # RSI (Relative Strength Index)
    df['rsi_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14, 
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # ===== Trend Indicators =====
    # ADX (Average Directional Index)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # Moving averages with different windows
    for window in [20, 50, 200]:
        df[f'ma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}'] - 1  # Distance from MA
    
    # ===== Volatility Indicators =====
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']  # Normalized width
    df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])  # Position in bands
    
    # ATR (Average True Range) - normalized by price
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['atr'] = atr.average_true_range()
    df['atr_pct'] = df['atr'] / df['Close'] * 100  # ATR as percentage
    
    # Historical volatility (annualized)
    for window in [10, 30, 60]:
        df[f'volatility_{window}'] = df['return'].rolling(window).std() * np.sqrt(252)  # Annualized
    
    # ===== Volume Indicators =====
    # OBV (On-Balance Volume)
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_ma_5'] = df['obv'].rolling(5).mean()
    df['obv_ratio'] = df['obv'] / df['obv_ma_5']
    
    # Volume Price Trend
    df['vpt'] = (df['Volume'] * df['return']).cumsum()
    
    # Money Flow Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(df['return'] > 0, 0).rolling(14).sum()
    negative_flow = money_flow.where(df['return'] < 0, 0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow
    df['mfi'] = 100 - (100 / (1 + money_ratio))
    
    # ===== Price Action Features =====
    # Candle patterns
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['body_ratio'] = df['body'] / (df['High'] - df['Low'])
    
    # Gap features
    df['gap'] = df['Open'] / df['Close'].shift(1) - 1
    df['gap_direction'] = np.sign(df['gap'])
    
    # Price position within day's range
    df['day_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    return df

def add_market_regime_features(df):
    """
    Add features that capture market regime/state
    Using NUMERIC encodings instead of strings
    """
    # Trend strength
    df['trend_strength'] = abs(df['ma_50'] - df['ma_200']) / df['Close']
    
    # Volatility regime (numeric)
    vol_bins = [0, 0.1, 0.2, 0.3, 1]
    vol_labels = [0, 1, 2, 3]  # Numeric: 0=very_low, 1=low, 2=medium, 3=high
    df['vol_regime'] = pd.cut(
        df['volatility_30'], 
        bins=vol_bins, 
        labels=vol_labels
    ).astype(float)
    
    # Market phase detection (numeric)
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    
    conditions = [
        (df['Close'] > sma_50) & (sma_50 > sma_200),  # Bull market
        (df['Close'] < sma_50) & (sma_50 < sma_200),  # Bear market
        (df['Close'] > sma_50) & (sma_50 < sma_200),  # Recovery
        (df['Close'] < sma_50) & (sma_50 > sma_200)   # Correction
    ]
    phase_values = [4, 0, 3, 1]  # Numeric: 0=bear, 1=correction, 3=recovery, 4=bull
    df['market_phase'] = np.select(conditions, phase_values, default=2)  # 2=neutral
    
    return df

def add_cross_etf_features(df, ticker, all_data):
    """
    Add features comparing with other ETFs
    """
    if ticker != 'SPY':
        # Compare with SPY (market benchmark)
        spy_data = all_data['SPY']
        aligned_data = spy_data.loc[df.index]  # Align dates
        df['relative_strength'] = df['Close'] / aligned_data['Close']
        df['relative_volume'] = df['Volume'] / aligned_data['Volume']
        
    if ticker == 'QQQ':
        # Tech vs market
        df['qqq_spy_ratio'] = df['Close'] / all_data['SPY'].loc[df.index]['Close']
    elif ticker == 'IWM':
        # Small cap vs market
        df['iwm_spy_ratio'] = df['Close'] / all_data['SPY'].loc[df.index]['Close']
    
    return df

def select_best_features(df):
    """
    Feature selection based on correlation with target
    Only use numeric columns for correlation
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation with target
    correlations = numeric_df.corr()['label'].drop('label').abs().sort_values(ascending=False)
    
    print("\nTop 10 features by correlation with target:")
    print(correlations.head(10))
    
    # Return the feature names
    top_features = correlations.head(20).index.tolist()
    
    return top_features

def process_ticker(ticker, all_data=None):
    """
    Process a single ticker with enhanced features
    """
    print(f"\n{'='*60}")
    print(f"Processing {ticker}...")
    print(f"{'='*60}")
    
    RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", f"{ticker}_historical.csv")
    
    # ====== Load ======
    df = pd.read_csv(RAW_PATH, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    
    # ====== Basic Features ======
    df["return"] = df["Close"].pct_change()
    df["label"] = (df["return"].shift(-1) > 0).astype(int)
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'lag_return_{lag}'] = df['return'].shift(lag)
    
    # ====== Advanced Technical Indicators ======
    df = add_technical_indicators(df)
    
    # ====== Market Regime Features ======
    df = add_market_regime_features(df)
    
    # ====== Cross-ETF Features (if data available) ======
    if all_data is not None:
        df = add_cross_etf_features(df, ticker, all_data)
    
    # ====== Drop NaN rows from feature creation ======
    df = df.dropna()
    
    # ====== Feature Selection ======
    feature_cols = select_best_features(df)
    
    # ====== Save processed data ======
    processed_path = os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv")
    df.to_csv(processed_path)
    
    print(f"\n✅ Saved: {processed_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Label distribution: {df['label'].mean():.3f}")
    print(f"   Numeric features: {len(feature_cols)}")
    
    # ====== Chronological split ======
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    train.to_csv(os.path.join(PROCESSED_DIR, f"{ticker}_train.csv"))
    test.to_csv(os.path.join(PROCESSED_DIR, f"{ticker}_test.csv"))
    
    print(f"   Train: {len(train)} samples ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"   Test: {len(test)} samples ({test.index[0].date()} to {test.index[-1].date()})")
    
    return df, feature_cols

def main():
    """
    Main execution
    """
    print("=" * 60)
    print("CS 506 - FEATURE ENGINEERING PHASE (WEEK 4)")
    print("=" * 60)
    
    # First pass: load all data for cross-ETF features
    all_data = {}
    for ticker in TICKERS:
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", f"{ticker}_historical.csv")
        df = pd.read_csv(raw_path, skiprows=2)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        all_data[ticker] = df.set_index("Date")[['Close', 'Volume']]  # Store more than just Close
    
    # Process each ticker
    all_features = {}
    for ticker in TICKERS:
        df, features = process_ticker(ticker, all_data)
        all_features[ticker] = features
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    
    for ticker in TICKERS:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv"), index_col=0, parse_dates=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_count = len([col for col in numeric_cols if col != 'label'])
        print(f"\n{ticker}:")
        print(f"  Total numeric features: {feature_count}")
        print(f"  Samples: {len(df)}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    print("\n✅ Feature engineering complete!")

if __name__ == "__main__":
    # Install ta library if not present
    try:
        import ta
    except ImportError:
        print("Installing ta library...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
    
    main()