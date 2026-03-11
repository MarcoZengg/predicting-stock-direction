# Instructions

## Getting Started

### How to Download Data

1. **Install dependencies**

   ```bash
   pip install yfinance pandas
   ```

2. **Create the data directory** (if it doesn't exist)

   ```bash
   mkdir -p data/raw
   ```

3. **Run the fetch script** from the project root

   ```bash
   python scripts/fetch_data.py
   ```

4. **Verify output**

   The script downloads historical OHLCV data for QQQ, SPY, and IWM (2010–2025) and saves one CSV per ticker to `data/raw/`:

   - `data/raw/QQQ_historical.csv`
   - `data/raw/SPY_historical.csv`
   - `data/raw/IWM_historical.csv`

   Each file contains Open, High, Low, Close, and Volume for approximately 4,000 trading days.
