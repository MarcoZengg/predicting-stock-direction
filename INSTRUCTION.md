# Instructions

## Getting Started

From the repository root, run:

```bash
make install
make test
```

### How to Download Data

1. **Install dependencies**

   ```bash
   make install
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

### How to Process Data

Run this **after** you have raw data in `data/raw/` (see *How to Download Data* above).

1. **Install dependencies** (if not already installed)

   ```bash
   make install
   ```

2. **Run the process script** from the project root

   ```bash
   python scripts/process_data.py
   ```

3. **Verify output**

   The script reads each `data/raw/{TICKER}_historical.csv`, cleans and standardizes the data, adds features (returns, momentum, moving averages, volatility, volume features), and builds the next-day direction label. It then saves:

   - **Full processed dataset** (per ticker): `data/processed/QQQ_processed.csv`, `SPY_processed.csv`, `IWM_processed.csv`
   - **Train/test split** (80% / 20% by time): `data/processed/{TICKER}_train.csv` and `data/processed/{TICKER}_test.csv`

   Rows with missing values (e.g. from rolling windows) are dropped. Use the `*_train.csv` and `*_test.csv` files for model training and evaluation.
