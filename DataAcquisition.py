
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ETFDataAcquisition:
    """
    Data acquisition class for downloading ETF historical data from Yahoo Finance.
    Designed for CS 506 project: QQQ, SPY, IWM from 2010-2026.
    """
    
    def __init__(self, tickers=['QQQ', 'SPY', 'IWM'], start_date='2010-01-01', end_date='2026-03-06'):
        """
        Initialize the data acquisition module.
        
        Args:
            tickers (list): List of ETF ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = {}
        self.combined_data = None
        
    def download_data(self, use_cache=True, max_retries=3):
        """
        Download historical data for all tickers with retry logic.
        
        Args:
            use_cache (bool): Whether to use cached session
            max_retries (int): Number of retry attempts on failure
            
        Returns:
            dict: Dictionary of DataFrames for each ticker
        """
        logging.info(f"Downloading data for {self.tickers} from {self.start_date} to {self.end_date}")
        
        # Setup session with caching to avoid rate limiting
        if use_cache:
            try:
                import requests_cache
                session = requests_cache.CachedSession('yfinance.cache')
                session.headers['User-agent'] = 'CS506-Project/1.0'
                logging.info("Using cached session for requests")
            except ImportError:
                logging.warning("requests_cache not installed. Install with: pip install requests_cache")
                session = None
        else:
            session = None
        
        for ticker in self.tickers:
            for attempt in range(max_retries):
                try:
                    logging.info(f"Downloading {ticker} (attempt {attempt + 1}/{max_retries})")
                    
                    # Create Ticker object with optional session
                    if session:
                        ticker_obj = yf.Ticker(ticker, session=session)
                    else:
                        ticker_obj = yf.Ticker(ticker)
                    
                    # Download historical data
                    data = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval='1d',
                        auto_adjust=True,  # Adjust for splits/dividends
                        actions=True  # Include dividends and splits
                    )
                    
                    if data.empty:
                        raise ValueError(f"No data returned for {ticker}")
                    
                    # Add ticker column for identification when combining
                    data['Ticker'] = ticker
                    
                    self.raw_data[ticker] = data
                    logging.info(f"Successfully downloaded {ticker}: {len(data)} trading days")
                    
                    # Add delay between requests to be respectful to the API
                    time.sleep(1)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logging.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Failed to download {ticker} after {max_retries} attempts")
                        self.raw_data[ticker] = None
        
        return self.raw_data
    
    def verify_data_completeness(self):
        """
        Verify that downloaded data meets project requirements.
        
        Returns:
            dict: Summary statistics for each ticker
        """
        verification_results = {}
        
        for ticker, data in self.raw_data.items():
            if data is None or data.empty:
                verification_results[ticker] = {
                    'status': 'FAILED',
                    'error': 'No data available'
                }
                continue
            
            # Calculate expected trading days (approx 252 trading days per year)
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            years = (end - start).days / 365.25
            expected_days = int(years * 252)  # Approximate trading days
            
            actual_days = len(data)
            completeness_pct = (actual_days / expected_days) * 100
            
            # Check for missing values
            missing_values = data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum()
            
            # Check date range coverage
            date_range_coverage = {
                'start_actual': data.index.min().strftime('%Y-%m-%d'),
                'end_actual': data.index.max().strftime('%Y-%m-%d'),
                'expected_start': self.start_date,
                'expected_end': self.end_date
            }
            
            verification_results[ticker] = {
                'status': 'PASS' if completeness_pct > 95 else 'WARNING',
                'trading_days': actual_days,
                'expected_days': expected_days,
                'completeness_pct': round(completeness_pct, 2),
                'missing_values': missing_values.to_dict(),
                'date_range': date_range_coverage,
                'first_date': data.index.min().strftime('%Y-%m-%d'),
                'last_date': data.index.max().strftime('%Y-%m-%d')
            }
            
            logging.info(f"{ticker} verification: {verification_results[ticker]['status']} "
                        f"({verification_results[ticker]['completeness_pct']}% complete)")
        
        return verification_results
    
    def combine_datasets(self):
        """
        Combine all ETF data into a single multi-level DataFrame.
        
        Returns:
            pd.DataFrame: Combined dataset with multi-level columns
        """
        if not self.raw_data:
            logging.error("No data to combine. Run download_data() first.")
            return None
        
        # Method 1: Multi-level columns (ticker, OHLCV)
        combined = pd.concat(self.raw_data, axis=1)
        
        # Method 2: Long format with Ticker column (alternative)
        long_format = pd.concat([data for data in self.raw_data.values() if data is not None])
        
        self.combined_data = {
            'multi_level': combined,
            'long_format': long_format
        }
        
        logging.info(f"Combined dataset created: {combined.shape[0]} rows, {combined.shape[1]} columns")
        return self.combined_data
    
    def save_data(self, base_filename='etf_data'):
        """
        Save downloaded data to CSV files with metadata.
        
        Args:
            base_filename (str): Base name for output files
        """
        if not self.combined_data:
            logging.error("No combined data to save. Run combine_datasets() first.")
            return
        
        # Save multi-level format
        self.combined_data['multi_level'].to_csv(f'{base_filename}_multilevel.csv')
        
        # Save long format
        self.combined_data['long_format'].to_csv(f'{base_filename}_long.csv')
        
        # Save metadata
        metadata = {
            'tickers': self.tickers,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trading_days': len(self.combined_data['long_format']),
            'verification': self.verify_data_completeness()
        }
        
        with open(f'{base_filename}_metadata.txt', 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Data saved to {base_filename}_*.csv and metadata file")
    
    def get_data_summary(self):
        """
        Generate a comprehensive summary of the downloaded data.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if not self.combined_data:
            logging.error("No data to summarize")
            return None
        
        long_data = self.combined_data['long_format']
        
        summary = long_data.groupby('Ticker').agg({
            'Open': ['count', 'min', 'max', 'mean'],
            'Close': ['min', 'max', 'mean'],
            'Volume': ['min', 'max', 'mean'],
            'Dividends': ['sum'],
            'Stock Splits': ['sum']
        }).round(2)
        
        return summary

# Example usage and testing
def main():
    """Main execution function for data acquisition phase."""
    
    print("=" * 60)
    print("CS 506 Project - Data Acquisition Phase (Week 2)")
    print("ETFs: QQQ, SPY, IWM | Period: 2010-2026")
    print("=" * 60)
    
    # Initialize data acquisition
    aquisition = ETFDataAcquisition(
        tickers=['QQQ', 'SPY', 'IWM'],
        start_date='2010-01-01',
        end_date='2026-03-06'
    )
    
    # Download data with caching
    print("\n[1/4] Downloading data from Yahoo Finance...")
    raw_data = aquisition.download_data(use_cache=True)
    
    # Verify data completeness
    print("\n[2/4] Verifying data completeness...")
    verification = aquisition.verify_data_completeness()
    
    # Display verification results
    for ticker, results in verification.items():
        status_symbol = "✅" if results['status'] == 'PASS' else "⚠️"
        print(f"  {status_symbol} {ticker}: {results['trading_days']} days "
              f"({results['completeness_pct']}% of expected)")
    
    # Combine datasets
    print("\n[3/4] Combining datasets...")
    combined = aquisition.combine_datasets()
    
    # Show summary
    print("\n[4/4] Data Summary:")
    summary = aquisition.get_data_summary()
    print(summary)
    
    # Save data
    print("\nSaving data to files...")
    aquisition.save_data('cs506_etf_data')
    
    print("\n✅ Data acquisition phase complete!")
    print("Files created:")
    print("  - cs506_etf_data_multilevel.csv (Multi-level columns format)")
    print("  - cs506_etf_data_long.csv (Long format with Ticker column)")
    print("  - cs506_etf_data_metadata.txt (Data verification metadata)")
    
    return aquisition

if __name__ == "__main__":
    # Install required packages if not already installed
    required_packages = ['yfinance', 'pandas', 'numpy', 'requests_cache']
    
    import subprocess
    import sys
    
    def install_package(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            install_package(package)
    
    # Run main acquisition
    data = main()