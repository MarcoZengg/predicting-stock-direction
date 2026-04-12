import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Dynamically determine the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_dir = os.path.join(base_dir, 'data', 'processed')

# Define path for saving images
image_save_dir = os.path.join(base_dir, 'data','images','data_analysis')
os.makedirs(image_save_dir, exist_ok=True)

# Color scheme for consistency
COLORS = {
    'raw': '#1f77b4',      # Blue
    'train': '#2ca02c',    # Green
    'test': '#d62728',     # Red
    'processed': '#9467bd' # Purple
}

THRESHOLD = 0.002  # 0.2% from processing script

def load_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def plot_stock_trends(stock_symbol):
    """Plot stock trends with clear train/test split visualization."""
    
    # File paths
    train_file = os.path.join(processed_data_dir, f"{stock_symbol}_train.csv")
    test_file = os.path.join(processed_data_dir, f"{stock_symbol}_test.csv")
    processed_file = os.path.join(processed_data_dir, f"{stock_symbol}_processed.csv")
    
    # Load data

    train_data = load_data(train_file)
    test_data = load_data(test_file)
    processed_data = load_data(processed_file)
    
    # Convert dates
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    processed_data['Date'] = pd.to_datetime(processed_data['Date'])
    
    # Sort by date
    train_data = train_data.sort_values('Date')
    test_data = test_data.sort_values('Date')
    processed_data = processed_data.sort_values('Date')
    
    # Create figure with subplots
    fig, axes = plt.subplots( figsize=(15, 10))
    fig.suptitle(f'{stock_symbol} - Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    
    # Plot Train vs Test Split
    axes.plot(train_data['Date'], train_data['Close'], color=COLORS['train'], linewidth=1.5, label='Train Set (80%)', alpha=0.8)
    axes.plot(test_data['Date'], test_data['Close'], color=COLORS['test'], linewidth=1.5, label='Test Set (20%)', alpha=0.8)
    
    # Add vertical line at split point
    split_date = train_data['Date'].iloc[-1]
    axes.axvline(x=split_date, color='black', linestyle='--', linewidth=1, label='Train/Test Split')
    
    axes.set_title(f'{stock_symbol} - Train/Test Split (Chronological)')
    axes.set_xlabel('Date')
    axes.set_ylabel('Close Price ($)')
    axes.legend(loc='upper left')
    axes.grid(True, alpha=0.3)
    

    plt.tight_layout()
    
    # Save the combined plot
    combined_image_path = os.path.join(image_save_dir, f"{stock_symbol}_data_analysis.png")
    plt.savefig(combined_image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Data analysis plot saved to {combined_image_path}")
    
    # Also create a simple overlay plot showing all together
    fig2, ax = plt.subplots(figsize=(14, 7))
    

    ax.plot(train_data['Date'], train_data['Close'], color=COLORS['train'], linewidth=1.2, label='Train Set', alpha=0.7)
    ax.plot(test_data['Date'], test_data['Close'], color=COLORS['test'], linewidth=1.2, label='Test Set', alpha=0.7)
    ax.plot(processed_data['Date'], processed_data['Close'], color=COLORS['processed'], linewidth=0.8, alpha=0.4, label='Processed Data')
    
    # Highlight the split
    ax.axvline(x=train_data['Date'].iloc[-1], color='black', linestyle='--', linewidth=1.5, 
               label=f'Train/Test Split: {train_data["Date"].iloc[-1].strftime("%Y-%m-%d")}')
    
    ax.set_title(f'{stock_symbol} - All Data Overlay (Raw, Train, Test, Processed)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price ($)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotation (FIXED)
    # total_processed = len(train_data) + len(test_data)
    # stats_text = f"""
    # Statistics:
    # • Train set: {len(train_data):,} days ({len(train_data)/total_processed*100:.0f}% of processed)
    # • Test set: {len(test_data):,} days ({len(test_data)/total_processed*100:.0f}% of processed)
    # • Total processed: {total_processed:,} days (after feature engineering)
    # • Threshold: Up if return > {THRESHOLD*100:.1f}%
    # """
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # plt.tight_layout()
    
    # Save overlay plot
    # overlay_path = os.path.join(image_save_dir, f"{stock_symbol}_all_data_overlay.png")
    # plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"✅ Overlay plot saved to {overlay_path}")

def plot_return_distribution(stock_symbol):
    """Plot return distribution to show threshold impact."""
    
    processed_file = os.path.join(processed_data_dir, f"{stock_symbol}_processed.csv")
    df = pd.read_csv(processed_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Return histogram
    axes[0].hist(df['return'].dropna(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0].axvline(x=THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD*100:.1f}%)')
    axes[0].set_xlabel('Daily Return')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{stock_symbol} - Daily Return Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Class balance pie chart
    labels = [f'Up (>{THRESHOLD*100:.1f}%)', f'Down (≤{THRESHOLD*100:.1f}%)']
    sizes = [df['label'].sum(), len(df) - df['label'].sum()]
    colors = ['green', 'red']
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{stock_symbol} - Class Distribution')
    
    plt.tight_layout()
    save_path = os.path.join(image_save_dir, f"{stock_symbol}_return_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Return distribution plot saved to {save_path}")

def create_class_balance_plot():
    """Create a bar chart showing class balance for all ETFs."""
    
    symbols = ['SPY', 'QQQ', 'IWM']
    up_ratios = []
    
    for symbol in symbols:
        processed_file = os.path.join(processed_data_dir, f"{symbol}_processed.csv")
        df = pd.read_csv(processed_file)
        up_ratio = df['label'].mean()
        up_ratios.append(up_ratio)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(symbols))
    bars = ax.bar(x_pos, up_ratios, color=['blue', 'green', 'red'], alpha=0.7)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Random (50%)')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, up_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('ETF')
    ax.set_ylabel('Up Days Ratio')
    ax.set_title(f'Class Balance: Up Days vs Down Days (Threshold > {THRESHOLD*100:.1f}%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(symbols)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation with threshold info
    ax.text(0.02, 0.98, 
            f'Note: Label = 1 only if return > {THRESHOLD*100:.1f}%\n'
            'All ETFs have more down days than up days,\nreflecting the long-term bull market bias.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    balance_path = os.path.join(image_save_dir, "class_balance_all_etfs.png")
    plt.savefig(balance_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Class balance plot saved to {balance_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("CS 506 - DATA VISUALIZATION")
    print("=" * 60)
    
    stock_symbols = ['SPY', 'QQQ', 'IWM']
    
    # Generate individual ETF plots
    for symbol in stock_symbols:
        print(f"\n📊 Visualizing data for {symbol}...")
        plot_stock_trends(symbol)
        plot_return_distribution(symbol)  # NEW: Add return distribution
    
    # Generate class balance comparison
    print(f"\n📈 Creating class balance comparison...")
    create_class_balance_plot()
    
    print("\n" + "=" * 60)
    print(f"✅ All visualizations saved to: {image_save_dir}")
    print("=" * 60)