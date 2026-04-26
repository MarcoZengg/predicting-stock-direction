from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_test_split_timeline.png"

    tickers = ["SPY", "QQQ", "IWM"]
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for idx, ticker in enumerate(tickers):
        train_df = pd.read_csv(processed_dir / f"{ticker}_train.csv")
        test_df = pd.read_csv(processed_dir / f"{ticker}_test.csv")

        train_start = pd.to_datetime(train_df["Date"]).min()
        train_end = pd.to_datetime(train_df["Date"]).max()
        test_start = pd.to_datetime(test_df["Date"]).min()
        test_end = pd.to_datetime(test_df["Date"]).max()

        # Train segment
        ax.hlines(
            y=idx,
            xmin=train_start,
            xmax=train_end,
            color="#2ca02c",
            linewidth=10,
            label="Train (80%)" if idx == 0 else None,
        )
        # Test segment
        ax.hlines(
            y=idx,
            xmin=test_start,
            xmax=test_end,
            color="#d62728",
            linewidth=10,
            label="Test (20%)" if idx == 0 else None,
        )
        # Split marker
        ax.vlines(
            x=train_end,
            ymin=idx - 0.25,
            ymax=idx + 0.25,
            color="black",
            linewidth=1.5,
            linestyles="--",
        )

    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    ax.set_xlabel("Date")
    ax.set_title("Chronological Train/Test Split Timeline by Ticker")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved timeline figure to: {output_path}")


if __name__ == "__main__":
    main()
