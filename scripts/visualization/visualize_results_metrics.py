from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def apply_zoomed_axis(ax, values: pd.Series) -> None:
    """Set a comfortable zoomed y-axis with readable tick spacing."""
    vmin = float(values.min())
    vmax = float(values.max())
    padding = 0.005
    lower = max(0.0, np.floor((vmin - padding) / 0.01) * 0.01)
    upper = min(1.0, np.ceil((vmax + padding) / 0.01) * 0.01)
    if upper - lower < 0.03:
        upper = min(1.0, lower + 0.03)
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.arange(lower, upper + 1e-9, 0.01))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    metrics_path = project_root / "results" / "metrics.csv"
    output_dir = project_root / "data" / "images" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)

    metric_specs = [
        ("accuracy", "Accuracy Comparison by Ticker and Model", "accuracy_comparison.png"),
        ("precision", "Precision Comparison by Ticker and Model", "precision_comparison.png"),
        ("recall", "Recall Comparison by Ticker and Model", "recall_comparison.png"),
        ("f1", "F1 Score Comparison by Ticker and Model", "f1_comparison.png"),
        ("roc_auc", "ROC-AUC Comparison by Ticker and Model", "roc_auc_comparison.png"),
    ]

    saved_paths = []
    for metric, title, filename in metric_specs:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="ticker", y=metric, hue="model")
        plt.title(title)
        plt.xlabel("Ticker")
        plt.ylabel(metric.upper() if metric != "roc_auc" else "ROC-AUC")
        if metric in {"accuracy", "roc_auc"}:
            apply_zoomed_axis(plt.gca(), df[metric])
        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=220)
        plt.close()
        saved_paths.append(path)

    # Optional all-in-one panel for quick report insertion.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for idx, (metric, title, _) in enumerate(metric_specs):
        sns.barplot(data=df, x="ticker", y=metric, hue="model", ax=axes[idx])
        axes[idx].set_title(title.replace(" Comparison by Ticker and Model", ""))
        axes[idx].set_xlabel("Ticker")
        axes[idx].set_ylabel(metric.upper() if metric != "roc_auc" else "ROC-AUC")
        if metric in {"accuracy", "roc_auc"}:
            apply_zoomed_axis(axes[idx], df[metric])
        if idx != 0:
            axes[idx].get_legend().remove()
        else:
            axes[idx].legend(title="Model", fontsize=8, title_fontsize=9)
    axes[-1].axis("off")
    plt.tight_layout()
    panel_path = output_dir / "all_metric_comparisons.png"
    plt.savefig(panel_path, dpi=220)
    plt.close()

    for path in saved_paths:
        print(f"Saved: {path}")
    print(f"Saved: {panel_path}")


if __name__ == "__main__":
    main()
