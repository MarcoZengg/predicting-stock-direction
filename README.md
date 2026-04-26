# Predict Stock Direction With Supervised Learning

Course: [BU CS 506 Final Project](https://gallettilance.github.io/final_project/)

## Presentation Video (Required)

YouTube link (temporary): [Project Presentation (replace with final upload)](https://example.com/cs506-final-presentation)

---

## Quick Navigation

- [Build and Run](#build-and-run-most-important)
- [Testing and GitHub Workflow](#testing-and-github-workflow)
- [Project Overview and Goal](#project-overview-and-goal)
- [Data Collection](#data-collection)
- [Data Cleaning](#data-cleaning)
- [Feature Extraction](#feature-extraction)
- [Modeling](#modeling)
- [Visualizations of Data](#visualizations-of-data-interactive-encouraged)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Team Contributions](#team-contributions)

---

## Build and Run (Most Important)

### Environment


| Item         | Value              |
| ------------ | ------------------ |
| OS           | macOS, Linux       |
| Python       | 3.10+              |
| Dependencies | `requirements.txt` |
| Task runner  | `Makefile`         |


### Quick Start

```bash
make install
make test
make fetch-data   # optional if you want fresh raw data
make reproduce
```

`make reproduce` runs:

- `scripts/data/process_data.py`
- `scripts/training/train_logistic.py`
- `scripts/training/train_random_forest.py`

---

## Testing and GitHub Workflow

### What this section covers

How tests run locally and in CI.

### Code/files used

- `tests/test_smoke.py`
- `.github/workflows/ci.yml`

### Output artifacts

- Passing local test run with `make test`
- Passing CI run on push/PR

### Full visualization outputs (all models, all tickers)

- Temporary CI badge: CI Status
- Temporary CI link: [GitHub Actions Workflow (replace with real repo URL)](https://example.com/github-actions-ci)

---

## Project Overview and Goal

### What this section covers

Problem definition and prediction target.

### Label definition

$$
y_{t+1} =
\begin{cases}
1, & \text{if the return from day } t \text{ to } t+1 \text{ is positive} 
0, & \text{otherwise}
\end{cases}
$$

### Goal

- Predict next-day direction (up/down) for `SPY`, `QQQ`, and `IWM`.
- Compare linear and non-linear supervised models.
- Evaluate with chronological train/test splits.

### Architecture and data flow

```mermaid
flowchart LR
  rawData["Raw OHLCV Data (Yahoo Finance)"] --> cleaning["Data Cleaning"]
  cleaning --> features["Feature Extraction"]
  features --> split["Chronological Train/Test Split"]
  split --> models["Model Training (Logistic + Random Forest)"]
  models --> metrics["Evaluation Metrics"]
  metrics --> results["Results and Interpretation"]
```

Caption: This workflow shows the end-to-end pipeline used in this project, from raw market data to evaluated model results.

### Why this target matters

Predicting exact next-day prices is typically too noisy for small academic projects, but directional prediction is a practical and measurable objective. The binary target is directly useful for comparing models, evaluating signal quality, and discussing whether engineered features contain predictive information beyond naive baselines.

---

## Data Collection

### What this section covers

Where data comes from and how it is pulled.

### Code/files used

- `scripts/data/fetch_data.py`
- `data/raw/{TICKER}_historical.csv`

### Output artifacts

- Raw OHLCV data for `SPY`, `QQQ`, `IWM` in `data/raw/`

### Source summary


| Item        | Details                            |
| ----------- | ---------------------------------- |
| Source      | Yahoo Finance (`yfinance`)         |
| Assets      | `SPY`, `QQQ`, `IWM`                |
| Pull script | `scripts/data/fetch_data.py`       |
| Output path | `data/raw/{TICKER}_historical.csv` |


ETFs (`SPY`, `QQQ`, `IWM`) were selected instead of individual stocks to reduce idiosyncratic company-level noise (e.g., earnings surprises, one-off corporate events) and focus on broader market behavior. This makes the directional-label task more stable, improves comparability across assets, and better matches the project goal of testing whether generalizable market features can predict next-day direction.

### Data coverage


| Ticker | Start Date | End Date   | Total Rows | Train Rows | Test Rows |
| ------ | ---------- | ---------- | ---------- | ---------- | --------- |
| `SPY`  | 2010-02-02 | 2025-12-29 | 4,000      | 3,200      | 800       |
| `QQQ`  | 2010-02-02 | 2025-12-29 | 4,000      | 3,200      | 800       |
| `IWM`  | 2010-02-02 | 2025-12-29 | 4,000      | 3,200      | 800       |


---

## Data Cleaning

### What this section covers

How raw data is standardized and split.

### Code/files used

- `scripts/data/process_data.py`

### Output artifacts

- `data/processed/{TICKER}_processed.csv`
- `data/processed/{TICKER}_train.csv`
- `data/processed/{TICKER}_test.csv`

### Processing steps

1. Parse and standardize date/price columns.
2. Sort chronologically.
3. Compute returns and label.
4. Drop NaN rows from rolling features.
5. Split 80/20 by time.

### To finalize before submission

![Train/Test Split Timeline](data/images/train_test_split_timeline.png)

Chronological splitting prevents data leakage by ensuring every test observation occurs strictly after the training window, so the model never learns from future information during training.

---

## Feature Extraction

### What this section covers

Feature families used for model input.

### Code/files used

- `scripts/data/process_data.py`
- `scripts/data/feature_engineering.py`
- `results/feature_importance.csv`

### Output artifacts

- Engineered feature columns in `data/processed/*_processed.csv`
- Ranked feature-importance export in `results/feature_importance.csv`

### Feature groups

- Lag returns
- Rolling averages and volatility
- Momentum indicators
- Moving-average ratios
- Volume features

### To finalize before submission

### Feature dictionary (key features)


| Feature          | Type              | Description                                                           |
| ---------------- | ----------------- | --------------------------------------------------------------------- |
| `return`         | Price momentum    | Daily percentage return of the ETF close price                        |
| `lag_return_1`   | Price momentum    | Previous-day return (1-day lag)                                       |
| `lag_return_5`   | Price momentum    | Return lagged by 5 trading days                                       |
| `rolling_mean_5` | Trend             | 5-day moving average of daily returns                                 |
| `rolling_std_10` | Volatility        | 10-day rolling standard deviation of returns                          |
| `momentum_10`    | Trend/momentum    | 10-day price momentum (relative change over 10 days)                  |
| `close_to_ma_10` | Trend strength    | Relative distance between close price and 10-day moving average       |
| `volume_change`  | Volume dynamics   | Day-over-day percent change in traded volume                          |
| `vix_level`      | Market regime     | Market-implied volatility level used as a risk regime proxy           |
| `tnx_level`      | Macro rate signal | 10-year Treasury yield level used as an interest-rate context feature |


### Interpretation of top features

- Rolling return and volatility features (such as `rolling_mean_5` and `rolling_std_10`) appear frequently in high-importance rows, suggesting short-horizon trend/volatility state carries more signal than raw price level alone.
- Cross-asset regime features (`vix_level`, `tnx_level`, and related rate/volatility returns) rank highly in multiple models, indicating that broader market context contributes to next-day direction prediction.
- The mixed signs and modest magnitudes in linear-model coefficients support the project’s core finding: predictive signal exists but is weak, so performance depends on combining many small effects rather than one dominant feature.

---

## Modeling

### What this section covers

Model training setup and evaluation strategy.

### Code/files used

- `scripts/training/train_logistic.py`
- `scripts/training/train_random_forest.py`
- `results/metrics.csv`

### Output artifacts

- Model-level performance metrics in `results/metrics.csv`

### Evaluation metrics


| Metric    | Why this metric matters                          |
| --------- | ------------------------------------------------ |
| Accuracy  | Overall correctness across all predictions       |
| Precision | Reliability of predicted positive class          |
| Recall    | Ability to capture actual positive class         |
| F1        | Balance between precision and recall             |
| ROC-AUC   | Ranking quality across classification thresholds |


### To finalize before submission

### Model comparison (from `results/metrics.csv`)

![All Metric Comparisons](data/images/results/all_metric_comparisons.png)

Caption: Side-by-side grouped bar charts for Accuracy, Precision, Recall, F1, and ROC-AUC across all ticker/model pairs from the latest regenerated `results/metrics.csv`.  
Takeaway: No single model dominates every metric; Gradient Boosting and Logistic Regression provide stronger balance metrics, while baseline methods can still appear strong on raw accuracy.

Individual metric charts:

![Accuracy Comparison](data/images/results/accuracy_comparison.png)
![Precision Comparison](data/images/results/precision_comparison.png)
![Recall Comparison](data/images/results/recall_comparison.png)
![F1 Comparison](data/images/results/f1_comparison.png)
![ROC-AUC Comparison](data/images/results/roc_auc_comparison.png)

These individual views are included for easier inspection of each evaluation criterion.

---

## Visualizations of Data (Interactive Encouraged)

### What this section covers

Static and interactive visual evidence supporting conclusions.

### Code/files used

- `scripts/visualization/visualization_data.py`
- `scripts/visualization/visualization_train.py`

### Output artifacts

- Figures exported to your image/output folders

### To finalize before submission

**Figure 1 - Price trends by ticker**

![Price Trend and Train/Test Split](data/images/data_analysis/SPY_data_analysis.png)

Caption: SPY price trend with chronological train/test segmentation.  
Takeaway: The split keeps test observations in the latest market regime, which better reflects realistic forward prediction.

**Figure 2 - Class balance across ETFs**

![Class Balance Across ETFs](data/images/data_analysis/class_balance_all_etfs.png)

Caption: Up/down label proportions for SPY, QQQ, and IWM after preprocessing.  
Takeaway: All tickers show mild class imbalance toward positive labels, which is why recall and ROC-AUC are reported alongside accuracy.

**Figure 3 - Confusion matrices (all model families on SPY)**

![SPY Logistic Confusion Matrix](data/images/all_models/spy/logistic_regression/spy_logistic_regression_1_confusion_matrix.png)
![SPY Random Forest Confusion Matrix](data/images/all_models/spy/random_forest/spy_random_forest_1_confusion_matrix.png)
![SPY Gradient Boosting Confusion Matrix](data/images/all_models/spy/gradient_boosting/spy_gradient_boosting_1_confusion_matrix.png)

Caption: Confusion matrices for logistic regression, random forest, and gradient boosting on SPY.  
Takeaway: All model families are now visualized, enabling side-by-side comparison of error structure rather than relying on a single example model.

**Figure 4 - ROC and PR curves (all model families on SPY)**

![SPY Logistic ROC](data/images/all_models/spy/logistic_regression/spy_logistic_regression_2_roc_curve.png)
![SPY Random Forest ROC](data/images/all_models/spy/random_forest/spy_random_forest_2_roc_curve.png)
![SPY Gradient Boosting ROC](data/images/all_models/spy/gradient_boosting/spy_gradient_boosting_2_roc_curve.png)
![SPY Logistic PR](data/images/all_models/spy/logistic_regression/spy_logistic_regression_5_precision_recall_curve.png)
![SPY Random Forest PR](data/images/all_models/spy/random_forest/spy_random_forest_5_precision_recall_curve.png)
![SPY Gradient Boosting PR](data/images/all_models/spy/gradient_boosting/spy_gradient_boosting_5_precision_recall_curve.png)

Caption: ROC and precision-recall curves for all three model families on SPY.  
Takeaway: Curves across models can now be compared directly under the same ticker; equivalent sets were also generated for QQQ and IWM.

Additional generated outputs for all tickers/models:

- `data/images/all_models/spy/`
- `data/images/all_models/qqq/`
- `data/images/all_models/iwm/`

---

## Results

### What this section covers

Observed outcomes from generated result files.

### Code/files used

- `results/metrics.csv`
- `results/data_summary.csv`

### Output artifacts

- Model performance summary
- Dataset size and class balance summary

### Current highlights

- Majority baseline has high raw accuracy on SPY/QQQ because the class distribution is skewed, but it offers no discrimination (`ROC-AUC = 0.5` by design).
- Best ROC-AUC in the current regenerated run is `0.5546` (IWM Logistic Regression).
- Best F1 in the current regenerated run is `0.2834` (IWM Gradient Boosting), indicating improved balance between precision and recall versus other models.

### Dataset summary

- Around 4,000 rows per ticker
- 3,200 train and 800 test rows per ticker
- Positive label rate around 53% to 56%

### To finalize before submission

### Final comparison summary


| Ticker | Best Accuracy Model | Best ROC-AUC Model  | Best ROC-AUC |
| ------ | ------------------- | ------------------- | ------------ |
| `SPY`  | Random Forest       | Logistic Regression | 0.5382       |
| `QQQ`  | Gradient Boosting   | Logistic Regression | 0.5255       |
| `IWM`  | Gradient Boosting   | Logistic Regression | 0.5546       |


### Did we achieve our goal?

Partially. The project successfully trained and compared multiple supervised models with reproducible chronological evaluation, and several models now exceed random discrimination (`ROC-AUC > 0.5`), with the best at `0.5546` on IWM. However, absolute performance remains modest, so this is evidence of weak but non-zero predictive signal rather than a robust trading-grade predictor.

---

## Limitations and Future Work

### What this section covers

Constraints of current approach and next steps.

### Current limitations

- Daily directional prediction is weak/noisy.
- Accuracy alone can be misleading with class imbalance.
- Results depend on date range and feature choices.

### To finalize before submission

- Better validation plan: use rolling-window backtesting (train on an expanding window and test on the next fixed horizon) to measure temporal stability and reduce single-split sensitivity.
- Expanded feature plan: incorporate aligned news sentiment, macro event indicators, and richer regime-state features to improve signal beyond price/volume-only inputs.
- Practical deployment idea: expose daily probability outputs in a lightweight dashboard as decision support (not automated trading), with alert thresholds and recent model-calibration diagnostics.

---

## Team Contributions


| Team Member   | Main Responsibilities |
| ------------- | --------------------- |
| Xiankun Zeng  | Data Acquisition      |
| Haoran Zhang  | Data Processing       |
| Hoang Anh Vu  | Data Processing       |
| Team Member 4 | ...                   |


### To finalize before submission

- Xiankun Zeng: implemented and validated data collection scripts; verified ticker/date coverage and raw file integrity; documented source assumptions and retrieval flow.
- Haoran Zhang: built core preprocessing steps (sorting, labeling, split logic); ensured processed outputs were reproducible; validated train/test artifacts.
- Hoang Anh Vu: expanded feature engineering and indicator creation; supported feature interpretation and artifact generation; assisted with pipeline consistency checks.
- Team Member 4: executed model training and metric comparison; consolidated evaluation outputs; supported result interpretation and visualization linkage in the report.

---

## Repository Organization

- `scripts/` - data pipeline and modeling scripts
- `data/raw/` - collected raw datasets
- `data/processed/` - cleaned and split datasets
- `results/` - reproducible summary outputs
- `tests/` - automated test code
- `.github/workflows/` - CI workflow definitions

---

## Final Submission Checklist

- Replace temporary URLs (video, CI link, interactive dashboard) with final project links.
- Add YouTube video link at the top.
- Add final model comparison table.
- Add required visuals with captions and interpretation.
- Add at least one interactive visualization link.
- Verify `make install`, `make test`, and `make reproduce`.
- Confirm GitHub Actions CI passes on latest commit.

---

## License

MIT (see `LICENSE`).