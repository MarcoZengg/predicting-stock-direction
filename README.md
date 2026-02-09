# Predict Stock Direction With Supervised Learning

## CS 506 Project Proposal

**Course:** [BU CS 506](https://gallettilance.github.io/final_project/)  
**Project Describtion:** https://gallettilance.github.io/final_project/

---

## 1. Project Description

This project goal is to **compare the effectiveness of linear and non-linear supervised learning models in predicting short-term ETF return direction** for major U.S. equity ETFs (QQQ, SPY, and IWM). Predicting exact stock or ETF prices is highly noisy and unreliable in practice. In contrast, short-term directional prediction provides a more robust and more measurable objective.

Given a stock’s historical daily data up to trading day *t* (open, high, low, close, volume), the task is to predict the **binary direction of the next-day return**:

$$
y_{t+1} = 
\begin{cases}
1, & \text{if the return from day } t \text{ to } t+1 \text{ is positive} \\
0, & \text{otherwise}
\end{cases}
$$

This problem is formulated as a binary classification task, which is more robust and interpretable than direct price regression and aligns well with supervised learning techniques.

---

## 2. Project Goals

| Goal | Success criterion |
|------|--------------------|
| **Primary** | Predict next-day return direction (up/down) from historical daily features. |
| **Comparison** | Compare Logistic Regression, Random Forest, and/or Gradient Boosted Trees on the same features and splits. |
| **Evaluation** | Quantify predictive value via Accuracy, Precision, Recall, and ROC-AUC on a held-out test set. |
| **Baselines** | Compare against random guessing (50%), always-positive, and minimal-feature logistic regression. |

**Good (rubric-style):** “Successfully predict the direction of next-day stock return using open/high/low/close/volume and derived features, with clear baselines and metrics.”  
**Scope:** Two months; individual project. All data processing, modeling, and analysis by the author.

---

## 3. Data Collection Plan

| Item | Plan |
|------|------|
| **Source** | [Yahoo Finance](https://finance.yahoo.com/) (e.g., via `yfinance` or equivalent API/library). |
| **Assets** | Select U.S. equity ETFs (e.g., QQQ, SPY, IWM). ETFs are used instead of individual stocks to reduce noise and improve the robustness of the analysis. |
| **Time range** | Approximately 2010–2026. |
| **Raw features** | Open, High, Low, Close prices; Trading volume. |
| **Method** | Programmatic download via public API or library; no scraping of protected content. Data collection will be implemented in code and documented in the final README. |

**Collection method:** API/library-based (e.g., `yfinance` in Python).

---

## 4. Modeling

- **Baseline:** Logistic Regression.  
- **Non-linear:** Random Forest and/or Gradient Boosted Trees (e.g., scikit-learn, XGBoost).  
- **Optional:** Simple feed-forward neural network.  

Same feature sets and chronological train/validation/test splits for all models to ensure fair comparison and avoid look-ahead bias.

**Features (derived):** Daily and log returns, rolling moving averages (e.g., 5-day, 10-day), rolling volatility, momentum-style indicators.

---

## 5. Visualization Plan

- **Exploratory:** Time series of close price and volume; distribution of returns; correlation heatmap of features.  
- **Model evaluation:** ROC curves; precision–recall curves; bar charts of Accuracy / Precision / Recall / AUC across models and baselines.  
- **Results:** Tables and plots comparing models vs. baselines; optional confusion matrices.  

Visualizations will be clear, labeled, and support interpretation of data and results (aligning with course check-in and final report rubric).

---

## 6. Test Plan 

- **Split:** train/validation/test (e.g., train on earlier years, validate on a held-out period, test on latest period) to prevent look-ahead bias.  
- **Metrics:** Accuracy, Precision, Recall, ROC-AUC on the **test set**.  
- **Analysis:** Compare performance across models and baselines; discuss whether added complexity yields meaningful improvement.  

*Example:* Train on 2010–2019, validate on 2020–2021, test on 2022–2024 (exact dates to be set in implementation).

---

## 7. Anticipated Challenges

| Challenge | Possible Solution |
|-----------|------------|
| High noise in financial data | Feature smoothing (e.g., rolling stats), regularization. |
| Overfitting | Validation set, limit model complexity, avoid excess features. |
| Class imbalance | Report Precision/Recall/AUC in addition to Accuracy; consider class weight or sampling if needed. |

---

## 8. Project Timeline

| Phase | Tasks |
|-------|--------|
| **Data & setup** | Set up repo, fetch data (Yahoo Finance), document collection. |
| **Cleaning & features**  | Clean data, derive returns/rolling features, train/val/test split. |
| **Baselines & models** |  Implement Logistic Regression, Random Forest/GBM; optional neural net. |
| **Evaluation & viz** | Metrics, ROC/PR curves, comparison tables and figures. |
| **Report & polish** |  README (build/run, reproducibility), testing, final report and submission. |

**Fallback:** If scope is too large, reduce to fewer stocks or a shorter date range; if too small, add more assets or features. Total target: about two months of work.

---

## 9. Team Contributions

**Team size:** 4 members

| Team Member       | Main Responsibilities                           |
|-------------------|------------------------------------------------|
| Teammate 1        | ...   |
| Teammate 2        | ...   |
| Teammate 3        | ...      |
| Teammate 4        | ...       |
