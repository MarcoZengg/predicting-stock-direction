# Predict Stock Direction With Supervised Learning

## CS 506 Project Proposal

**Course:** [BU CS 506](https://gallettilance.github.io/final_project/)  
**Course project requirements:** https://gallettilance.github.io/final_project/

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

| Goal Type | Description |
|-----------|-------------|
| **Primary Objective** | Evaluate whether historical daily market features contain predictive signal for next-day return direction (up/down) of major U.S. equity ETFs (QQQ, SPY, IWM). |
| **Model Comparison** | Compare the predictive performance of linear (Logistic Regression) and non-linear models (Random Forest and Gradient Boosted Trees) using identical feature sets and chronological data splits. |
| **Evaluation Strategy** | Assess model performance on a held-out test set using Accuracy, Precision, Recall, and ROC-AUC to capture both overall correctness and class-level behavior. |
| **Baselines** | Benchmark all models against random guessing (50%), an always-positive predictor, and a minimal-feature logistic regression model to contextualize results. |

---

## 3. Data Collection Plan

| Item | Plan |
|------|------|
| **Source** | [Yahoo Finance](https://finance.yahoo.com/) (e.g., via `yfinance` or equivalent API/library). |
| **Assets** | Selected U.S. equity ETFs: QQQ, SPY, IWM (used instead of single-name stocks to reduce noise). |
| **Time range** | Approximately 2010–2026. |
| **Raw features** | Open, High, Low, Close; Volume. |
| **Method** | Programmatic download via public API (e.g., `yfinance` in Python). Collection and usage will be implemented in code and documented in this README. |

---

## 4. Modeling
We can start from building logistic regression and linear regression to explore analytic methods, and find potential OVB, multicolinearity and Heteroskedasticity. Then we move on to advanced ML models like XGBoost, Decision Tree and Random Forest.


---

## 5. Visualization Plan
Tableau or PowerBI is preffered. Another option is to make a website or publish the dataset to Kaggle.


---

## 6. Test Plan


---

## 7. Project Schedule

| Phase                    | Key Tasks                                                                                          |
|--------------------------|----------------------------------------------------------------------------------------------------|
| Planning & Setup         | Define project objectives and requirements; set up software tools and team organization.           |
| Data Acquisition         | Gather/source all raw data (e.g., via API or download) and ensure appropriate coverage.            |
| Data Processing          | Clean and preprocess data for consistency, handle missing values, and structure for analysis.       |
| Feature Engineering      | Develop and select informative features to support model learning and improve predictions.          |
| Model Development        | Build, train, and tune multiple models using chosen features and methodologies.                     |
| Evaluation               | Assess models using balanced performance metrics (accuracy, recall, precision, ROC-AUC).            |
| Visualization            | Develop clear and unbiased visualizations to communicate results and insights.                      |
| Finalization & Reporting | Summarize findings evenly, report results, and prepare all project documentation for submission.    |

**Balance Plan:** If project workload becomes unbalanced, adjust scope—reduce ETF count or date range if overextended, or add assets/features if under-scoped—to keep effort and timeline (~2 months) manageable and fair.

---
## 8. Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Collection** | `yfinance` Python library | Fetch historical OHLCV data from Yahoo Finance |
| **Data Processing** | `pandas`, `numpy` | Data manipulation, cleaning, and feature engineering |
| **Machine Learning** | `scikit-learn` (v1.3+) | Model implementation (Logistic Regression, Random Forest, Gradient Boosting) |
| **Feature Engineering** | `ta` (Technical Analysis library) | Calculate technical indicators (RSI, MACD, etc.) |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` | Create static and interactive visualizations |
| **Version Control** | Git, GitHub | Code management and collaboration |
| **Environment** | Python 3.9+, Jupyter Notebooks, VS Code | Development environment and reproducible analysis |
| **Model Persistence** | `joblib` or `pickle` | Save trained models for evaluation |
| **Documentation** | Markdown, Jupyter | Project documentation and reporting |

---
## 9. Team Contributions

**Team size:** 4 members

| Team Member       | Main Responsibilities                           |
|-------------------|------------------------------------------------|
| Teammate 1        | ...   |
| Teammate 2        | ...   |
| Hoang Anh Vu        | ...      |
| Teammate 4        | ...       |
