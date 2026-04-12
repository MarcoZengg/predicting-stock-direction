import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_train import plot_model_results

# ====== Project Paths ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")



# ====== Load Data (SPY only for now) ======
# Load preprocessed train/test datasets
train = pd.read_csv(os.path.join(DATA_DIR, "SPY_train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "SPY_test.csv"))

# ====== Select Features ======
# These features were engineered in process_data.py
# They include return-based, momentum, volatility, and volume indicators

FEATURES = [
    "lag_return_1",      # previous day's return
    "rolling_mean_5",    # 5-day average return
    "rolling_std_5",     # 5-day return volatility

    "momentum_5",        # 5-day momentum
    "momentum_10",       # 10-day momentum

    "ma_5",              # 5-day moving average of price
    "ma_10",             # 10-day moving average of price

    "volatility_10",     # 10-day rolling volatility

    "volume_change",     # daily percentage change in volume
    "volume_ma_5"        # 5-day moving average of volume
]

X_train = train[FEATURES]
y_train = train["label"]

X_test = test[FEATURES]
y_test = test["label"]

# ====== Scaling ======
# Standardize features to have zero mean and unit variance
# This is important for logistic regression
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====== Train Model ======
# Train a logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ====== Predictions ======
# Predict class labels and probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ====== Evaluation ======
# Evaluate classification performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("===== SPY Logistic Regression Baseline =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

print("Predicted class distribution:")
print(pd.Series(y_pred).value_counts(normalize=True))



# Visualize results with comprehensive plots
VIZ_DIR = os.path.join(PROJECT_ROOT, "data","images","logistic")
os.makedirs(VIZ_DIR, exist_ok=True)
metrics, files = plot_model_results(
    y_test=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    model=model,
    feature_names=FEATURES,
    model_name="LogisticRegression",
    save_dir=VIZ_DIR,
    test_data=test,
    date_col='Date'
)