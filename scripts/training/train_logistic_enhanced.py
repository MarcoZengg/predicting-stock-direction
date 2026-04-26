import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "scripts", "visualization")
if VISUALIZATION_DIR not in sys.path:
    sys.path.append(VISUALIZATION_DIR)

from visualization_train import plot_model_results

# ====== Project Paths ======

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_enhanced")

# ====== Choose Ticker ======
TICKER = "SPY"
#TICKER = "QQQ"
#TICKER = "IWM"

# ====== Load Data ======
train = pd.read_csv(os.path.join(DATA_DIR, f"{TICKER}_train_enhanced.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, f"{TICKER}_test_enhanced.csv"))

# ====== Select Features ======
# Original features
FEATURES = [
    "lag_return_1",
    "rolling_mean_5",
    "rolling_std_5",
    "momentum_5",
    "momentum_10",
    "ma_5",
    "ma_10",
    "volatility_10",
    "volume_change",
    "volume_ma_5"
]

if TICKER == "SPY":
    FEATURES += [
        "vix_return",
        "vix_return_ma_3",
        "vix_return_ma_5",
        "tlt_return",
        "tlt_return_ma_3",
        "tlt_return_ma_5",
        "gld_return",
        "gld_return_ma_3",
        "gld_return_ma_5",
        "qqq_return",
        "iwm_return"
    ]
elif TICKER == "QQQ":
    FEATURES += [
        "vix_return",
        "vix_return_ma_3",
        "vix_return_ma_5",
        "tlt_return",
        "tlt_return_ma_3",
        "tlt_return_ma_5",
        "gld_return",
        "gld_return_ma_3",
        "gld_return_ma_5",
        "spy_return",
        "iwm_return"
    ]
elif TICKER == "IWM":
    FEATURES += [
        "vix_return",
        "vix_return_ma_3",
        "vix_return_ma_5",
        "tlt_return",
        "tlt_return_ma_3",
        "tlt_return_ma_5",
        "gld_return",
        "gld_return_ma_3",
        "gld_return_ma_5",
        "spy_return",
        "qqq_return"
    ]

X_train = train[FEATURES]
y_train = train["label"]

X_test = test[FEATURES]
y_test = test["label"]

# ====== Scaling ======
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====== Train Model ======
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# ====== Predictions ======
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ====== Evaluation ======
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"===== {TICKER} Logistic Regression Enhanced =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

print("Predicted class distribution:")
print(pd.Series(y_pred).value_counts(normalize=True))


VIZ_DIR = os.path.join(PROJECT_ROOT, "data","images","logistic_enhanced")
os.makedirs(VIZ_DIR, exist_ok=True)
metrics, files = plot_model_results(
    y_test=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    model=model,
    feature_names=FEATURES,
    model_name="LogisticRegressionEnhanced",
    save_dir=VIZ_DIR,
    test_data=test,
    date_col='Date'
)