import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ====== Project Paths ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ====== Load Data (SPY only for now) ======
train = pd.read_csv(os.path.join(DATA_DIR, "SPY_train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "SPY_test.csv"))

# ====== Select Features ======
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

# Drop rows with missing values in selected features
train_clean = train[FEATURES + ["label"]].dropna()
test_clean = test[FEATURES + ["label"]].dropna()

X_train = train_clean[FEATURES]
y_train = train_clean["label"]

X_test = test_clean[FEATURES]
y_test = test_clean["label"]

# ====== Train Model ======
# Random Forest is a non-linear baseline model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ====== Predictions ======
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ====== Evaluation ======
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("===== SPY Random Forest =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

print("Predicted class distribution:")
print(pd.Series(y_pred).value_counts(normalize=True))

# ====== Feature Importance ======
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature importances:")
print(importance_df)