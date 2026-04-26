import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


TICKERS = ["SPY", "QQQ", "IWM"]
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
    "volume_ma_5",
]


def safe_roc_auc(y_true: pd.Series, y_proba: pd.Series) -> float:
    # roc_auc_score fails when y_true has a single class.
    if y_true.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_proba))


def evaluate_classifier(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(
            2
            * precision_score(y_true, y_pred, zero_division=0)
            * recall_score(y_true, y_pred, zero_division=0)
            / (
                precision_score(y_true, y_pred, zero_division=0)
                + recall_score(y_true, y_pred, zero_division=0)
                + 1e-12
            )
        ),
        "predicted_positive_rate": float(pd.Series(y_pred).mean()),
        "roc_auc": safe_roc_auc(y_true, y_proba),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    importance_rows = []

    for ticker in TICKERS:
        train_path = data_dir / f"{ticker}_train.csv"
        test_path = data_dir / f"{ticker}_test.csv"

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        train_clean = train[FEATURES + ["label"]].dropna()
        test_clean = test[FEATURES + ["label"]].dropna()

        X_train = train_clean[FEATURES]
        y_train = train_clean["label"]
        X_test = test_clean[FEATURES]
        y_test = test_clean["label"]

        # Majority baseline from training-set majority class
        majority_class = int(y_train.mode().iloc[0])
        majority_pred = pd.Series([majority_class] * len(y_test))
        majority_proba = pd.Series([majority_class] * len(y_test), dtype=float)
        majority_metrics = evaluate_classifier(y_test, majority_pred, majority_proba)
        metrics_rows.append({"ticker": ticker, "model": "majority_baseline", **majority_metrics})

        # Logistic Regression (with scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train_scaled, y_train)
        log_pred = pd.Series(log_model.predict(X_test_scaled))
        log_proba = pd.Series(log_model.predict_proba(X_test_scaled)[:, 1])
        log_metrics = evaluate_classifier(y_test, log_pred, log_proba)
        metrics_rows.append({"ticker": ticker, "model": "logistic_regression", **log_metrics})
        for fname, imp in zip(FEATURES, log_model.coef_[0]):
            importance_rows.append(
                {"ticker": ticker, "model": "logistic_regression", "feature": fname, "importance": float(imp)}
            )

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_pred = pd.Series(rf_model.predict(X_test))
        rf_proba = pd.Series(rf_model.predict_proba(X_test)[:, 1])
        rf_metrics = evaluate_classifier(y_test, rf_pred, rf_proba)
        metrics_rows.append({"ticker": ticker, "model": "random_forest", **rf_metrics})
        for fname, imp in zip(FEATURES, rf_model.feature_importances_):
            importance_rows.append(
                {"ticker": ticker, "model": "random_forest", "feature": fname, "importance": float(imp)}
            )

        # Gradient Boosting
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = pd.Series(gb_model.predict(X_test))
        gb_proba = pd.Series(gb_model.predict_proba(X_test)[:, 1])
        gb_metrics = evaluate_classifier(y_test, gb_pred, gb_proba)
        metrics_rows.append({"ticker": ticker, "model": "gradient_boosting", **gb_metrics})
        for fname, imp in zip(FEATURES, gb_model.feature_importances_):
            importance_rows.append(
                {"ticker": ticker, "model": "gradient_boosting", "feature": fname, "importance": float(imp)}
            )

    metrics_df = pd.DataFrame(metrics_rows)
    importance_df = pd.DataFrame(importance_rows)

    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    print(f"Saved: {results_dir / 'metrics.csv'}")
    print(f"Saved: {results_dir / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
