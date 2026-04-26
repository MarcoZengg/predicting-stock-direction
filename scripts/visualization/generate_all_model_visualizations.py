import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from visualization_train import plot_model_results


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


def _load_train_test(data_dir: Path, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / f"{ticker}_train.csv")
    test = pd.read_csv(data_dir / f"{ticker}_test.csv")
    return train, test


def _run_logistic(train: pd.DataFrame, test: pd.DataFrame):
    X_train = train[FEATURES]
    y_train = train["label"]
    X_test = test[FEATURES]
    y_test = test["label"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, y_test, y_pred, y_proba


def _run_random_forest(train: pd.DataFrame, test: pd.DataFrame):
    train_clean = train[FEATURES + ["label"]].dropna()
    test_clean = test[FEATURES + ["label"]].dropna()

    X_train = train_clean[FEATURES]
    y_train = train_clean["label"]
    X_test = test_clean[FEATURES]
    y_test = test_clean["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, y_test, y_pred, y_proba


def _run_gradient_boosting(train: pd.DataFrame, test: pd.DataFrame):
    train_clean = train[FEATURES + ["label"]].dropna()
    test_clean = test[FEATURES + ["label"]].dropna()

    X_train = train_clean[FEATURES]
    y_train = train_clean["label"]
    X_test = test_clean[FEATURES]
    y_test = test_clean["label"]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, y_test, y_pred, y_proba


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    output_root = project_root / "data" / "images" / "all_models"
    output_root.mkdir(parents=True, exist_ok=True)

    runners = {
        "logistic_regression": _run_logistic,
        "random_forest": _run_random_forest,
        "gradient_boosting": _run_gradient_boosting,
    }

    for ticker in TICKERS:
        train, test = _load_train_test(data_dir, ticker)
        for model_name, runner in runners.items():
            model, y_test, y_pred, y_proba = runner(train, test)
            save_dir = output_root / ticker.lower() / model_name
            os.makedirs(save_dir, exist_ok=True)

            full_model_name = f"{ticker}_{model_name}"
            plot_model_results(
                y_test=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                model=model,
                feature_names=FEATURES,
                model_name=full_model_name,
                save_dir=str(save_dir),
                test_data=test,
                date_col="Date",
            )
            print(f"Generated visualizations for {full_model_name} -> {save_dir}")


if __name__ == "__main__":
    main()
