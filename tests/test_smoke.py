from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    required = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "Makefile",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "scripts" / "process_data.py",
        PROJECT_ROOT / "scripts" / "train_logistic.py",
        PROJECT_ROOT / "scripts" / "train_random_forest.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Missing required files: {missing}"


def test_results_metrics_schema():
    metrics_path = PROJECT_ROOT / "results" / "metrics.csv"
    assert metrics_path.exists(), "results/metrics.csv is required for report reproducibility"
    df = pd.read_csv(metrics_path)
    expected_cols = {
        "ticker",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "predicted_positive_rate",
        "roc_auc",
    }
    assert expected_cols.issubset(df.columns), "metrics.csv schema does not match expected columns"
    assert len(df) > 0, "metrics.csv should contain at least one row"


def test_pipeline_scripts_helpfully_fail_or_run():
    script = PROJECT_ROOT / "scripts" / "process_data.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    # Accept success or a clear data-file-related failure message.
    assert result.returncode == 0 or "No such file" in (result.stdout + result.stderr)
