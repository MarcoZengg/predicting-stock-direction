import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)

# ====== Project Paths ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VIZ_DIR = os.path.join(PROJECT_ROOT, "data","images","logistic","visualizations")

# Create visualizations directory if it doesn't exist
os.makedirs(VIZ_DIR, exist_ok=True)

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


# ====== Set Style ======
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================
# 1. CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Down', 'Up'], 
            yticklabels=['Down', 'Up'],
            annot_kws={'size': 14})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix - SPY Direction Prediction\nAccuracy: {accuracy:.3f}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '1_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 2. ROC CURVE
# ============================================
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - SPY Direction Prediction', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '2_roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 3. FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================
coef_df = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=True)  # ascending for horizontal bar

plt.figure(figsize=(10, 7))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df['Coefficient']]
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black', alpha=0.8)
plt.xlabel('Coefficient Magnitude', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance - Logistic Regression\n(Green = Positive Impact on Up Prediction)', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '3_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 4. PREDICTION PROBABILITY DISTRIBUTION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual Down days (label = 0)
axes[0].hist(y_proba[y_test == 0], bins=30, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.5)
axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Boundary')
axes[0].set_xlabel('Predicted Probability of Up', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title(f'Actual Down Days (n={sum(y_test==0)})', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Actual Up days (label = 1)
axes[1].hist(y_proba[y_test == 1], bins=30, alpha=0.7, color='#2ecc71', edgecolor='black', linewidth=0.5)
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Boundary')
axes[1].set_xlabel('Predicted Probability of Up', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title(f'Actual Up Days (n={sum(y_test==1)})', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Predicted Probability Distributions by Actual Class', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '4_probability_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 5. PRECISION-RECALL CURVE
# ============================================
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, 'b-', linewidth=2, label='Logistic Regression')
plt.fill_between(recall_vals, precision_vals, alpha=0.2, color='blue')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '5_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 6. TIME SERIES OF PREDICTIONS
# ============================================
# Check if test set has a date column
date_col = None
for col in ['date', 'Date', 'datetime', 'index']:
    if col in test.columns:
        date_col = col
        break

if date_col:
    test_with_preds = test.copy()
    test_with_preds['pred_proba'] = y_proba
    test_with_preds['pred_label'] = y_pred
    test_with_preds[date_col] = pd.to_datetime(test_with_preds[date_col])
    
    plt.figure(figsize=(14, 6))
    
    # Plot prediction probability
    plt.plot(test_with_preds[date_col], test_with_preds['pred_proba'], 'b-', alpha=0.7, linewidth=0.8, label='Predicted Probability (Up)')
    
    # Fill between regions
    plt.fill_between(test_with_preds[date_col], 0.5, test_with_preds['pred_proba'], 
                     where=(test_with_preds['pred_proba'] >= 0.5), 
                     color='#2ecc71', alpha=0.3, label='Predicted Up')
    plt.fill_between(test_with_preds[date_col], test_with_preds['pred_proba'], 0.5, 
                     where=(test_with_preds['pred_proba'] < 0.5), 
                     color='#e74c3c', alpha=0.3, label='Predicted Down')
    
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Decision Boundary')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Predicted Probability of Up', fontsize=12)
    plt.title('SPY Direction Predictions Over Time', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '6_time_series_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
else:
    # Create a simple index plot if no date column
    plt.figure(figsize=(14, 6))
    plt.plot(y_proba, 'b-', alpha=0.7, linewidth=0.8, label='Predicted Probability (Up)')
    plt.fill_between(range(len(y_proba)), 0.5, y_proba, 
                     where=(y_proba >= 0.5), color='#2ecc71', alpha=0.3, label='Predicted Up')
    plt.fill_between(range(len(y_proba)), y_proba, 0.5, 
                     where=(y_proba < 0.5), color='#e74c3c', alpha=0.3, label='Predicted Down')
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Predicted Probability of Up', fontsize=12)
    plt.title('SPY Direction Predictions Over Time', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '6_time_series_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# 7. CLASSIFICATION REPORT HEATMAP
# ============================================
report = classification_report(y_test, y_pred, output_dict=True, target_names=['Down', 'Up'])
report_df = pd.DataFrame(report).transpose()

# Select only relevant metrics for heatmap
metrics_to_plot = ['precision', 'recall', 'f1-score']
plot_df = report_df.loc[['Down', 'Up', 'accuracy'], metrics_to_plot].copy()
plot_df.loc['macro avg'] = report_df.loc['macro avg', metrics_to_plot]
plot_df.loc['weighted avg'] = report_df.loc['weighted avg', metrics_to_plot]

plt.figure(figsize=(8, 6))
sns.heatmap(plot_df.astype(float), annot=True, cmap='RdYlGn', fmt='.3f', linewidths=0.5, cbar_kws={'label': 'Score'})
plt.title('Classification Report - SPY Direction Prediction', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '7_classification_report.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# BONUS: COEFFICIENT BAR PLOT (Alternative View)
# ============================================
coef_df_desc = coef_df.sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df_desc['Coefficient']]
plt.bar(coef_df_desc['Feature'], coef_df_desc['Coefficient'], color=colors, edgecolor='black', alpha=0.7)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Coefficient', fontsize=12)
plt.title('Feature Coefficients (Absolute Importance)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '8_coefficient_bar_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("VISUALIZATION SUMMARY")
print("="*50)
print(f"All visualizations saved to: {VIZ_DIR}")
print("\nFiles created:")
for i, f in enumerate(os.listdir(VIZ_DIR), 1):
    if f.endswith('.png'):
        print(f"  {f}")