import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)

def plot_model_results(y_test, y_pred, y_proba, model=None, feature_names=None, 
                       model_name="Model", save_dir="images/visualizations", test_data=None, date_col=None,
                        show_thresholds=True, n_thresholds=10):
    """
    Generate comprehensive visualizations for classification model results.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities (for positive class)
    model : object, optional
        Trained model with .coef_ or .feature_importances_ attribute
    feature_names : list, optional
        List of feature names (required if model is provided)
    model_name : str
        Name of the model (used in titles and filenames)
    save_dir : str
        Directory to save visualizations
    test_data : DataFrame, optional
        Test data containing date column for time series plot
    date_col : str, optional
        Name of date column in test_data
    
    Returns:
    --------
    dict: Dictionary containing metrics and file paths
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }
    
    print(f"\n===== {model_name} Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("Set2")
    
    saved_files = []
    
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
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy:.3f}', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_1_confusion_matrix.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)
    
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
    plt.title(f'{model_name} - ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_2_roc_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)
    
    # ============================================
    # 3. FEATURE IMPORTANCE / COEFFICIENTS
    # ============================================
    if model is not None and feature_names is not None:
        # Check if model has coefficients (Logistic Regression, Linear models)
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', key=abs, ascending=True)
            
            plt.figure(figsize=(10, 7))
            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in imp_df['Importance']]
            plt.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='black', alpha=0.8)
            plt.xlabel('Coefficient Magnitude', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'{model_name} - Feature Coefficients\n(Green = Positive Impact on Up Prediction)', fontsize=12)
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            filepath = os.path.join(save_dir, f'{model_name.lower()}_3_feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filepath)
            
        # Check if model has feature_importances_ (Tree-based models)
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            plt.figure(figsize=(10, 7))
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(imp_df)))
            plt.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='black', alpha=0.8)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'{model_name} - Feature Importance', fontsize=14)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            filepath = os.path.join(save_dir, f'{model_name.lower()}_3_feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filepath)
    
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
    
    plt.suptitle(f'{model_name} - Predicted Probability Distributions by Actual Class', fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_4_probability_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)
    
    # ============================================
    # 5. PRECISION-RECALL CURVE
    # ============================================
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve
    plt.plot(recall_vals, precision_vals, 'b-', linewidth=2, label=f'{model_name} (AUC-PR = {np.trapezoid(precision_vals, recall_vals):.3f})')
    plt.fill_between(recall_vals, precision_vals, alpha=0.2, color='blue')
    
    # Mark optimal threshold (max F1)
    plt.scatter(recall_vals[optimal_idx], precision_vals[optimal_idx], 
               color='red', s=150, zorder=5, marker='*',
               label=f'Optimal (F1={f1_scores[optimal_idx]:.3f}, thresh={optimal_threshold:.3f})')
    
    # Add threshold annotations
    if show_thresholds:
        # Select thresholds to display
        threshold_indices = np.linspace(0, len(thresholds_pr)-1, min(n_thresholds, len(thresholds_pr)), dtype=int)
        
        for idx in threshold_indices:
            if idx < len(thresholds_pr):
                thresh = thresholds_pr[idx]
                rec = recall_vals[idx]
                prec = precision_vals[idx]
                
                # Skip if point is too close to optimal to avoid clutter
                if abs(idx - optimal_idx) > len(thresholds_pr) // n_thresholds:
                    # Add small dot
                    plt.scatter(rec, prec, color='orange', s=30, alpha=0.6, zorder=3)
                    # Add threshold label
                    plt.annotate(f'{thresh:.2f}', 
                                xy=(rec, prec),
                                xytext=(5, 5), 
                                textcoords='offset points',
                                fontsize=7,
                                alpha=0.6,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))
        
        # Also add key thresholds (0.3, 0.4, 0.5, 0.6, 0.7)
        key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for key_thresh in key_thresholds:
            # Find closest threshold
            close_idx = np.argmin(np.abs(thresholds_pr - key_thresh))
            if close_idx < len(thresholds_pr):
                rec = recall_vals[close_idx]
                prec = precision_vals[close_idx]
                plt.scatter(rec, prec, color='purple', s=50, zorder=4, marker='s',
                           label=f'Threshold={key_thresh}' if key_thresh == 0.5 else "")
                plt.annotate(f'{key_thresh}', 
                            xy=(rec, prec),
                            xytext=(10, -8), 
                            textcoords='offset points',
                            fontsize=8,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title(f'{model_name} - Precision-Recall Curve with Thresholds', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_5_precision_recall_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)
    
    # ============================================
    # 6. TIME SERIES OF PREDICTIONS
    # ============================================
    if test_data is not None:
        # Find date column if not specified
        if date_col is None:
            for col in ['date', 'Date', 'datetime', 'index']:
                if col in test_data.columns:
                    date_col = col
                    break
        
        if date_col and date_col in test_data.columns:
            test_with_preds = test_data.copy()
            test_with_preds['pred_proba'] = y_proba
            test_with_preds['pred_label'] = y_pred
            test_with_preds[date_col] = pd.to_datetime(test_with_preds[date_col])
            
            plt.figure(figsize=(14, 6))
            plt.plot(test_with_preds[date_col], test_with_preds['pred_proba'], 'b-', alpha=0.7, linewidth=0.8, label='Predicted Probability (Up)')
            plt.fill_between(test_with_preds[date_col], 0.5, test_with_preds['pred_proba'], 
                             where=(test_with_preds['pred_proba'] >= 0.5), 
                             color='#2ecc71', alpha=0.3, label='Predicted Up')
            plt.fill_between(test_with_preds[date_col], test_with_preds['pred_proba'], 0.5, 
                             where=(test_with_preds['pred_proba'] < 0.5), 
                             color='#e74c3c', alpha=0.3, label='Predicted Down')
            plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Decision Boundary')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Predicted Probability of Up', fontsize=12)
            plt.title(f'{model_name} - Direction Predictions Over Time', fontsize=14)
            plt.legend(loc='upper left', fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            filepath = os.path.join(save_dir, f'{model_name.lower()}_6_time_series_predictions.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filepath)
    
    # ============================================
    # 7. CLASSIFICATION REPORT HEATMAP
    # ============================================
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Down', 'Up'])
    report_df = pd.DataFrame(report).transpose()
    
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    plot_df = report_df.loc[['Down', 'Up', 'accuracy'], metrics_to_plot].copy()
    plot_df.loc['macro avg'] = report_df.loc['macro avg', metrics_to_plot]
    plot_df.loc['weighted avg'] = report_df.loc['weighted avg', metrics_to_plot]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(plot_df.astype(float), annot=True, cmap='RdYlGn', fmt='.3f', linewidths=0.5, cbar_kws={'label': 'Score'})
    plt.title(f'{model_name} - Classification Report', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_7_classification_report.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)
    
    # ============================================
    # 8. COEFFICIENT BAR PLOT (Alternative View)
    # ============================================
    if model is not None and feature_names is not None and hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df['Coefficient']]
        plt.bar(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black', alpha=0.7)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Coefficient', fontsize=12)
        plt.title(f'{model_name} - Feature Coefficients (Absolute Importance)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(save_dir, f'{model_name.lower()}_8_coefficient_bar_plot.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*50)
    print(f"{model_name} VISUALIZATION SUMMARY")
    print("="*50)
    print(f"All visualizations saved to: {save_dir}")
    print(f"\nFiles created: {len(saved_files)}")
    
    return metrics, saved_files


# ============================================
# USAGE EXAMPLES
# ============================================

# Example 1: Logistic Regression
# from visualization_train import plot_model_results
# VIZ_DIR = os.path.join(PROJECT_ROOT, "data","images","logistic")
# os.makedirs(VIZ_DIR, exist_ok=True)
# metrics, files = plot_model_results(
#     y_test=y_test,
#     y_pred=y_pred,
#     y_proba=y_proba,
#     model=model,
#     feature_names=FEATURES,
#     model_name="LogisticRegression",
#     save_dir=VIZ_DIR,
#     test_data=test,
#     date_col='Date'
# )

# Example 2: XGBoost
# from visualization_train import plot_model_results
# VIZ_DIR = os.path.join(PROJECT_ROOT, "data","images","xgboost")
# os.makedirs(VIZ_DIR, exist_ok=True)
# metrics, files = plot_model_results(
#     y_test=y_processed,
#     y_pred=y_pred_all,
#     y_proba=y_proba_all,
#     model=model,
#     feature_names=FEATURES,
#     model_name="XGBoost",
#     save_dir=VIZ_DIR,
#     test_data=processed_clean,
#     date_col='Date'
# )

# Example 3: Without model (just predictions)
# metrics, files = plot_model_results(
#     y_test=y_test,
#     y_pred=y_pred,
#     y_proba=y_proba,
#     model_name="MyModel",
#     save_dir="visualizations"
# )