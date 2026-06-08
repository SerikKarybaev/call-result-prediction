# Importing necessary libraries for model evaluation and visualization

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    auc,
    confusion_matrix,
    classification_report
)
import shap
from pathlib import Path
import json

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate the model on the test set and print key metrics and reports.
    """
    
    print("\n" + "=" * 60)
    print(f"EVALUATION: {model_name}")
    print("=" * 60)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n Test Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    
    # Classification report
    print(f"\n Classification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=['No Result', 'Result']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]:,}")
    print(f"  False Positives: {cm[0, 1]:,}")
    print(f"  False Negatives: {cm[1, 0]:,}")
    print(f"  True Positives:  {cm[1, 1]:,}")
    
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics, y_pred_proba


def plot_roc_curve(y_test, y_pred_proba, model_name='Model'):
    """
    Roc curve visualization
    """
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    return plt.gcf()


def plot_precision_recall_curve(y_test, y_pred_proba, model_name='Model'):
    """
    Precision-Recall curve visualization.
    """
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    return plt.gcf()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Feature importance visualization.
    """
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("⚠️  Model doesn't have feature importances")
        return None
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Take top N features
    top_features = feature_importance_df.head(top_n)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt.gcf(), feature_importance_df


def plot_shap_summary(model, X_test, feature_names, max_display=20):
    """
    SHAP summary plot for model interpretability.
    """
    
    print("\n🔍 Computing SHAP values (this may take a while)...")
    
    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_test)
    
    # Compute SHAP values (take a sample for speed)
    X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # If binary classification, take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                      max_display=max_display, show=False)
    plt.tight_layout()
    
    return plt.gcf()


def save_evaluation_results(metrics, model_name='lightgbm'):
    """
    Save evaluation results.
    """
    
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_file = results_path / f'{model_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Metrics saved to: {metrics_file}")