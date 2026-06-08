"""
Call Result Prediction - Training Pipeline
===========================================
The training pipeline for the call result prediction model.

Usage:
    python train.py
"""

import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from src.data_preparation import load_and_prepare_data
from src.feature_engineering import prepare_features, split_train_test_by_time
from src.train_model import (
    train_baseline_model, 
    train_random_forest, 
    train_lightgbm,
    save_model
)
from src.evaluate_model import (
    evaluate_model,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_shap_summary,
    save_evaluation_results
)


def main():
    """
    The main function of the pipeline.
    """
    
    print("\n" + "=" * 60)
    print("CALL RESULT PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Loading the configuration
    with open('config/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 1. Loading and preparing data
    df = load_and_prepare_data(config)
    
    # 2. Splitting into train/test
    train_df, test_df = split_train_test_by_time(df, config)
    
    # 3. Feature Engineering
    X_train, y_train, encoders = prepare_features(train_df, config, fit_encoders=True)
    X_test, y_test, _ = prepare_features(test_df, config, fit_encoders=False, encoders=encoders)
    
    feature_names = X_train.columns.tolist()
    
    # 4. Training models
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Baseline
    lr_model = train_baseline_model(X_train, y_train, config)
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, config)
    
    # LightGBM
    lgbm_model = train_lightgbm(X_train, y_train, config)
    
    # 5. Evaluation
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'LightGBM': lgbm_model
    }
    
    results = {}
    
    for model_name, model in models.items():
        metrics, y_pred_proba = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred_proba
        }
    
    # 6. Comparison of models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        model_name: {
            'ROC-AUC': result['metrics']['roc_auc'],
            'PR-AUC': result['metrics']['pr_auc']
        }
        for model_name, result in results.items()
    }).T
    
    print("\n", comparison_df)
    
    # Choosing the best model (by ROC-AUC)
    best_model_name = comparison_df['ROC-AUC'].idxmax()
    best_model = models[best_model_name]
    
    print(f"\n Best model: {best_model_name}")
    print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'ROC-AUC']:.4f}")
    
    # 7. Saving the best model
    save_model(best_model, encoders, config, model_name=best_model_name.lower().replace(' ', '_'))
    
    # 8. Visualization of the best model's results
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    
    best_predictions = results[best_model_name]['predictions']
    
    # ROC curve
    fig_roc = plot_roc_curve(y_test, best_predictions, best_model_name)
    fig_roc.savefig(results_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig_roc)
    print("  ✓ ROC curve saved")
    
    # PR curve
    fig_pr = plot_precision_recall_curve(y_test, best_predictions, best_model_name)
    fig_pr.savefig(results_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig_pr)
    print("  ✓ PR curve saved")
    
    # Feature importance
    fig_fi, importance_df = plot_feature_importance(best_model, feature_names, top_n=20)
    if fig_fi:
        fig_fi.savefig(results_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close(fig_fi)
        print("  ✓ Feature importance saved")
        
        # Save importance to CSV
        importance_df.to_csv(results_path / 'feature_importance.csv', index=False)
    
    # SHAP summary
    try:
        fig_shap = plot_shap_summary(best_model, X_test, feature_names, max_display=20)
        fig_shap.savefig(results_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig_shap)
        print("  ✓ SHAP summary saved")
    except Exception as e:
        print(f"    SHAP visualization failed: {e}")
    
    # 9. Saving metrics
    save_evaluation_results(results[best_model_name]['metrics'], 
                           model_name=best_model_name.lower().replace(' ', '_'))
    
    # 10. Saving model comparison
    comparison_df.to_csv(results_path / 'model_comparison.csv')
    print(f"\n Model comparison saved to: {results_path / 'model_comparison.csv'}")
    
    print("\n" + "=" * 60)
    print("✓✓✓ TRAINING PIPELINE COMPLETE ✓✓✓")
    print("=" * 60)


if __name__ == '__main__':
    import pandas as pd  # Add pandas import here to avoid circular imports
    main()