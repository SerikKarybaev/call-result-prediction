# ============================================================
# Train Model
# ============================================================

# Import necessary libraries

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pickle
from pathlib import Path

def train_baseline_model(X_train, y_train, config):
    """
    Training the baseline model (Logistic Regression).
    """
    
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL: Logistic Regression")
    print("=" * 60)
    
    params = config['models']['logistic_regression']
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Train score
    y_train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)
    
    print(f"\n✓ Model trained")
    print(f"  Train ROC-AUC: {train_auc:.4f}")
    
    return model


def train_random_forest(X_train, y_train, config):
    """
    Training Random Forest.
    """
    
    print("\n" + "=" * 60)
    print("TRAINING MODEL: Random Forest")
    print("=" * 60)
    
    params = config['models']['random_forest']
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Train score
    y_train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)
    
    print(f"\n✓ Model trained")
    print(f"  Train ROC-AUC: {train_auc:.4f}")
    
    return model


def train_lightgbm(X_train, y_train, config):
    """
    Training LightGBM.
    """
    
    print("\n" + "=" * 60)
    print("TRAINING MODEL: LightGBM")
    print("=" * 60)
    
    params = config['models']['lightgbm']
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    # Train score
    y_train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)
    
    print(f"\n✓ Model trained")
    print(f"  Train ROC-AUC: {train_auc:.4f}")
    
    return model


def save_model(model, encoders, config, model_name='lightgbm'):
    """
    Saving the model and encoders.
    """
    
    models_path = Path('models')
    models_path.mkdir(exist_ok=True)
    
    # Save the model
    model_file = models_path / f'{model_name}_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the encoders
    encoders_file = models_path / 'encoders.pkl'
    with open(encoders_file, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save the config
    config_file = models_path / 'model_config.pkl'
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"\n Model saved:")
    print(f"  Model: {model_file}")
    print(f"  Encoders: {encoders_file}")
    print(f"  Config: {config_file}")