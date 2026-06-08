# ============================================================
# Feature Engineering
# ============================================================

# import libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_features(df, config, fit_encoders=True, encoders=None):
    """
    Preparation of features for the model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    config : dict
        Configuration
    fit_encoders : bool
        Train new encoders (True for train, False for test)
    encoders : dict
        Pre-trained encoders (for test set)
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    encoders : dict
        Encoders (for saving)
    """
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # Target variable
    target = config['target']
    y = df[target]
    
    # Выбираем признаки
    categorical_features = config['features']['categorical']
    numerical_features = config['features']['numerical']
    
    print(f"\n Features:")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Numerical: {len(numerical_features)}")
    
    # Check for missing features
    missing_features = []
    for feat in categorical_features + numerical_features:
        if feat not in df.columns:
            missing_features.append(feat)
    
    if missing_features:
        print(f"\n  WARNING: Missing features: {missing_features}")
        # Delete missing features
        categorical_features = [f for f in categorical_features if f in df.columns]
        numerical_features = [f for f in numerical_features if f in df.columns]
    
    # Preparation of categorical features
    if fit_encoders:
        encoders = {}
    
    X_categorical = pd.DataFrame(index=df.index)
    
    for feat in categorical_features:
        if fit_encoders:
            le = LabelEncoder()
            X_categorical[feat] = le.fit_transform(df[feat].astype(str))
            encoders[feat] = le
        else:
            # Use pre-trained encoder
            le = encoders[feat]
            # Handle unknown categories
            def safe_transform(x):
                try:
                    return le.transform([str(x)])[0]
                except ValueError:
                    # Unknown category → assign -1
                    return -1
            
            X_categorical[feat] = df[feat].apply(safe_transform)
    
    # Numerical features
    X_numerical = df[numerical_features].copy()
    
    # Fill missing values
    X_numerical = X_numerical.fillna(X_numerical.median())
    
    # Merge features
    X = pd.concat([X_numerical, X_categorical], axis=1)
    
    print(f"\n✓ Feature matrix shape: {X.shape}")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    
    # Check for missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n  WARNING: {missing_count} missing values detected!")
    
    return X, y, encoders


def split_train_test_by_time(df, config):
    """
    Split train/test by time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data
    config : dict
        Configuration
    
    Returns:
    --------
    train_df, test_df : pd.DataFrame
        Train and test sets
    """
    
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)
    
    test_start_date = pd.Timestamp(config['train_test_split']['test_start_date'])
    
    train_df = df[df['start_time'] < test_start_date].copy()
    test_df = df[df['start_time'] >= test_start_date].copy()
    
    print(f"\n Split by time (test starts: {test_start_date.date()}):")
    print(f"  Train: {len(train_df):,} samples ({train_df['start_time'].min().date()} to {train_df['start_time'].max().date()})")
    print(f"  Test:  {len(test_df):,} samples ({test_df['start_time'].min().date()} to {test_df['start_time'].max().date()})")
    
    print(f"\n  Train target distribution:")
    print(f"    Positive: {train_df['is_result'].sum():,} ({train_df['is_result'].mean():.2%})")
    print(f"  Test target distribution:")
    print(f"    Positive: {test_df['is_result'].sum():,} ({test_df['is_result'].mean():.2%})")
    
    return train_df, test_df