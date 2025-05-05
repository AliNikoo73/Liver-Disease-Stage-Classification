"""
Feature selection and engineering module.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List

def select_important_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    importance_threshold: float = 0.05
) -> Tuple[List[str], pd.Series]:
    """
    Select important features using RandomForest feature importance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        importance_threshold: Minimum importance threshold for feature selection
        
    Returns:
        Tuple containing list of important feature names and feature importance series
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    imp_features = pd.Series(
        rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    # Select features above threshold
    important_features = imp_features[imp_features > importance_threshold].index.tolist()
    
    return important_features, imp_features

def get_feature_correlations(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for features.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Correlation matrix
    """
    return X.corr()

def get_cumulative_importance(imp_features: pd.Series) -> pd.Series:
    """
    Calculate cumulative feature importance.
    
    Args:
        imp_features: Series of feature importances
        
    Returns:
        Cumulative importance series
    """
    return imp_features.cumsum() 