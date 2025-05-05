"""
Model training and tuning module.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator

def get_default_param_grid() -> Dict[str, list]:
    """Get default parameter grid for RandomForest."""
    return {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, list] = None,
    cv_folds: int = 5,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a RandomForest model using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for GridSearchCV
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (best model, best parameters)
    """
    if param_grid is None:
        param_grid = get_default_param_grid()
        
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=kfold,
        scoring="accuracy",
        n_jobs=-1,
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using trained model.
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        Array of predictions
    """
    return model.predict(X)

def get_feature_importance(model: RandomForestClassifier, feature_names: list) -> pd.Series:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained RandomForest model
        feature_names: List of feature names
        
    Returns:
        Series of feature importances
    """
    return pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False) 