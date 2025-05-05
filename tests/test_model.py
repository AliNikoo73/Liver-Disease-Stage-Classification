"""
Tests for model training and prediction functions.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.model import train_model, predict, get_feature_importance

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    y = pd.Series(np.random.randint(0, 3, 100))  # 3 classes
    return X, y

def test_train_model(sample_data):
    """Test model training."""
    X, y = sample_data
    
    # Test with default parameters
    model, params = train_model(X, y)
    
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(params, dict)
    assert set(params.keys()) == {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'}
    
    # Test with custom parameters
    custom_params = {
        'n_estimators': [10],
        'max_depth': [3],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    model, params = train_model(X, y, param_grid=custom_params)
    
    assert params['n_estimators'] == 10
    assert params['max_depth'] == 3

def test_predict(sample_data):
    """Test model prediction."""
    X, y = sample_data
    
    # Train model
    model, _ = train_model(X, y)
    
    # Test predictions
    predictions = predict(model, X)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, (np.integer, int)) for pred in predictions)
    assert all(0 <= pred <= 2 for pred in predictions)  # 3 classes (0, 1, 2)

def test_get_feature_importance(sample_data):
    """Test feature importance extraction."""
    X, y = sample_data
    
    # Train model
    model, _ = train_model(X, y)
    
    # Get feature importance
    importance = get_feature_importance(model, X.columns.tolist())
    
    assert isinstance(importance, pd.Series)
    assert len(importance) == len(X.columns)
    assert all(0 <= imp <= 1 for imp in importance)
    assert importance.index.tolist() == X.columns.tolist() 