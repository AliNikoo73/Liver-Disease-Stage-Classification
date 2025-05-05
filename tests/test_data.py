"""
Tests for data processing functions.
"""
import pytest
import pandas as pd
import numpy as np
from src.data import preprocess_data, remove_outliers_iqr, scale_features

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Stage': ['1', '2', '3'],
        'N_Days': [10, 20, 30],
        'Status': ['A', 'B', 'C'],
        'Drug': ['X', 'Y', 'Z'],
        'Edema': [0, 1, 0],
        'Sex': ['M', 'F', 'M'],
        'Age': [25, 35, 45],
        'Albumin': [3.5, 4.0, 3.0],
        'Ascites': [0, 1, 0]
    })

def test_preprocess_data(sample_data):
    """Test data preprocessing."""
    X, y = preprocess_data(sample_data)
    
    # Check correct columns are dropped
    assert 'Stage' not in X.columns
    assert 'N_Days' not in X.columns
    assert 'Status' not in X.columns
    assert 'Drug' not in X.columns
    assert 'Edema' not in X.columns
    assert 'Sex' not in X.columns
    
    # Check remaining columns
    assert 'Age' in X.columns
    assert 'Albumin' in X.columns
    assert 'Ascites' in X.columns
    
    # Check target variable
    assert isinstance(y, pd.Series)
    assert all(y == sample_data['Stage'])

def test_remove_outliers_iqr():
    """Test outlier removal."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 100],  # 100 is an outlier
        'B': [1, 2, 3, 4]
    })
    
    result = remove_outliers_iqr(df, ['A'])
    
    assert len(result) == 3  # Outlier should be removed
    assert 100 not in result['A'].values
    assert all(result['B'] == [1, 2, 3])  # Other column should be unchanged

def test_scale_features():
    """Test feature scaling."""
    X_train = pd.DataFrame({
        'numeric': [1, 2, 3],
        'categorical': ['A', 'B', 'C']
    })
    X_test = pd.DataFrame({
        'numeric': [4, 5, 6],
        'categorical': ['A', 'B', 'C']
    })
    
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Check numeric features are scaled
    assert np.allclose(X_train_scaled['numeric'].mean(), 0, atol=1e-10)
    assert np.allclose(X_train_scaled['numeric'].std(), 1, atol=1e-10)
    
    # Check categorical features are preserved
    assert all(X_train_scaled['categorical'] == X_train['categorical'])
    assert all(X_test_scaled['categorical'] == X_test['categorical']) 