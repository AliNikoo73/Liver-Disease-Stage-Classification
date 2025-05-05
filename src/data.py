"""
Data loading and preprocessing module.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def load_data(filepath: str) -> pd.DataFrame:
    """Load the liver disease dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the data by dropping unnecessary columns and separating features/target."""
    X = df.drop(columns=["Stage", "N_Days", "Status", "Drug", "Edema", "Sex"])
    y = df["Stage"]
    return X, y

def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove outliers using the IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale numerical features using StandardScaler."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    bool_cols = X_train.select_dtypes(include=[np.object_]).columns
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]), 
        columns=num_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]), 
        columns=num_cols
    )
    
    # Merge scaled numeric & boolean columns
    X_train_final = pd.concat([X_train_scaled, X_train[bool_cols].reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test[bool_cols].reset_index(drop=True)], axis=1)
    
    # Impute NaNs with mode
    X_train_final.fillna(X_train_final.mode().iloc[0], inplace=True)
    X_test_final.fillna(X_train_final.mode().iloc[0], inplace=True)
    
    return X_train_final, X_test_final 