"""
Model evaluation and visualization module.
"""
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance(
    importance_series: pd.Series,
    title: str = "Feature Importance",
    save_path: str = None
):
    """
    Plot feature importance.
    
    Args:
        importance_series: Series of feature importances
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    importance_series.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Feature Correlation Heatmap",
    save_path: str = None
):
    """
    Plot correlation heatmap.
    
    Args:
        correlation_matrix: Feature correlation matrix
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics(
    metrics: Dict[str, float],
    title: str = "Model Performance Metrics",
    save_path: str = None
):
    """
    Plot model performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.barh(names, values, color="skyblue")
    plt.xlim(0, 1)
    plt.title(title)
    
    for i, v in enumerate(values):
        plt.text(v, i, f"{v:.2f}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 