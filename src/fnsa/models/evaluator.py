"""
Model Evaluator Module

Functions for evaluating ML model performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, List


def evaluate_classification_model(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        # Handle binary case (1D array) or multiclass (2D array)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            score = y_proba[:, 1]
        else:
            score = y_proba
            
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, score)
        except ValueError:
            metrics['roc_auc'] = None
            
    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """
    Perform K-fold cross-validation.
    
    Args:
        model: The model object (must have fit/predict methods or be an sklearn estimator)
        X: Features
        y: Target
        cv: Number of folds
        
    Returns:
        Dictionary with mean and std of scores
    """
    # If model is our custom class, use its internal pipeline if available, 
    # otherwise we can't easily use sklearn's cross_val_score directly without wrapping.
    # Assuming model is an sklearn estimator or pipeline for this function.
    
    estimator = model.pipeline if hasattr(model, 'pipeline') else model
    
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(estimator, X, y, cv=cv, scoring='f1')
    
    return {
        'cv_accuracy_mean': scores.mean(),
        'cv_accuracy_std': scores.std(),
        'cv_f1_mean': f1_scores.mean(),
        'cv_f1_std': f1_scores.std()
    }
