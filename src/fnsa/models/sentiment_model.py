"""
Sentiment Model Module

ML model training pipeline for predicting stock movements based on sentiment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional


class SentimentStockPredictor:
    """
    ML model to predict stock movement from sentiment and other features.
    """
    
    def __init__(self, model_type: str = 'random_forest', 
                 hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'logistic_regression', 'gradient_boosting')
            hyperparameters: Dictionary of model hyperparameters
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.pipeline = None
        self.is_trained = False
        self.feature_names = None
        
    def _get_model(self):
        """Factory method to get the underlying sklearn model."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.hyperparameters.get('n_estimators', 100),
                max_depth=self.hyperparameters.get('max_depth', None),
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.hyperparameters.get('C', 1.0),
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.hyperparameters.get('n_estimators', 100),
                learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def create_pipeline(self):
        """Create the sklearn pipeline with preprocessing."""
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', self._get_model())
        ])
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with training metadata
        """
        if self.pipeline is None:
            self.create_pipeline()
            
        self.feature_names = X.columns.tolist()
        self.pipeline.fit(X, y)
        self.is_trained = True
        
        return {
            'model_type': self.model_type,
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'features': self.feature_names
        }
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict_proba(X)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if supported by the model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
            
        model = self.pipeline.named_steps['classifier']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
            
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


def create_training_pipeline(sentiment_df: pd.DataFrame, 
                             returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Helper to prepare data for training.
    
    Note: This is a placeholder. Actual data preparation logic should be in 
    src/fnsa/data/alignment.py prepare_ml_features function.
    """
    pass
