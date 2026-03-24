"""
Base Model Interface - Abstract Base Class for ML Models
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit


class BaseMLModel(ABC):
    """
    Interface abstraite pour tous les modèles ML.
    Tous les modèles doivent hériter de cette classe.
    """
    
    def __init__(self, **kwargs):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.params = kwargs
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dict avec métriques d'entraînement
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        
        Args:
            X: Features
            
        Returns:
            Array de prédictions
        """
        pass
    
    def train_with_cv(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Train with temporal cross-validation.
        
        Returns:
            Dict with metrics and mean
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'rmse': [],
            'mae': [],
            'direction_accuracy': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train on this fold
            self.train(X_train, y_train)
            
            # Predict
            y_pred = self.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Direction accuracy (important for trading!)
            direction_actual = (y_test > 0).astype(int)
            direction_pred = (y_pred > 0).astype(int)
            direction_acc = (direction_actual == direction_pred).mean()
            
            cv_results['rmse'].append(rmse)
            cv_results['mae'].append(mae)
            cv_results['direction_accuracy'].append(direction_acc)
        
        # Average metrics
        return {
            'cv_rmse': np.mean(cv_results['rmse']),
            'cv_mae': np.mean(cv_results['mae']),
            'cv_direction_accuracy': np.mean(cv_results['direction_accuracy']),
            'cv_rmse_std': np.std(cv_results['rmse']),
            'cv_mae_std': np.std(cv_results['mae']),
            'cv_direction_accuracy_std': np.std(cv_results['direction_accuracy'])
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return the feature_importance_ if exist.
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save"""
        import joblib
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseMLModel':
        """load a model"""
        import joblib
        data = joblib.load(filepath)
        
        instance = cls(**data['params'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.is_trained = True
        
        return instance