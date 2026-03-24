"""
Random Forest Model Implementation
"""
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestRegressor

from src.infrastructure.ml.models.base_model import BaseMLModel


class RandomForestModel(BaseMLModel):
    """
    Random Forest pour prédiction de rendements.
    
    Hyperparamètres optimisés pour séries temporelles financières.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Training metrics
        y_pred = self.model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Direction accuracy
        direction_actual = (y > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_acc = (direction_actual == direction_pred).mean()
        
        return {
            'train_rmse': rmse,
            'train_mae': mae,
            'train_r2': r2,
            'train_direction_accuracy': direction_acc
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future return.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """
        Prédit avec intervalle de confiance (via arbres individuels).
        
        Returns:
            (predictions, std_dev)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        # Prédictions de chaque arbre
        tree_predictions = np.array([
            tree.predict(X) 
            for tree in self.model.estimators_
        ])
        
        # mean and std
        predictions = tree_predictions.mean(axis=0)
        std_dev = tree_predictions.std(axis=0)
        
        return predictions, std_dev