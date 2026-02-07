"""Return Predictor - XGBoost, LightGBM, LSTM implementations."""
import numpy as np
import pandas as pd
from typing import Dict
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from src.domain.ports.ml.ml_services_interface import IReturnPredictor, PredictionResult
from config.logging_config import get_logger

logger = get_logger(__name__)

class XGBoostReturnPredictor(IReturnPredictor):
    def __init__(self, **params):
        self.params = params or {'n_estimators': 100, 'max_depth': 6}
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, **kwargs) -> Dict:
        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train_scaled, y_train)
        val_pred = self.model.predict(X_val_scaled)
        return {'val_mse': float(np.mean((val_pred - y_val) ** 2))}
    
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        preds = self.model.predict(self.scaler.transform(X))
        return PredictionResult(
            predictions={str(i): float(p) for i, p in enumerate(preds)},
            confidence_intervals=None,
            metadata={'model_type': 'xgboost'}
        )
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        preds = self.model.predict(self.scaler.transform(X))
        return {'mse': float(np.mean((preds - y) ** 2))}
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model, self.scaler = data['model'], data['scaler']
