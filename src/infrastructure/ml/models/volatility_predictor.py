"""Volatility Predictor - GARCH and ML implementations."""
import numpy as np
import pandas as pd
from typing import Dict
from arch import arch_model
from src.domain.ports.ml.ml_services_interface import IVolatilityPredictor, PredictionResult
from config.logging_config import get_logger

logger = get_logger(__name__)

class GARCHVolatilityPredictor(IVolatilityPredictor):
    def __init__(self, p: int = 1, q: int = 1):
        self.p, self.q = p, q
        self.fitted_model = None
    
    def train(self, returns: pd.Series, **kwargs) -> Dict:
        model = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q)
        self.fitted_model = model.fit(disp='off')
        return {'aic': float(self.fitted_model.aic)}
    
    def predict(self, horizon: int = 1) -> PredictionResult:
        forecasts = self.fitted_model.forecast(horizon=horizon)
        var_forecast = forecasts.variance.values[-1, :]
        vol_forecast = np.sqrt(var_forecast) / 100 * np.sqrt(252)
        return PredictionResult(
            predictions={f"t+{i+1}": float(v) for i, v in enumerate(vol_forecast)},
            confidence_intervals=None,
            metadata={'model': 'GARCH', 'p': self.p, 'q': self.q}
        )
    
    def forecast(self, steps: int = 30) -> pd.Series:
        result = self.predict(horizon=steps)
        return pd.Series(result.predictions)
