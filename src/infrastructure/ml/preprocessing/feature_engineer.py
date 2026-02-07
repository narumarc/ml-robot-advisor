"""Feature Engineering for financial time series."""
import pandas as pd
import numpy as np
from typing import List
from config.logging_config import get_logger

logger = get_logger(__name__)

class FinancialFeatureEngineer:
    """Engineer features for financial ML models."""
    
    def create_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators."""
        features = pd.DataFrame(index=prices.index)
        
        # Returns
        features['return_1d'] = prices.pct_change(1)
        features['return_5d'] = prices.pct_change(5)
        features['return_20d'] = prices.pct_change(20)
        
        # Volatility
        features['volatility_20d'] = prices.pct_change().rolling(20).std()
        
        # Moving averages
        features['sma_20'] = prices.rolling(20).mean() / prices
        features['sma_50'] = prices.rolling(50).mean() / prices
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        
        # Bollinger Bands
        bb_mid = prices.rolling(20).mean()
        bb_std = prices.rolling(20).std()
        features['bb_position'] = (prices - bb_mid) / (2 * bb_std)
        
        return features.dropna()
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lagged features."""
        features = data.copy()
        for col in data.columns:
            for lag in lags:
                features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        return features.dropna()
