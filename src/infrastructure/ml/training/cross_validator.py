"""Time Series Cross-Validation."""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from config.logging_config import get_logger

logger = get_logger(__name__)

class TimeSeriesCrossValidator:
    """Cross-validator for time series data."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        return list(tscv.split(X))
    
    def validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation."""
        splits = self.split(X, y)
        scores = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            scores.append(metrics['mse'])
        
        return {
            'mean_cv_score': float(np.mean(scores)),
            'std_cv_score': float(np.std(scores)),
            'scores': scores
        }
