"""Model Performance Tracker - Track predictions and performance over time."""
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)

class ModelPerformanceTracker:
    """Track model predictions and performance metrics."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.predictions_history = []
        self.metrics_history = []
    
    async def log_prediction(
        self,
        model_name: str,
        prediction: float,
        actual: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Log a single prediction."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) if actual is not None else None,
            'metadata': metadata or {}
        }
        self.predictions_history.append(record)
        logger.debug(f"Logged prediction for {model_name}: {prediction}")
    
    async def check_performance(
        self,
        model_name: str,
        threshold: float
    ) -> Dict:
        """Check if model performance is acceptable."""
        recent_predictions = [
            p for p in self.predictions_history[-100:]
            if p['model_name'] == model_name and p['actual'] is not None
        ]
        
        if len(recent_predictions) < 10:
            return {'sufficient_data': False, 'count': len(recent_predictions)}
        
        errors = [p['error'] for p in recent_predictions]
        mae = sum(errors) / len(errors)
        
        return {
            'mae': mae,
            'performance_acceptable': mae < threshold,
            'threshold': threshold,
            'n_predictions': len(recent_predictions)
        }
    
    async def should_retrain(
        self,
        model_name: str,
        criteria: Dict
    ) -> bool:
        """Determine if model should be retrained."""
        performance = await self.check_performance(
            model_name,
            criteria.get('performance_threshold', 0.05)
        )
        
        if not performance.get('sufficient_data', True):
            return False
        
        return not performance.get('performance_acceptable', True)
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of recent metrics."""
        if not self.predictions_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.predictions_history)
        df = df[df['actual'].notna()]
        
        summary = df.groupby('model_name').agg({
            'error': ['mean', 'std', 'min', 'max'],
            'prediction': 'count'
        })
        
        return summary
