"""Model Trainer - Generic training logic for ML models."""
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
import mlflow
from config.logging_config import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Generic trainer for ML models with MLflow integration."""
    
    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    async def train_model(
        self,
        model_name: str,
        training_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a model with given config."""
        logger.info(f"Training {model_name} with config: {config}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(config)
            
            # Extract features and target
            X = training_data[config['feature_cols']]
            y = training_data[config['target_col']]
            
            # Initialize model based on type
            model_class = config['model_class']
            model = model_class(**config.get('model_params', {}))
            
            # Train
            metrics = model.train(X, y, **config.get('train_params', {}))
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            logger.info(f"Training complete: {metrics}")
            return {'model': model, 'metrics': metrics, 'run_id': mlflow.active_run().info.run_id}
    
    async def hyperparameter_tuning(
        self,
        model_name: str,
        training_data: pd.DataFrame,
        param_grid: Dict
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_params = None
        best_score = float('inf')
        
        # Grid search (simplified)
        import itertools
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        for params in param_combinations[:10]:  # Limit to 10 for demo
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                # Train and evaluate (simplified)
                # ... training logic here ...
                score = 0.0  # Placeholder
                mlflow.log_metric('score', score)
                
                if score < best_score:
                    best_score = score
                    best_params = params
        
        logger.info(f"Best params: {best_params}, Best score: {best_score}")
        return {'best_params': best_params, 'best_score': best_score}
    
    async def validate_model(
        self,
        model: Any,
        validation_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Validate model on holdout data."""
        X_val = validation_data.drop('target', axis=1)
        y_val = validation_data['target']
        
        metrics = model.evaluate(X_val, y_val)
        logger.info(f"Validation metrics: {metrics}")
        return metrics
