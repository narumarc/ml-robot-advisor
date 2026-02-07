"""
Hyperparameter Tuner - Optimisation des hyperparamètres.
"""
from typing import Dict, Any, List
import optuna
from sklearn.model_selection import cross_val_score
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, n_trials: int = 50, direction: str = "minimize"):
        self.n_trials = n_trials
        self.direction = direction
        logger.info(f"Hyperparameter Tuner initialized (n_trials={n_trials})")
    
    def tune(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Returns:
            Best parameters and study results
        """
        def objective(trial):
            # Sample parameters
            params = {}
            for name, config in param_space.items():
                if config['type'] == 'int':
                    params[name] = trial.suggest_int(name, config['low'], config['high'])
                elif config['type'] == 'float':
                    params[name] = trial.suggest_float(name, config['low'], config['high'])
                elif config['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, config['choices'])
            
            # Create and evaluate model
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            return scores.mean()
        
        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
