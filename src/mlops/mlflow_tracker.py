"""
MLflow Tracker - Unified MLflow Integration
"""
import mlflow
from typing import Dict, Any, Optional
from datetime import datetime

from src.domain.entities.portfolio import Portfolio


class MLflowTracker:
    """
        MLflow Tracker - Unified MLflow Integration

    Centralized manager for all MLflow tracking.

    Features:

    Log experiments

    Track metrics

    Save models

    Manage artifacts
    """
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: URI of the MLflow server (e.g., "http://localhost:5000")
            experiment_name: Name of the experiment
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new Mlflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log of one metrics"""
        mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: Any, artifact_path: str, **kwargs):
        """Save a model"""
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_portfolio(self, portfolio: Portfolio):
        """
        Log an optimized portfolio in MLflow.

        Logs:

            All weights

            Performance metrics

            Metadata
        """
        # Log weights as metrics
        for ticker, weight in portfolio.weights.items():
            mlflow.log_metric(f"weight_{ticker}", weight)
        
        # Log performance metrics
        if portfolio.sharpe_ratio:
            mlflow.log_metric("sharpe_ratio", portfolio.sharpe_ratio)
        if portfolio.expected_return:
            mlflow.log_metric("expected_return", portfolio.expected_return)
        if portfolio.volatility:
            mlflow.log_metric("volatility", portfolio.volatility)
        
        # Log metadata as params
        mlflow.log_params({
            "portfolio_id": portfolio.portfolio_id,
            "strategy": portfolio.strategy or "unknown",
            "n_assets": len(portfolio.tickers),
            "tickers": ",".join(portfolio.tickers)
        })
        
        # Set tags
        mlflow.set_tags({
            "portfolio_name": portfolio.name,
            "created_at": portfolio.created_at.isoformat(),
            "model_type": "portfolio_optimization"
        })
    
    def log_ml_training(
        self,
        model_type: str,
        ticker: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ):
        """
        Log the training of a ML model.
        """
        # Log params
        mlflow.log_params({
            "model_type": model_type,
            "ticker": ticker,
            **params
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        if feature_importance:
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"importance_{feature}", importance)
        
        # Set tags
        mlflow.set_tags({
            "task": "ml_training",
            "ticker": ticker,
            "model": model_type
        })
    
    def log_drift_detection(
        self,
        drift_score: float,
        drift_detected: bool,
        affected_features: list
    ):
        """
        Log results of the drift detection.
        """
        mlflow.log_metrics({
            "drift_score": drift_score,
            "drift_detected": float(drift_detected)
        })
        
        mlflow.log_params({
            "affected_features": ",".join(affected_features),
            "n_drifted_features": len(affected_features)
        })
        
        mlflow.set_tag("task", "drift_detection")
    
    def log_backtest(
        self,
        strategy: str,
        metrics: Dict[str, float]
    ):
        """
        Log backtesting results.
        """
        mlflow.log_params({"strategy": strategy})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("task", "backtest")
    
    def get_best_run(self, metric: str = "sharpe_ratio", ascending: bool = False):
        """
        Retrieve the best run based on a metric.

        Args:

            metric: Metric to optimize

            ascending: If True, minimizes; otherwise maximizes

        Returns:

            MLflow run
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) == 0:
            return None
        
        return runs.iloc[0]
    
    def compare_runs(self, run_ids: list) -> Dict:
        """
        Compare multiple runs.

        Returns:

            Dictionary with metric comparisons
        """
        runs = mlflow.search_runs(
            filter_string=f"run_id IN {tuple(run_ids)}"
        )
        
        return runs.to_dict('records')