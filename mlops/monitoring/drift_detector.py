"""
MLOps - Drift Detection avec Evidently.
Détection de drift des données et des modèles.
"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DriftReport:
    """Rapport de drift detection."""
    drift_detected: bool
    drift_score: float
    drifted_features: list
    timestamp: str
    details: Dict


class DriftDetector:
    """
    Détecteur de drift utilisant Evidently.
    """
    
    def __init__(self, drift_threshold: float = 0.1):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Threshold pour détecter le drift (0-1)
        """
        self.drift_threshold = drift_threshold
        logger.info(f"Drift Detector initialized (threshold={drift_threshold})")
    
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Optional[Dict] = None
    ) -> DriftReport:
        """
        Detect data drift between reference and current data.
        
        Args:
            reference_data: Historical reference data
            current_data: Current production data
            column_mapping: Optional column type mapping
            
        Returns:
            DriftReport with detection results
        """
        logger.info("Detecting data drift...")
        
        try:
            # Create Evidently report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                DatasetDriftMetric()
            ])
            
            # Run report
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Extract results
            results = report.as_dict()
            
            # Parse drift metrics
            dataset_drift = results['metrics'][2]['result']
            drift_detected = dataset_drift['dataset_drift']
            drift_share = dataset_drift['drift_share']
            
            # Get drifted features
            drifted_features = []
            if 'drift_by_columns' in dataset_drift:
                for col, metrics in dataset_drift['drift_by_columns'].items():
                    if metrics.get('drift_detected', False):
                        drifted_features.append(col)
            
            logger.info(
                f"Drift detection: drift={drift_detected}, "
                f"share={drift_share:.2%}, "
                f"features={len(drifted_features)}"
            )
            
            return DriftReport(
                drift_detected=drift_detected,
                drift_score=drift_share,
                drifted_features=drifted_features,
                timestamp=datetime.now().isoformat(),
                details=dataset_drift
            )
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            raise
    
    def detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str
    ) -> Dict:
        """
        Detect drift for a specific feature.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            feature_name: Feature to check
            
        Returns:
            Dictionary with drift metrics
        """
        logger.debug(f"Checking drift for feature: {feature_name}")
        
        try:
            report = Report(metrics=[
                ColumnDriftMetric(column_name=feature_name)
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            results = report.as_dict()
            drift_result = results['metrics'][0]['result']
            
            return {
                "feature": feature_name,
                "drift_detected": drift_result.get('drift_detected', False),
                "drift_score": drift_result.get('drift_score', 0.0),
                "method": drift_result.get('stattest_name', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift for {feature_name}: {e}")
            raise
    
    def generate_html_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str = "drift_report.html"
    ) -> str:
        """
        Generate HTML drift report.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            output_path: Path to save HTML report
            
        Returns:
            Path to saved report
        """
        logger.info(f"Generating HTML drift report: {output_path}")
        
        try:
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Save report
            report.save_html(output_path)
            
            logger.info(f"HTML report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise


class ModelPerformanceMonitor:
    """
    Monitor de performance des modèles ML.
    """
    
    def __init__(self, performance_threshold: float = 0.15):
        """
        Initialize performance monitor.
        
        Args:
            performance_threshold: Threshold de dégradation (0-1)
        """
        self.performance_threshold = performance_threshold
        self.metrics_history = []
        logger.info(f"Model Performance Monitor initialized (threshold={performance_threshold})")
    
    def log_prediction(
        self,
        model_name: str,
        prediction: float,
        actual: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a model prediction.
        
        Args:
            model_name: Name of model
            prediction: Predicted value
            actual: Actual value (if available)
            metadata: Additional metadata
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prediction": prediction,
            "actual": actual,
            "metadata": metadata or {}
        }
        
        self.metrics_history.append(record)
        
        if actual is not None:
            error = abs(prediction - actual)
            logger.debug(f"Prediction logged: model={model_name}, error={error:.4f}")
    
    def check_performance_degradation(
        self,
        model_name: str,
        recent_window: int = 100
    ) -> Dict:
        """
        Check if model performance has degraded.
        
        Args:
            model_name: Model to check
            recent_window: Number of recent predictions to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Checking performance for model: {model_name}")
        
        # Filter predictions for this model
        model_predictions = [
            r for r in self.metrics_history
            if r["model_name"] == model_name and r["actual"] is not None
        ]
        
        if len(model_predictions) < recent_window:
            logger.warning(f"Insufficient data: {len(model_predictions)} predictions")
            return {
                "sufficient_data": False,
                "count": len(model_predictions)
            }
        
        # Get recent predictions
        recent = model_predictions[-recent_window:]
        
        # Calculate errors
        errors = [abs(r["prediction"] - r["actual"]) for r in recent]
        mae = sum(errors) / len(errors)
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        
        # Compare to historical baseline (if available)
        if len(model_predictions) > recent_window * 2:
            historical = model_predictions[:-recent_window]
            historical_errors = [abs(r["prediction"] - r["actual"]) for r in historical]
            historical_mae = sum(historical_errors) / len(historical_errors)
            
            degradation = (mae - historical_mae) / historical_mae if historical_mae > 0 else 0
            degraded = degradation > self.performance_threshold
            
            logger.info(
                f"Performance check: MAE={mae:.4f}, "
                f"Historical MAE={historical_mae:.4f}, "
                f"Degradation={degradation:.2%}, "
                f"Degraded={degraded}"
            )
            
            return {
                "sufficient_data": True,
                "mae": mae,
                "rmse": rmse,
                "historical_mae": historical_mae,
                "degradation": degradation,
                "degraded": degraded,
                "threshold": self.performance_threshold
            }
        else:
            logger.info(f"Current MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            return {
                "sufficient_data": True,
                "mae": mae,
                "rmse": rmse,
                "degraded": False
            }
    
    def should_retrain(
        self,
        model_name: str,
        drift_report: Optional[DriftReport] = None,
        performance_check: Optional[Dict] = None
    ) -> Dict:
        """
        Determine if model should be retrained.
        
        Args:
            model_name: Model name
            drift_report: Optional drift report
            performance_check: Optional performance check
            
        Returns:
            Dictionary with retraining decision
        """
        reasons = []
        should_retrain = False
        
        # Check drift
        if drift_report and drift_report.drift_detected:
            reasons.append(f"Data drift detected ({drift_report.drift_score:.2%})")
            should_retrain = True
        
        # Check performance
        if performance_check and performance_check.get("degraded", False):
            degradation = performance_check.get("degradation", 0)
            reasons.append(f"Performance degraded ({degradation:.2%})")
            should_retrain = True
        
        decision = {
            "model_name": model_name,
            "should_retrain": should_retrain,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat()
        }
        
        if should_retrain:
            logger.warning(f"⚠️ Retraining recommended for {model_name}: {', '.join(reasons)}")
        else:
            logger.info(f"✓ {model_name} performing well, no retraining needed")
        
        return decision
