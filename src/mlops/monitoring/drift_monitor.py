"""
Model Drift Detector

Detects data drift and concept drift in ML models:
- Statistical tests: KS, PSI, Wasserstein
- Distribution monitoring
- Performance degradation detection
- Automated alerting
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""
    is_drift_detected: bool
    drift_score: float
    test_name: str
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    details: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class DataDriftDetector:
    """
    Detect data drift using statistical tests.
    
    Supports multiple drift detection methods:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Wasserstein distance
    - Chi-squared test
    """
    
    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.1,
        wasserstein_threshold: float = 0.1
    ):
        """
        Initialize drift detector.
        
        Args:
            ks_threshold: P-value threshold for KS test (lower = stricter)
            psi_threshold: PSI threshold (higher = more tolerant)
            wasserstein_threshold: Wasserstein distance threshold
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
    
    def kolmogorov_smirnov_test(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftResult:
        """
        Perform Kolmogorov-Smirnov test for drift detection.
        
        Tests whether two samples come from the same distribution.
        
        Args:
            reference_data: Reference (training) data
            current_data: Current (production) data
            feature_name: Name of feature being tested
        
        Returns:
            DriftResult object
        """
        # Perform KS test
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        
        is_drift = p_value < self.ks_threshold
        
        details = {
            "feature_name": feature_name,
            "ks_statistic": float(statistic),
            "reference_mean": float(np.mean(reference_data)),
            "reference_std": float(np.std(reference_data)),
            "current_mean": float(np.mean(current_data)),
            "current_std": float(np.std(current_data))
        }
        
        if is_drift:
            logger.warning(
                f"KS Test: Drift detected for '{feature_name}'. "
                f"Statistic={statistic:.4f}, p-value={p_value:.4f}"
            )
        
        return DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(statistic),
            test_name="kolmogorov_smirnov",
            p_value=float(p_value),
            threshold=self.ks_threshold,
            details=details
        )
    
    def population_stability_index(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        n_bins: int = 10,
        feature_name: str = "feature"
    ) -> DriftResult:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between two samples.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        
        Args:
            reference_data: Reference (training) data
            current_data: Current (production) data
            n_bins: Number of bins for discretization
            feature_name: Name of feature being tested
        
        Returns:
            DriftResult object
        """
        # Create bins based on reference data
        bins = np.histogram_bin_edges(reference_data, bins=n_bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference_data, bins=bins)
        curr_counts, _ = np.histogram(current_data, bins=bins)
        
        # Convert to proportions (add small epsilon to avoid log(0))
        epsilon = 1e-10
        ref_props = (ref_counts + epsilon) / (len(reference_data) + n_bins * epsilon)
        curr_props = (curr_counts + epsilon) / (len(current_data) + n_bins * epsilon)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        is_drift = psi > self.psi_threshold
        
        # Interpret PSI value
        if psi < 0.1:
            interpretation = "No significant change"
        elif psi < 0.2:
            interpretation = "Moderate change"
        else:
            interpretation = "Significant change"
        
        details = {
            "feature_name": feature_name,
            "psi_value": float(psi),
            "interpretation": interpretation,
            "n_bins": n_bins
        }
        
        if is_drift:
            logger.warning(
                f"PSI: Drift detected for '{feature_name}'. "
                f"PSI={psi:.4f} ({interpretation})"
            )
        
        return DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(psi),
            test_name="population_stability_index",
            threshold=self.psi_threshold,
            details=details
        )
    
    def wasserstein_distance(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftResult:
        """
        Calculate Wasserstein distance (Earth Mover's Distance).
        
        Measures the minimum cost to transform one distribution to another.
        
        Args:
            reference_data: Reference (training) data
            current_data: Current (production) data
            feature_name: Name of feature being tested
        
        Returns:
            DriftResult object
        """
        # Calculate Wasserstein distance
        distance = stats.wasserstein_distance(reference_data, current_data)
        
        # Normalize by reference std (for interpretability)
        ref_std = np.std(reference_data)
        normalized_distance = distance / ref_std if ref_std > 0 else distance
        
        is_drift = normalized_distance > self.wasserstein_threshold
        
        details = {
            "feature_name": feature_name,
            "raw_distance": float(distance),
            "normalized_distance": float(normalized_distance),
            "reference_std": float(ref_std)
        }
        
        if is_drift:
            logger.warning(
                f"Wasserstein: Drift detected for '{feature_name}'. "
                f"Distance={normalized_distance:.4f}"
            )
        
        return DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(normalized_distance),
            test_name="wasserstein_distance",
            threshold=self.wasserstein_threshold,
            details=details
        )
    
    def detect_multivariate_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        method: str = "all"
    ) -> Dict[str, List[DriftResult]]:
        """
        Detect drift across multiple features.
        
        Args:
            reference_data: Reference dataframe
            current_data: Current dataframe
            method: Test method ("ks", "psi", "wasserstein", or "all")
        
        Returns:
            Dictionary mapping feature names to list of drift results
        """
        results = {}
        
        for column in reference_data.columns:
            ref_values = reference_data[column].values
            curr_values = current_data[column].values
            
            column_results = []
            
            if method in ["ks", "all"]:
                ks_result = self.kolmogorov_smirnov_test(
                    ref_values, curr_values, feature_name=column
                )
                column_results.append(ks_result)
            
            if method in ["psi", "all"]:
                psi_result = self.population_stability_index(
                    ref_values, curr_values, feature_name=column
                )
                column_results.append(psi_result)
            
            if method in ["wasserstein", "all"]:
                wass_result = self.wasserstein_distance(
                    ref_values, curr_values, feature_name=column
                )
                column_results.append(wass_result)
            
            results[column] = column_results
        
        # Summary
        total_tests = sum(len(tests) for tests in results.values())
        drift_detected = sum(
            1 for tests in results.values() 
            for test in tests if test.is_drift_detected
        )
        
        logger.info(
            f"Multivariate drift detection: {drift_detected}/{total_tests} tests detected drift"
        )
        
        return results


class ConceptDriftDetector:
    """
    Detect concept drift (model performance degradation).
    
    Monitors model prediction quality over time.
    """
    
    def __init__(
        self,
        performance_threshold: float = 0.05,
        window_size: int = 100
    ):
        """
        Initialize concept drift detector.
        
        Args:
            performance_threshold: Acceptable performance drop (e.g., 5%)
            window_size: Window size for rolling metrics
        """
        self.performance_threshold = performance_threshold
        self.window_size = window_size
    
    def detect_performance_drift(
        self,
        reference_predictions: np.ndarray,
        reference_actuals: np.ndarray,
        current_predictions: np.ndarray,
        current_actuals: np.ndarray,
        metric: str = "mae"
    ) -> DriftResult:
        """
        Detect concept drift by comparing model performance.
        
        Args:
            reference_predictions: Predictions on reference data
            reference_actuals: Actual values for reference data
            current_predictions: Predictions on current data
            current_actuals: Actual values for current data
            metric: Performance metric ("mae", "rmse", "r2")
        
        Returns:
            DriftResult object
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Calculate reference performance
        if metric == "mae":
            ref_perf = mean_absolute_error(reference_actuals, reference_predictions)
            curr_perf = mean_absolute_error(current_actuals, current_predictions)
            lower_is_better = True
        elif metric == "rmse":
            ref_perf = np.sqrt(mean_squared_error(reference_actuals, reference_predictions))
            curr_perf = np.sqrt(mean_squared_error(current_actuals, current_predictions))
            lower_is_better = True
        elif metric == "r2":
            ref_perf = r2_score(reference_actuals, reference_predictions)
            curr_perf = r2_score(current_actuals, current_predictions)
            lower_is_better = False
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Calculate performance change
        if lower_is_better:
            performance_change = (curr_perf - ref_perf) / ref_perf
        else:
            performance_change = (ref_perf - curr_perf) / ref_perf
        
        is_drift = performance_change > self.performance_threshold
        
        details = {
            "metric": metric,
            "reference_performance": float(ref_perf),
            "current_performance": float(curr_perf),
            "performance_change_pct": float(performance_change * 100),
            "lower_is_better": lower_is_better
        }
        
        if is_drift:
            logger.warning(
                f"Performance drift detected. {metric.upper()} degraded by "
                f"{performance_change*100:.2f}% (threshold: {self.performance_threshold*100:.1f}%)"
            )
        
        return DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(abs(performance_change)),
            test_name="performance_drift",
            threshold=self.performance_threshold,
            details=details
        )
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> DriftResult:
        """
        Detect drift in prediction distributions.
        
        Useful when ground truth is not immediately available.
        
        Args:
            reference_predictions: Predictions on reference data
            current_predictions: Predictions on current data
        
        Returns:
            DriftResult object
        """
        # Use KS test on predictions
        statistic, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        
        is_drift = p_value < 0.05
        
        details = {
            "ks_statistic": float(statistic),
            "reference_pred_mean": float(np.mean(reference_predictions)),
            "reference_pred_std": float(np.std(reference_predictions)),
            "current_pred_mean": float(np.mean(current_predictions)),
            "current_pred_std": float(np.std(current_predictions))
        }
        
        if is_drift:
            logger.warning(
                f"Prediction drift detected. KS statistic={statistic:.4f}, p-value={p_value:.4f}"
            )
        
        return DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(statistic),
            test_name="prediction_drift",
            p_value=float(p_value),
            threshold=0.05,
            details=details
        )


class DriftMonitor:
    """
    Comprehensive drift monitoring system.
    
    Combines data drift and concept drift detection.
    Maintains drift history and triggers alerts.
    """
    
    def __init__(
        self,
        data_drift_detector: DataDriftDetector,
        concept_drift_detector: ConceptDriftDetector
    ):
        """
        Initialize drift monitor.
        
        Args:
            data_drift_detector: Data drift detector instance
            concept_drift_detector: Concept drift detector instance
        """
        self.data_drift_detector = data_drift_detector
        self.concept_drift_detector = concept_drift_detector
        self.drift_history: List[DriftResult] = []
    
    def monitor(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_predictions: Optional[np.ndarray] = None,
        reference_actuals: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
        current_actuals: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Comprehensive drift monitoring.
        
        Args:
            reference_features: Reference feature data
            current_features: Current feature data
            reference_predictions: Reference predictions (optional)
            reference_actuals: Reference actuals (optional)
            current_predictions: Current predictions (optional)
            current_actuals: Current actuals (optional)
        
        Returns:
            Dictionary with monitoring results
        """
        results = {
            "timestamp": datetime.utcnow(),
            "data_drift": {},
            "concept_drift": {},
            "alerts": []
        }
        
        # Data drift detection
        logger.info("Running data drift detection...")
        data_drift_results = self.data_drift_detector.detect_multivariate_drift(
            reference_features, current_features, method="all"
        )
        results["data_drift"] = data_drift_results
        
        # Check for critical data drift
        drift_features = [
            feature for feature, tests in data_drift_results.items()
            if any(test.is_drift_detected for test in tests)
        ]
        
        if drift_features:
            alert = {
                "type": "data_drift",
                "severity": "warning",
                "message": f"Data drift detected in features: {', '.join(drift_features)}",
                "timestamp": datetime.utcnow()
            }
            results["alerts"].append(alert)
            logger.warning(alert["message"])
        
        # Concept drift detection (if predictions available)
        if (current_predictions is not None and 
            reference_predictions is not None):
            
            logger.info("Running concept drift detection...")
            
            # Prediction drift
            pred_drift = self.concept_drift_detector.detect_prediction_drift(
                reference_predictions, current_predictions
            )
            results["concept_drift"]["prediction_drift"] = pred_drift
            
            if pred_drift.is_drift_detected:
                alert = {
                    "type": "prediction_drift",
                    "severity": "warning",
                    "message": "Prediction distribution has changed significantly",
                    "timestamp": datetime.utcnow()
                }
                results["alerts"].append(alert)
            
            # Performance drift (if actuals available)
            if (current_actuals is not None and 
                reference_actuals is not None):
                
                perf_drift = self.concept_drift_detector.detect_performance_drift(
                    reference_predictions,
                    reference_actuals,
                    current_predictions,
                    current_actuals
                )
                results["concept_drift"]["performance_drift"] = perf_drift
                
                if perf_drift.is_drift_detected:
                    alert = {
                        "type": "performance_drift",
                        "severity": "critical",
                        "message": f"Model performance degradation detected: {perf_drift.details['performance_change_pct']:.2f}%",
                        "timestamp": datetime.utcnow()
                    }
                    results["alerts"].append(alert)
                    logger.critical(alert["message"])
        
        # Store in history
        for tests in data_drift_results.values():
            self.drift_history.extend(tests)
        
        return results
    
    def should_retrain(self, results: Dict) -> bool:
        """
        Decide if model should be retrained based on drift results.
        
        Args:
            results: Monitoring results from monitor()
        
        Returns:
            True if retraining is recommended
        """
        # Retrain if performance drift detected
        if "performance_drift" in results.get("concept_drift", {}):
            if results["concept_drift"]["performance_drift"].is_drift_detected:
                logger.info("Retraining recommended: Performance drift detected")
                return True
        
        # Retrain if significant data drift in multiple features
        data_drift = results.get("data_drift", {})
        drift_count = sum(
            1 for tests in data_drift.values()
            if any(test.is_drift_detected for test in tests)
        )
        
        if drift_count >= len(data_drift) * 0.3:  # 30% of features
            logger.info(
                f"Retraining recommended: Data drift in {drift_count}/{len(data_drift)} features"
            )
            return True
        
        return False
