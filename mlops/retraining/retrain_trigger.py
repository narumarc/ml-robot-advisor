#!/usr/bin/env python
"""Retraining trigger logic - decides when to retrain."""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

class RetrainingTrigger:
    """Determine if models should be retrained."""
    
    def __init__(self, config: dict):
        self.config = config
        self.reasons = []
    
    def check_time_based(self) -> bool:
        """Check if enough time has passed since last training."""
        # Placeholder: Check last training timestamp
        logger.info("Checking time-based trigger...")
        return False
    
    def check_drift(self, drift_score: float, threshold: float = 0.1) -> bool:
        """Check if drift exceeds threshold."""
        if drift_score > threshold:
            self.reasons.append(f"Drift detected: {drift_score:.2%} > {threshold:.2%}")
            return True
        return False
    
    def check_performance(self, current_mse: float, baseline_mse: float, threshold: float = 0.2) -> bool:
        """Check if performance degraded."""
        degradation = (current_mse - baseline_mse) / baseline_mse
        if degradation > threshold:
            self.reasons.append(f"Performance degraded: {degradation:.2%} > {threshold:.2%}")
            return True
        return False
    
    def check_data_volume(self, new_samples: int, threshold: int = 10000) -> bool:
        """Check if enough new data accumulated."""
        if new_samples > threshold:
            self.reasons.append(f"New data accumulated: {new_samples} samples")
            return True
        return False
    
    def should_retrain(self, metrics: dict) -> dict:
        """Main decision logic."""
        logger.info("Evaluating retraining triggers...")
        
        triggers = [
            self.check_time_based(),
            self.check_drift(metrics.get('drift_score', 0)),
            self.check_performance(
                metrics.get('current_mse', 0),
                metrics.get('baseline_mse', 0.001)
            ),
            self.check_data_volume(metrics.get('new_samples', 0))
        ]
        
        should_retrain = any(triggers)
        
        return {
            'should_retrain': should_retrain,
            'reasons': self.reasons,
            'timestamp': datetime.now().isoformat()
        }

def main():
    trigger = RetrainingTrigger(config={})
    
    # Example metrics
    metrics = {
        'drift_score': 0.15,
        'current_mse': 0.0015,
        'baseline_mse': 0.001,
        'new_samples': 5000
    }
    
    decision = trigger.should_retrain(metrics)
    
    logger.info(f"Should retrain: {decision['should_retrain']}")
    if decision['reasons']:
        logger.info("Reasons:")
        for reason in decision['reasons']:
            logger.info(f"  - {reason}")
    
    sys.exit(0 if not decision['should_retrain'] else 1)

if __name__ == "__main__":
    main()
