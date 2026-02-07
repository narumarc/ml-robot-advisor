#!/usr/bin/env python
"""
Automated retraining pipeline.
Checks drift and performance, then retrains if needed.
Usage: python mlops/retraining/auto_retrain_pipeline.py
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def run_command(cmd: list, description: str) -> tuple:
    """Run a command and return (success, output)."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        success = result.returncode == 0
        
        if success:
            logger.info(f"✅ {description} - SUCCESS")
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Error: {result.stderr}")
        
        return success, result.stdout
        
    except Exception as e:
        logger.error(f"❌ {description} - ERROR: {e}")
        return False, str(e)


def check_drift() -> bool:
    """Check if drift is detected."""
    logger.info("Checking for data drift...")
    
    cmd = [
        'python',
        'mlops/monitoring/check_drift.py',
        '--reference', 'data/reference.csv',
        '--current', 'data/production.csv'
    ]
    
    success, _ = run_command(cmd, "Drift Detection")
    
    # If check_drift exits with 1, drift was detected
    drift_detected = not success
    
    if drift_detected:
        logger.warning("⚠️  Drift detected!")
    else:
        logger.info("✅ No drift detected")
    
    return drift_detected


def check_performance() -> bool:
    """Check if model performance has degraded."""
    logger.info("Checking model performance...")
    
    cmd = [
        'python',
        'mlops/monitoring/check_performance.py',
        '--model', 'models/return_predictor/return_predictor_latest.pkl',
        '--test-data', 'data/test.csv'
    ]
    
    success, _ = run_command(cmd, "Performance Check")
    
    # If check fails, performance degraded
    performance_degraded = not success
    
    if performance_degraded:
        logger.warning("⚠️  Performance degraded!")
    else:
        logger.info("✅ Performance acceptable")
    
    return performance_degraded


def trigger_retraining() -> bool:
    """Trigger model retraining."""
    logger.info("Triggering model retraining...")
    
    cmd = [
        'python',
        'mlops/training/train_all_models.py'
    ]
    
    success, _ = run_command(cmd, "Model Retraining")
    
    if success:
        logger.info("✅ Retraining completed successfully")
    else:
        logger.error("❌ Retraining failed")
    
    return success


def deploy_new_model() -> bool:
    """Deploy newly trained model."""
    logger.info("Deploying new model...")
    
    cmd = [
        'python',
        'mlops/deployment/deploy_model.py',
        '--model-path', 'models/return_predictor/return_predictor_latest.pkl'
    ]
    
    success, _ = run_command(cmd, "Model Deployment")
    
    if success:
        logger.info("✅ Model deployed successfully")
    else:
        logger.error("❌ Deployment failed")
    
    return success


def main():
    """Automated retraining pipeline."""
    logger.info("="*80)
    logger.info("AUTOMATED RETRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        # Step 1: Check drift
        drift_detected = check_drift()
        
        # Step 2: Check performance
        performance_degraded = check_performance()
        
        # Step 3: Decide if retraining needed
        should_retrain = drift_detected or performance_degraded
        
        if not should_retrain:
            logger.info("✅ No retraining needed - model performing well")
            logger.info("="*80)
            sys.exit(0)
        
        logger.warning("⚠️  Retraining triggered due to:")
        if drift_detected:
            logger.warning("  - Data drift detected")
        if performance_degraded:
            logger.warning("  - Performance degradation")
        
        # Step 4: Retrain models
        retrain_success = trigger_retraining()
        
        if not retrain_success:
            logger.error("❌ Retraining failed - aborting")
            sys.exit(1)
        
        # Step 5: Deploy new model
        deploy_success = deploy_new_model()
        
        if not deploy_success:
            logger.error("❌ Deployment failed")
            sys.exit(1)
        
        logger.info("="*80)
        logger.info("✅ AUTOMATED RETRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
