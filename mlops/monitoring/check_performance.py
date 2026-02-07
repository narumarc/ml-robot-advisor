#!/usr/bin/env python
"""Model performance monitoring script."""
import sys
import argparse
import pandas as pd
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger
from src.infrastructure.ml.evaluation.regression_metrics import calculate_regression_metrics

setup_logging()
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Check model performance")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--test-data", type=str, required=True, help="Test data CSV")
    parser.add_argument("--threshold-mse", type=float, default=0.001, help="MSE threshold")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MODEL PERFORMANCE CHECK")
    logger.info("="*80)
    
    try:
        # Load model
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Load test data
        test_df = pd.read_csv(args.test_data)
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        # Predict
        X_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_scaled)
        
        # Evaluate
        metrics = calculate_regression_metrics(y_test.values, y_pred)
        
        logger.info(f"MSE: {metrics['mse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        
        # Check threshold
        if metrics['mse'] > args.threshold_mse:
            logger.warning(f"⚠️  Performance degraded! MSE {metrics['mse']:.6f} > {args.threshold_mse}")
            sys.exit(1)
        else:
            logger.info("✅ Performance acceptable")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Performance check failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
