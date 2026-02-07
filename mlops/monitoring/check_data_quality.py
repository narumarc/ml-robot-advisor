#!/usr/bin/env python
"""
Data quality check script.
Usage: python mlops/monitoring/check_data_quality.py --train data/train.csv --test data/test.csv
"""
import sys
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.infrastructure.ml.monitoring.data_quality_checker import DataQualityChecker
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def main():
    """Check data quality."""
    parser = argparse.ArgumentParser(description="Check data quality")
    parser.add_argument("--train", type=str, required=True, help="Training data CSV")
    parser.add_argument("--test", type=str, required=True, help="Test data CSV")
    parser.add_argument("--output", type=str, default="reports/data_quality_report.json", help="Output report")
    parser.add_argument("--features", type=str, nargs='+', help="Feature columns")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DATA QUALITY CHECK")
    logger.info("="*80)
    
    try:
        # Load data
        train_data = pd.read_csv(args.train)
        test_data = pd.read_csv(args.test)
        
        logger.info(f"Train data: {train_data.shape}")
        logger.info(f"Test data: {test_data.shape}")
        
        # Infer feature columns if not provided
        if args.features:
            feature_cols = args.features
        else:
            # Assume all columns except 'target' are features
            feature_cols = [col for col in train_data.columns if col != 'target']
        
        logger.info(f"Feature columns: {feature_cols}")
        
        # Run checks
        checker = DataQualityChecker(verbose=True)
        results = checker.run_all_checks(
            train_data=train_data,
            test_data=test_data,
            feature_cols=feature_cols,
            target_col='target' if 'target' in train_data.columns else None
        )
        
        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Report saved to {output_path}")
        
        # Exit based on results
        if results["summary"]["critical_issues_count"] > 0:
            logger.error("❌ CRITICAL DATA QUALITY ISSUES FOUND!")
            sys.exit(1)
        else:
            logger.info("✅ All data quality checks passed")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Data quality check failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
