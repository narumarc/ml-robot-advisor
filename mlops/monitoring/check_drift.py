#!/usr/bin/env python
"""
Drift detection script - checks for data drift.
Usage: python mlops/monitoring/check_drift.py --reference data/reference.csv --current data/current.csv
"""
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops.monitoring.drift_detector import DriftDetector
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def send_alert(message: str, severity: str = "WARNING"):
    """Send alert (placeholder - integrate with Slack/Email)."""
    logger.warning(f"ALERT [{severity}]: {message}")
    # TODO: Implement actual alerting (Slack, Email, PagerDuty)
    print(f"\n🚨 ALERT: {message}\n")


def main():
    """Check for data drift."""
    parser = argparse.ArgumentParser(description="Check for data drift")
    parser.add_argument("--reference", type=str, required=True, help="Reference data CSV")
    parser.add_argument("--current", type=str, required=True, help="Current data CSV")
    parser.add_argument("--output", type=str, default="reports/drift_report.html", help="Output HTML report")
    parser.add_argument("--threshold", type=float, default=0.1, help="Drift threshold")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DRIFT DETECTION CHECK")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        # Load data
        reference_data = load_data(args.reference)
        current_data = load_data(args.current)
        
        # Initialize detector
        detector = DriftDetector(drift_threshold=args.threshold)
        
        # Detect drift
        logger.info("Running drift detection...")
        drift_report = detector.detect_data_drift(reference_data, current_data)
        
        # Generate HTML report
        logger.info(f"Generating HTML report: {args.output}")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        detector.generate_html_report(
            reference_data=reference_data,
            current_data=current_data,
            output_path=str(output_path)
        )
        
        # Results
        logger.info("\n" + "="*80)
        logger.info("DRIFT DETECTION RESULTS")
        logger.info("="*80)
        logger.info(f"Drift detected: {drift_report.drift_detected}")
        logger.info(f"Drift score: {drift_report.drift_score:.2%}")
        logger.info(f"Drifted features: {len(drift_report.drifted_features)}")
        
        if drift_report.drifted_features:
            logger.warning(f"Drifted features: {drift_report.drifted_features}")
        
        logger.info(f"HTML report: {output_path}")
        logger.info("="*80)
        
        # Send alert if drift detected
        if drift_report.drift_detected:
            send_alert(
                f"Data drift detected! Score: {drift_report.drift_score:.2%}, "
                f"Features affected: {len(drift_report.drifted_features)}",
                severity="HIGH"
            )
            logger.warning("⚠️  DRIFT DETECTED - Consider retraining models")
            sys.exit(1)  # Exit with error code
        else:
            logger.info("✅ No significant drift detected")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Drift check failed: {e}", exc_info=True)
        send_alert(f"Drift check failed: {e}", severity="CRITICAL")
        sys.exit(1)


if __name__ == "__main__":
    main()
