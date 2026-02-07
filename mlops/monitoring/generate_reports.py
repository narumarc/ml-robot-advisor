#!/usr/bin/env python
"""Generate comprehensive monitoring reports."""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def generate_html_report(data: dict, output_path: str):
    """Generate HTML report from monitoring data."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLOps Monitoring Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metric {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .success {{ border-left: 4px solid #4CAF50; }}
            .warning {{ border-left: 4px solid #ff9800; }}
            .error {{ border-left: 4px solid #f44336; }}
        </style>
    </head>
    <body>
        <h1>MLOps Monitoring Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Model Performance</h2>
        <div class="metric success">
            <h3>✅ Return Predictor</h3>
            <p>MSE: {data.get('return_predictor_mse', 'N/A')}</p>
            <p>MAE: {data.get('return_predictor_mae', 'N/A')}</p>
        </div>
        
        <h2>Data Quality</h2>
        <div class="metric {'success' if not data.get('data_quality_issues') else 'warning'}">
            <h3>{'✅' if not data.get('data_quality_issues') else '⚠️'} Data Quality</h3>
            <p>Issues: {len(data.get('data_quality_issues', []))}</p>
        </div>
        
        <h2>Drift Detection</h2>
        <div class="metric {'success' if not data.get('drift_detected') else 'warning'}">
            <h3>{'✅' if not data.get('drift_detected') else '⚠️'} Drift Status</h3>
            <p>Drift detected: {data.get('drift_detected', False)}</p>
            <p>Drift score: {data.get('drift_score', 'N/A')}</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✅ HTML report generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate monitoring reports")
    parser.add_argument("--data", type=str, required=True, help="Monitoring data JSON")
    parser.add_argument("--output", type=str, default="reports/monitoring_report.html")
    args = parser.parse_args()
    
    logger.info("Generating monitoring report...")
    
    # Load monitoring data
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    # Generate HTML report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_html_report(data, str(output_path))
    
    logger.info("✅ Report generation complete")

if __name__ == "__main__":
    main()
