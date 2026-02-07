#!/usr/bin/env python
"""
Master training script - trains all models.
Usage: python mlops/training/train_all_models.py
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def run_script(script_path: str, description: str) -> bool:
    """Run a training script and return success status."""
    logger.info(f"▶️  Running: {description}")
    logger.info(f"   Script: {script_path}")
    
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCESS")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} - TIMEOUT (exceeded 1 hour)")
        return False
    except Exception as e:
        logger.error(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Train all models sequentially."""
    logger.info("="*80)
    logger.info("MASTER TRAINING PIPELINE - TRAIN ALL MODELS")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    scripts_dir = Path(__file__).parent
    
    # Define all training scripts
    training_jobs = [
        {
            'script': scripts_dir / 'train_return_predictor.py',
            'description': 'Return Predictor Model'
        },
        {
            'script': scripts_dir / 'train_volatility_model.py',
            'description': 'Volatility Predictor Model'
        },
        # Add more models here as needed
    ]
    
    results = []
    
    # Run each training job
    for i, job in enumerate(training_jobs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Job {i}/{len(training_jobs)}: {job['description']}")
        logger.info(f"{'='*80}\n")
        
        success = run_script(str(job['script']), job['description'])
        results.append({
            'job': job['description'],
            'success': success
        })
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        logger.info(f"{status} - {result['job']}")
    
    logger.info(f"\nTotal: {len(results)} jobs")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
