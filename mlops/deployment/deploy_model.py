#!/usr/bin/env python
"""
Model deployment script.
Usage: python mlops/deployment/deploy_model.py --model-path models/model.pkl --env production
"""
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def backup_current_model(deployment_dir: Path) -> Path:
    """Backup current production model."""
    current_model = deployment_dir / "model.pkl"
    
    if current_model.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = deployment_dir / "backups" / f"model_backup_{timestamp}.pkl"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(current_model, backup_path)
        logger.info(f"✅ Backed up current model to {backup_path}")
        return backup_path
    else:
        logger.info("No existing model to backup")
        return None


def deploy_model(model_path: Path, deployment_dir: Path) -> bool:
    """Deploy model to production directory."""
    try:
        # Copy model to deployment directory
        target_path = deployment_dir / "model.pkl"
        shutil.copy2(model_path, target_path)
        
        logger.info(f"✅ Model deployed to {target_path}")
        
        # Save deployment metadata
        metadata = {
            "deployed_at": datetime.now().isoformat(),
            "source_model": str(model_path),
            "deployment_dir": str(deployment_dir)
        }
        
        metadata_path = deployment_dir / "deployment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Deployment metadata saved to {metadata_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False


def verify_deployment(deployment_dir: Path) -> bool:
    """Verify deployed model."""
    try:
        model_path = deployment_dir / "model.pkl"
        
        if not model_path.exists():
            logger.error("❌ Model file not found after deployment")
            return False
        
        # Try loading the model
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info("✅ Model verification successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False


def main():
    """Deploy model to production."""
    parser = argparse.ArgumentParser(description="Deploy ML model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    parser.add_argument("--env", type=str, default="production", choices=["staging", "production"])
    parser.add_argument("--deployment-dir", type=str, default=None, help="Deployment directory")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MODEL DEPLOYMENT")
    logger.info(f"Environment: {args.env}")
    logger.info("="*80)
    
    try:
        # Validate model path
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        # Determine deployment directory
        if args.deployment_dir:
            deployment_dir = Path(args.deployment_dir)
        else:
            deployment_dir = Path(f"deployments/{args.env}")
        
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model: {model_path}")
        logger.info(f"Deployment directory: {deployment_dir}")
        
        # Step 1: Backup current model
        logger.info("\n1️⃣  Backing up current model...")
        backup_current_model(deployment_dir)
        
        # Step 2: Deploy new model
        logger.info("\n2️⃣  Deploying new model...")
        if not deploy_model(model_path, deployment_dir):
            logger.error("❌ Deployment failed")
            sys.exit(1)
        
        # Step 3: Verify deployment
        logger.info("\n3️⃣  Verifying deployment...")
        if not verify_deployment(deployment_dir):
            logger.error("❌ Verification failed - rolling back")
            # Implement rollback here if needed
            sys.exit(1)
        
        logger.info("\n" + "="*80)
        logger.info("✅ DEPLOYMENT SUCCESSFUL")
        logger.info("="*80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
