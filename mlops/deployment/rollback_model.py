#!/usr/bin/env python
"""Rollback model to previous version."""
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def list_backups(backup_dir: Path) -> list:
    """List available model backups."""
    backups = sorted(backup_dir.glob("model_backup_*.pkl"), reverse=True)
    return backups

def rollback_to_backup(backup_path: Path, deployment_dir: Path) -> bool:
    """Rollback to a specific backup."""
    try:
        target_path = deployment_dir / "model.pkl"
        
        # Backup current model first
        if target_path.exists():
            emergency_backup = deployment_dir / "backups" / f"emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            shutil.copy2(target_path, emergency_backup)
            logger.info(f"Created emergency backup: {emergency_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, target_path)
        logger.info(f"✅ Rolled back to: {backup_path.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Rollback model to previous version")
    parser.add_argument("--deployment-dir", type=str, default="deployments/production")
    parser.add_argument("--version", type=str, help="Specific backup to restore (timestamp)")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MODEL ROLLBACK")
    logger.info("="*80)
    
    deployment_dir = Path(args.deployment_dir)
    backup_dir = deployment_dir / "backups"
    
    if not backup_dir.exists():
        logger.error("❌ No backups found")
        sys.exit(1)
    
    # List backups
    backups = list_backups(backup_dir)
    
    if not backups:
        logger.error("❌ No backups available")
        sys.exit(1)
    
    logger.info(f"Found {len(backups)} backups:")
    for i, backup in enumerate(backups):
        logger.info(f"  {i+1}. {backup.name}")
    
    # Select backup
    if args.version:
        backup_path = backup_dir / f"model_backup_{args.version}.pkl"
        if not backup_path.exists():
            logger.error(f"❌ Backup not found: {backup_path}")
            sys.exit(1)
    else:
        # Use most recent backup
        backup_path = backups[0]
        logger.info(f"Using most recent backup: {backup_path.name}")
    
    # Perform rollback
    if rollback_to_backup(backup_path, deployment_dir):
        logger.info("✅ ROLLBACK SUCCESSFUL")
        sys.exit(0)
    else:
        logger.error("❌ ROLLBACK FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
