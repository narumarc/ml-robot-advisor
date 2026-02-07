#!/usr/bin/env python
"""Model versioning and registry management."""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

class ModelRegistry:
    """Simple model registry using JSON."""
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": []}
    
    def _save_registry(self):
        """Save registry to file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict,
        metadata: Dict = None
    ) -> str:
        """Register a new model version."""
        model_entry = {
            "model_name": model_name,
            "version": version,
            "model_path": model_path,
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "registered"
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        
        logger.info(f"✅ Registered {model_name} v{version}")
        return version
    
    def promote_to_production(self, model_name: str, version: str) -> bool:
        """Promote a model version to production."""
        # Demote current production model
        for model in self.registry["models"]:
            if model["model_name"] == model_name and model.get("status") == "production":
                model["status"] = "archived"
                model["archived_at"] = datetime.now().isoformat()
        
        # Promote new version
        for model in self.registry["models"]:
            if model["model_name"] == model_name and model["version"] == version:
                model["status"] = "production"
                model["promoted_at"] = datetime.now().isoformat()
                self._save_registry()
                logger.info(f"✅ Promoted {model_name} v{version} to production")
                return True
        
        logger.error(f"❌ Model {model_name} v{version} not found")
        return False
    
    def get_production_model(self, model_name: str) -> Dict:
        """Get current production model."""
        for model in self.registry["models"]:
            if model["model_name"] == model_name and model.get("status") == "production":
                return model
        return None
    
    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model."""
        return [m for m in self.registry["models"] if m["model_name"] == model_name]

def main():
    """Demo model registry usage."""
    registry = ModelRegistry()
    
    # Register a new model
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    registry.register_model(
        model_name="return_predictor",
        version=version,
        model_path=f"models/return_predictor/model_{version}.pkl",
        metrics={"mse": 0.001, "mae": 0.025},
        metadata={"trained_by": "auto_pipeline"}
    )
    
    # Promote to production
    registry.promote_to_production("return_predictor", version)
    
    # Get production model
    prod_model = registry.get_production_model("return_predictor")
    logger.info(f"Production model: {prod_model}")
    
    # List all versions
    versions = registry.list_versions("return_predictor")
    logger.info(f"Total versions: {len(versions)}")

if __name__ == "__main__":
    main()
