"""
Model Factory - Factory Pattern pour créer des modèles ML
"""
from typing import Dict, Any
from src.infrastructure.ml.models.base_model import BaseMLModel
from src.infrastructure.ml.models.random_forest import RandomForestModel


class ModelFactory:

    _models = {
        'random_forest': RandomForestModel,
        'rf': RandomForestModel,  # Alias
        # Ajouter d'autres modèles ici
        # 'xgboost': XGBoostModel,
        # 'lstm': LSTMModel,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseMLModel:
        """
        
        Args:
            model_type: Type de modèle ('random_forest', 'xgboost', etc.)
            **kwargs: Hyperparamètres du modèle
            
        Returns:
            Instance du modèle
            
        Raises:
            ValueError: Si le type de modèle est inconnu
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {available}"
            )
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        if not issubclass(model_class, BaseMLModel):
            raise TypeError(
                f"{model_class} must inherit from BaseMLModel"
            )
        
        cls._models[name.lower()] = model_class
    
    @classmethod
    def available_models(cls) -> list:
        return list(cls._models.keys())


# Convenience function
def create_model(model_type: str, **kwargs) -> BaseMLModel:
    """
    Helper for teh creatation of model.
    
    Usage:
        model = create_model('random_forest', n_estimators=200)
    """
    return ModelFactory.create(model_type, **kwargs)