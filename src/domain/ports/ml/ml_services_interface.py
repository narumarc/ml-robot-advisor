"""
ML Service interfaces (PORTS).
Définit les contrats pour les services ML.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Résultat d'une prédiction."""
    predictions: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]]
    metadata: Dict[str, any]


@dataclass
class DriftReport:
    """Rapport de drift detection."""
    drift_detected: bool
    drift_score: float
    affected_features: List[str]
    timestamp: str
    details: Dict[str, any]


class IReturnPredictor(ABC):
    """Interface pour la prédiction des rendements."""
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        **kwargs
    ) -> Dict[str, float]:
        """Entraîner le modèle de prédiction."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """Prédire les rendements."""
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Évaluer la performance du modèle."""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Sauvegarder le modèle."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Charger le modèle."""
        pass


class IVolatilityPredictor(ABC):
    """Interface pour la prédiction de volatilité."""
    
    @abstractmethod
    def train(self, returns: pd.Series, **kwargs) -> Dict[str, float]:
        """Entraîner le modèle de volatilité (ex: GARCH)."""
        pass
    
    @abstractmethod
    def predict(self, horizon: int = 1) -> PredictionResult:
        """Prédire la volatilité future."""
        pass
    
    @abstractmethod
    def forecast(self, steps: int = 30) -> pd.Series:
        """Prévoir la volatilité sur plusieurs périodes."""
        pass


class IAnomalyDetector(ABC):
    """Interface pour la détection d'anomalies."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Entraîner le détecteur d'anomalies."""
        pass
    
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Détecter les anomalies (retourne booléen par observation)."""
        pass
    
    @abstractmethod
    def get_anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        """Obtenir le score d'anomalie pour chaque observation."""
        pass


class IDriftDetector(ABC):
    """Interface pour la détection de drift."""
    
    @abstractmethod
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> DriftReport:
        """Détecter le drift entre données de référence et actuelles."""
        pass
    
    @abstractmethod
    def generate_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> dict:
        """Générer un rapport détaillé de drift."""
        pass


class IModelTrainer(ABC):
    """Interface pour l'entraînement de modèles."""
    
    @abstractmethod
    async def train_model(
        self,
        model_name: str,
        training_data: pd.DataFrame,
        config: dict
    ) -> Dict[str, any]:
        """Entraîner un modèle et retourner les métriques."""
        pass
    
    @abstractmethod
    async def hyperparameter_tuning(
        self,
        model_name: str,
        training_data: pd.DataFrame,
        param_grid: dict
    ) -> Dict[str, any]:
        """Optimiser les hyperparamètres."""
        pass
    
    @abstractmethod
    async def validate_model(
        self,
        model: any,
        validation_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Valider un modèle."""
        pass


class IModelMonitor(ABC):
    """Interface pour le monitoring des modèles."""
    
    @abstractmethod
    async def log_prediction(
        self,
        model_name: str,
        prediction: float,
        actual: Optional[float] = None,
        metadata: dict = None
    ) -> None:
        """Logger une prédiction."""
        pass
    
    @abstractmethod
    async def check_performance(
        self,
        model_name: str,
        threshold: float
    ) -> Dict[str, any]:
        """Vérifier la performance d'un modèle."""
        pass
    
    @abstractmethod
    async def should_retrain(
        self,
        model_name: str,
        criteria: dict
    ) -> bool:
        """Déterminer si le modèle doit être réentraîné."""
        pass
