"""Anomaly Detector - Isolation Forest and Autoencoder."""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.domain.ports.ml.ml_services_interface import IAnomalyDetector
from config.logging_config import get_logger

logger = get_logger(__name__)

class IsolationForestDetector(IAnomalyDetector):
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, data: pd.DataFrame):
        data_scaled = self.scaler.fit_transform(data)
        self.model = IsolationForest(contamination=self.contamination, random_state=42)
        self.model.fit(data_scaled)
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        data_scaled = self.scaler.transform(data)
        predictions = self.model.predict(data_scaled)
        return pd.Series(predictions == -1, index=data.index)
    
    def get_anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        data_scaled = self.scaler.transform(data)
        scores = -self.model.score_samples(data_scaled)
        return pd.Series(scores, index=data.index)
