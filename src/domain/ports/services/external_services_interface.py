"""
Service interfaces (PORTS) pour les services externes.
Définit les contrats pour les data sources, ML services, etc.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd


class IMarketDataService(ABC):
    """Interface pour les services de données de marché."""
    
    @abstractmethod
    async def get_current_price(self, ticker: str) -> float:
        """Récupérer le prix actuel d'un ticker."""
        pass
    
    @abstractmethod
    async def get_historical_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Récupérer l'historique des prix."""
        pass
    
    @abstractmethod
    async def get_multiple_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Récupérer les prix de plusieurs tickers."""
        pass
    
    @abstractmethod
    async def get_company_info(self, ticker: str) -> dict:
        """Récupérer les informations d'une entreprise."""
        pass


class IFeatureStoreService(ABC):
    """Interface pour le feature store (cache des features ML)."""
    
    @abstractmethod
    async def save_features(self, key: str, features: pd.DataFrame, ttl: int = 3600) -> None:
        """Sauvegarder des features avec TTL."""
        pass
    
    @abstractmethod
    async def get_features(self, key: str) -> Optional[pd.DataFrame]:
        """Récupérer des features depuis le cache."""
        pass
    
    @abstractmethod
    async def delete_features(self, key: str) -> bool:
        """Supprimer des features du cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Vérifier si une clé existe."""
        pass


class ICacheService(ABC):
    """Interface pour le service de cache général."""
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Définir une valeur dans le cache."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Récupérer une valeur du cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Supprimer une clé du cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Vérifier si une clé existe."""
        pass


class INotificationService(ABC):
    """Interface pour les notifications (email, SMS, etc.)."""
    
    @abstractmethod
    async def send_alert(
        self, 
        recipient: str,
        subject: str,
        message: str,
        priority: str = "normal"
    ) -> bool:
        """Envoyer une alerte."""
        pass
    
    @abstractmethod
    async def send_report(
        self,
        recipient: str,
        report_type: str,
        data: dict
    ) -> bool:
        """Envoyer un rapport."""
        pass
