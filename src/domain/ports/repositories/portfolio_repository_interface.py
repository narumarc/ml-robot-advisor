"""
Repository interfaces (PORTS) - Clean Architecture.
Ces interfaces définissent les contrats que les adapters doivent implémenter.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from src.domain.entities.portfolio import Portfolio
from src.domain.entities.asset import Asset


class IPortfolioRepository(ABC):
    """Interface pour la persistence des portfolios."""
    
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> None:
        """Sauvegarder un portfolio."""
        pass
    
    @abstractmethod
    async def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """Trouver un portfolio par son ID."""
        pass
    
    @abstractmethod
    async def find_by_owner(self, owner_id: str) -> List[Portfolio]:
        """Trouver tous les portfolios d'un propriétaire."""
        pass
    
    @abstractmethod
    async def delete(self, portfolio_id: UUID) -> bool:
        """Supprimer un portfolio."""
        pass
    
    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Portfolio]:
        """Lister tous les portfolios avec pagination."""
        pass


class IAssetRepository(ABC):
    """Interface pour la persistence des assets."""
    
    @abstractmethod
    async def save(self, asset: Asset) -> None:
        """Sauvegarder un asset."""
        pass
    
    @abstractmethod
    async def find_by_id(self, asset_id: UUID) -> Optional[Asset]:
        """Trouver un asset par son ID."""
        pass
    
    @abstractmethod
    async def find_by_ticker(self, ticker: str) -> Optional[Asset]:
        """Trouver un asset par son ticker."""
        pass
    
    @abstractmethod
    async def find_by_tickers(self, tickers: List[str]) -> List[Asset]:
        """Trouver plusieurs assets par leurs tickers."""
        pass
    
    @abstractmethod
    async def update_price(self, ticker: str, price: float, timestamp: datetime) -> None:
        """Mettre à jour le prix d'un asset."""
        pass
    
    @abstractmethod
    async def list_by_sector(self, sector: str) -> List[Asset]:
        """Lister les assets d'un secteur."""
        pass


class ITransactionRepository(ABC):
    """Interface pour la persistence des transactions."""
    
    @abstractmethod
    async def save_transaction(self, transaction: dict) -> str:
        """Sauvegarder une transaction et retourner son ID."""
        pass
    
    @abstractmethod
    async def find_by_portfolio(
        self, 
        portfolio_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[dict]:
        """Trouver toutes les transactions d'un portfolio."""
        pass
    
    @abstractmethod
    async def find_by_asset(self, asset_id: UUID) -> List[dict]:
        """Trouver toutes les transactions d'un asset."""
        pass


class IPriceHistoryRepository(ABC):
    """Interface pour l'historique des prix."""
    
    @abstractmethod
    async def save_price(self, ticker: str, price: float, timestamp: datetime, metadata: dict = None) -> None:
        """Sauvegarder un prix historique."""
        pass
    
    @abstractmethod
    async def get_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[dict]:
        """Récupérer l'historique des prix."""
        pass
    
    @abstractmethod
    async def get_latest_price(self, ticker: str) -> Optional[dict]:
        """Récupérer le dernier prix connu."""
        pass


class IModelRepository(ABC):
    """Interface pour la persistence des modèles ML."""
    
    @abstractmethod
    async def save_model(
        self, 
        model_name: str,
        model_version: str,
        model_path: str,
        metadata: dict
    ) -> str:
        """Sauvegarder les métadonnées d'un modèle."""
        pass
    
    @abstractmethod
    async def get_latest_model(self, model_name: str) -> Optional[dict]:
        """Récupérer le dernier modèle enregistré."""
        pass
    
    @abstractmethod
    async def get_model_by_version(self, model_name: str, version: str) -> Optional[dict]:
        """Récupérer un modèle par sa version."""
        pass
    
    @abstractmethod
    async def list_models(self, model_name: Optional[str] = None) -> List[dict]:
        """Lister tous les modèles."""
        pass
