"""Repository Interface (Abstract)"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.entities.portfolio import Portfolio


class PortfolioRepository(ABC):
    """Interface abstraite pour le stockage de portfolios"""
    
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> str:
        """Sauvegarde un portfolio, retourne l'ID"""
        pass
    
    @abstractmethod
    async def get_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """Récupère un portfolio par ID"""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Portfolio]:
        """Récupère tous les portfolios"""
        pass
    
    @abstractmethod
    async def delete(self, portfolio_id: str) -> bool:
        """Supprime un portfolio"""
        pass