"""MongoDB Implementation of Repository"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional
from src.domain.entities.portfolio import Portfolio
from src.infrastructure.data.repository import PortfolioRepository


class MongoDBPortfolioRepository(PortfolioRepository):
    """Implémentation MongoDB du repository"""
    
    def __init__(self, mongodb_url: str, database_name: str = 'robo_advisor'):
        self.client = AsyncIOMotorClient(mongodb_url)
        self.db = self.client[database_name]
        self.collection = self.db['portfolios']
    
    async def save(self, portfolio: Portfolio) -> str:
        """Sauvegarde dans MongoDB"""
        doc = portfolio.to_dict()
        result = await self.collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def get_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """Récupère depuis MongoDB"""
        doc = await self.collection.find_one({'portfolio_id': portfolio_id})
        if doc:
            return Portfolio.from_dict(doc)
        return None
    
    async def get_all(self) -> List[Portfolio]:
        """Récupère tous les portfolios"""
        cursor = self.collection.find()
        docs = await cursor.to_list(length=None)
        return [Portfolio.from_dict(doc) for doc in docs]
    
    async def delete(self, portfolio_id: str) -> bool:
        """Supprime un portfolio"""
        result = await self.collection.delete_one({'portfolio_id': portfolio_id})
        return result.deleted_count > 0