"""
MongoDB adapter pour Portfolio Repository.
Implémente IPortfolioRepository.
"""
from typing import List, Optional
from uuid import UUID
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from src.domain.ports.repositories.portfolio_repository_interface import IPortfolioRepository
from src.domain.entities.portfolio import Portfolio
from config.logging_config import get_logger

logger = get_logger(__name__)


class MongoDBPortfolioRepository(IPortfolioRepository):
    """
    Implementation MongoDB du Portfolio Repository.
    """
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """
        Initialize repository with MongoDB database.
        
        Args:
            database: Motor async MongoDB database instance
        """
        self.db = database
        self.collection = self.db.portfolios
        logger.info("MongoDB Portfolio Repository initialized")
    
    async def save(self, portfolio: Portfolio) -> None:
        """Save or update a portfolio."""
        try:
            portfolio_dict = portfolio.to_dict()
            
            # Upsert: update if exists, insert if not
            result = await self.collection.update_one(
                {"id": portfolio_dict["id"]},
                {"$set": portfolio_dict},
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Portfolio {portfolio.id} inserted")
            else:
                logger.info(f"Portfolio {portfolio.id} updated")
                
        except Exception as e:
            logger.error(f"Error saving portfolio {portfolio.id}: {e}")
            raise
    
    async def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """Find portfolio by ID."""
        try:
            doc = await self.collection.find_one({"id": str(portfolio_id)})
            
            if doc:
                # Remove MongoDB _id field
                doc.pop("_id", None)
                return Portfolio.from_dict(doc)
            
            logger.debug(f"Portfolio {portfolio_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding portfolio {portfolio_id}: {e}")
            raise
    
    async def find_by_owner(self, owner_id: str) -> List[Portfolio]:
        """Find all portfolios by owner ID."""
        try:
            cursor = self.collection.find({"owner_id": owner_id})
            portfolios = []
            
            async for doc in cursor:
                doc.pop("_id", None)
                portfolios.append(Portfolio.from_dict(doc))
            
            logger.info(f"Found {len(portfolios)} portfolios for owner {owner_id}")
            return portfolios
            
        except Exception as e:
            logger.error(f"Error finding portfolios for owner {owner_id}: {e}")
            raise
    
    async def delete(self, portfolio_id: UUID) -> bool:
        """Delete a portfolio."""
        try:
            result = await self.collection.delete_one({"id": str(portfolio_id)})
            
            if result.deleted_count > 0:
                logger.info(f"Portfolio {portfolio_id} deleted")
                return True
            else:
                logger.warning(f"Portfolio {portfolio_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting portfolio {portfolio_id}: {e}")
            raise
    
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Portfolio]:
        """List all portfolios with pagination."""
        try:
            cursor = self.collection.find().skip(skip).limit(limit)
            portfolios = []
            
            async for doc in cursor:
                doc.pop("_id", None)
                portfolios.append(Portfolio.from_dict(doc))
            
            logger.info(f"Listed {len(portfolios)} portfolios (skip={skip}, limit={limit})")
            return portfolios
            
        except Exception as e:
            logger.error(f"Error listing portfolios: {e}")
            raise
    
    async def count(self) -> int:
        """Count total portfolios."""
        try:
            count = await self.collection.count_documents({})
            return count
        except Exception as e:
            logger.error(f"Error counting portfolios: {e}")
            raise
    
    async def create_indexes(self) -> None:
        """Create indexes for better query performance."""
        try:
            # Index on id (unique)
            await self.collection.create_index("id", unique=True)
            
            # Index on owner_id (for finding all portfolios of a user)
            await self.collection.create_index("owner_id")
            
            # Index on created_at
            await self.collection.create_index("created_at")
            
            logger.info("Portfolio collection indexes created")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise


class MongoDBAssetRepository:
    """MongoDB implementation for Asset Repository."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.collection = self.db.assets
        logger.info("MongoDB Asset Repository initialized")
    
    async def save(self, asset: dict) -> None:
        """Save or update an asset."""
        try:
            await self.collection.update_one(
                {"ticker": asset["ticker"]},
                {"$set": asset},
                upsert=True
            )
            logger.info(f"Asset {asset['ticker']} saved")
        except Exception as e:
            logger.error(f"Error saving asset: {e}")
            raise
    
    async def find_by_ticker(self, ticker: str) -> Optional[dict]:
        """Find asset by ticker."""
        try:
            doc = await self.collection.find_one({"ticker": ticker})
            if doc:
                doc.pop("_id", None)
            return doc
        except Exception as e:
            logger.error(f"Error finding asset {ticker}: {e}")
            raise
    
    async def find_by_tickers(self, tickers: List[str]) -> List[dict]:
        """Find multiple assets by tickers."""
        try:
            cursor = self.collection.find({"ticker": {"$in": tickers}})
            assets = []
            async for doc in cursor:
                doc.pop("_id", None)
                assets.append(doc)
            return assets
        except Exception as e:
            logger.error(f"Error finding assets: {e}")
            raise
    
    async def update_price(self, ticker: str, price: float, timestamp: str) -> None:
        """Update asset price."""
        try:
            await self.collection.update_one(
                {"ticker": ticker},
                {
                    "$set": {
                        "current_price": price,
                        "last_price_update": timestamp
                    }
                }
            )
            logger.debug(f"Price updated for {ticker}: {price}")
        except Exception as e:
            logger.error(f"Error updating price for {ticker}: {e}")
            raise
    
    async def create_indexes(self) -> None:
        """Create indexes."""
        try:
            await self.collection.create_index("ticker", unique=True)
            await self.collection.create_index("sector")
            await self.collection.create_index("asset_type")
            logger.info("Asset collection indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise


class MongoDBTransactionRepository:
    """MongoDB implementation for Transaction Repository."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.collection = self.db.transactions
        logger.info("MongoDB Transaction Repository initialized")
    
    async def save_transaction(self, transaction: dict) -> str:
        """Save a transaction and return its ID."""
        try:
            result = await self.collection.insert_one(transaction)
            logger.info(f"Transaction saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving transaction: {e}")
            raise
    
    async def find_by_portfolio(
        self,
        portfolio_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[dict]:
        """Find transactions by portfolio ID."""
        try:
            query = {"portfolio_id": portfolio_id}
            
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["timestamp"] = date_filter
            
            cursor = self.collection.find(query).sort("timestamp", -1)
            transactions = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                transactions.append(doc)
            
            return transactions
        except Exception as e:
            logger.error(f"Error finding transactions: {e}")
            raise
    
    async def create_indexes(self) -> None:
        """Create indexes."""
        try:
            await self.collection.create_index("portfolio_id")
            await self.collection.create_index("asset_id")
            await self.collection.create_index("timestamp")
            await self.collection.create_index([("portfolio_id", 1), ("timestamp", -1)])
            logger.info("Transaction collection indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise


async def init_mongodb_repositories(mongo_uri: str, database_name: str) -> dict:
    """
    Initialize MongoDB repositories.
    
    Args:
        mongo_uri: MongoDB connection URI
        database_name: Database name
        
    Returns:
        Dictionary with repository instances
    """
    client = AsyncIOMotorClient(mongo_uri)
    db = client[database_name]
    
    # Create repositories
    portfolio_repo = MongoDBPortfolioRepository(db)
    asset_repo = MongoDBAssetRepository(db)
    transaction_repo = MongoDBTransactionRepository(db)
    
    # Create indexes
    await portfolio_repo.create_indexes()
    await asset_repo.create_indexes()
    await transaction_repo.create_indexes()
    
    logger.info(f"MongoDB repositories initialized for database: {database_name}")
    
    return {
        "portfolio": portfolio_repo,
        "asset": asset_repo,
        "transaction": transaction_repo,
        "client": client,
        "database": db
    }
