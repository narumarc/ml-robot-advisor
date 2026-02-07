"""
Redis adapter pour Feature Store et Cache.
Implémente IFeatureStoreService et ICacheService.
"""
from typing import Optional
import pickle
import json
import pandas as pd
import redis.asyncio as aioredis
from redis.asyncio import Redis

from src.domain.ports.services.external_services_interface import IFeatureStoreService, ICacheService
from config.logging_config import get_logger

logger = get_logger(__name__)


class RedisFeatureStore(IFeatureStoreService):
    """
    Redis implementation du Feature Store.
    Stocke les features ML avec TTL pour cache.
    """
    
    def __init__(self, redis_client: Redis, prefix: str = "features:"):
        """
        Initialize Feature Store.
        
        Args:
            redis_client: Async Redis client
            prefix: Key prefix for features
        """
        self.redis = redis_client
        self.prefix = prefix
        logger.info(f"Redis Feature Store initialized with prefix '{prefix}'")
    
    def _make_key(self, key: str) -> str:
        """Create full key with prefix."""
        return f"{self.prefix}{key}"
    
    async def save_features(self, key: str, features: pd.DataFrame, ttl: int = 3600) -> None:
        """
        Save features to Redis with TTL.
        
        Args:
            key: Feature key (e.g., "AAPL_technical_features")
            features: DataFrame of features
            ttl: Time to live in seconds
        """
        try:
            full_key = self._make_key(key)
            
            # Serialize DataFrame to pickle
            serialized = pickle.dumps(features)
            
            # Save to Redis with TTL
            await self.redis.setex(full_key, ttl, serialized)
            
            logger.debug(f"Features saved: {key} (shape={features.shape}, ttl={ttl}s)")
            
        except Exception as e:
            logger.error(f"Error saving features '{key}': {e}")
            raise
    
    async def get_features(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get features from Redis.
        
        Args:
            key: Feature key
            
        Returns:
            DataFrame if found, None otherwise
        """
        try:
            full_key = self._make_key(key)
            
            # Get from Redis
            data = await self.redis.get(full_key)
            
            if data is None:
                logger.debug(f"Features not found: {key}")
                return None
            
            # Deserialize
            features = pickle.loads(data)
            logger.debug(f"Features retrieved: {key} (shape={features.shape})")
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting features '{key}': {e}")
            raise
    
    async def delete_features(self, key: str) -> bool:
        """Delete features from cache."""
        try:
            full_key = self._make_key(key)
            result = await self.redis.delete(full_key)
            
            if result > 0:
                logger.debug(f"Features deleted: {key}")
                return True
            else:
                logger.debug(f"Features not found for deletion: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting features '{key}': {e}")
            raise
    
    async def exists(self, key: str) -> bool:
        """Check if feature key exists."""
        try:
            full_key = self._make_key(key)
            result = await self.redis.exists(full_key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking existence of '{key}': {e}")
            raise
    
    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        try:
            full_key = self._make_key(key)
            ttl = await self.redis.ttl(full_key)
            return ttl
        except Exception as e:
            logger.error(f"Error getting TTL for '{key}': {e}")
            raise


class RedisCacheService(ICacheService):
    """
    Redis implementation du Cache Service.
    Cache général pour tout type de données.
    """
    
    def __init__(self, redis_client: Redis, prefix: str = "cache:"):
        """
        Initialize Cache Service.
        
        Args:
            redis_client: Async Redis client
            prefix: Key prefix for cache
        """
        self.redis = redis_client
        self.prefix = prefix
        logger.info(f"Redis Cache Service initialized with prefix '{prefix}'")
    
    def _make_key(self, key: str) -> str:
        """Create full key with prefix."""
        return f"{self.prefix}{key}"
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to store (as string)
            ttl: Optional TTL in seconds
            
        Returns:
            True if successful
        """
        try:
            full_key = self._make_key(key)
            
            if ttl:
                await self.redis.setex(full_key, ttl, value)
            else:
                await self.redis.set(full_key, value)
            
            logger.debug(f"Cache set: {key} (ttl={ttl})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache '{key}': {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        try:
            full_key = self._make_key(key)
            value = await self.redis.get(full_key)
            
            if value:
                logger.debug(f"Cache hit: {key}")
                return value.decode('utf-8') if isinstance(value, bytes) else value
            else:
                logger.debug(f"Cache miss: {key}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cache '{key}': {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            full_key = self._make_key(key)
            result = await self.redis.delete(full_key)
            
            if result > 0:
                logger.debug(f"Cache deleted: {key}")
                return True
            else:
                logger.debug(f"Cache key not found for deletion: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cache '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            full_key = self._make_key(key)
            result = await self.redis.exists(full_key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking cache existence '{key}': {e}")
            return False
    
    async def set_json(self, key: str, data: dict, ttl: int = None) -> bool:
        """Set a JSON object in cache."""
        try:
            json_str = json.dumps(data)
            return await self.set(key, json_str, ttl)
        except Exception as e:
            logger.error(f"Error setting JSON cache '{key}': {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[dict]:
        """Get a JSON object from cache."""
        try:
            value = await self.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting JSON cache '{key}': {e}")
            return None
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        try:
            full_key = self._make_key(key)
            result = await self.redis.incrby(full_key, amount)
            return result
        except Exception as e:
            logger.error(f"Error incrementing '{key}': {e}")
            raise
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        try:
            full_key = self._make_key(key)
            result = await self.redis.expire(full_key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting expiry for '{key}': {e}")
            return False


class RedisPriceCache:
    """
    Specialized cache for market prices.
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.prefix = "price:"
        logger.info("Redis Price Cache initialized")
    
    async def set_price(self, ticker: str, price: float, ttl: int = 60) -> bool:
        """Set current price with short TTL (prices update frequently)."""
        try:
            key = f"{self.prefix}{ticker}"
            await self.redis.setex(key, ttl, str(price))
            logger.debug(f"Price cached: {ticker}=${price}")
            return True
        except Exception as e:
            logger.error(f"Error caching price for {ticker}: {e}")
            return False
    
    async def get_price(self, ticker: str) -> Optional[float]:
        """Get cached price."""
        try:
            key = f"{self.prefix}{ticker}"
            value = await self.redis.get(key)
            
            if value:
                price = float(value.decode('utf-8') if isinstance(value, bytes) else value)
                logger.debug(f"Price cache hit: {ticker}=${price}")
                return price
            else:
                logger.debug(f"Price cache miss: {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached price for {ticker}: {e}")
            return None
    
    async def set_multiple_prices(self, prices: dict, ttl: int = 60) -> bool:
        """Set multiple prices at once using pipeline."""
        try:
            async with self.redis.pipeline() as pipe:
                for ticker, price in prices.items():
                    key = f"{self.prefix}{ticker}"
                    pipe.setex(key, ttl, str(price))
                await pipe.execute()
            
            logger.debug(f"Multiple prices cached: {len(prices)} tickers")
            return True
        except Exception as e:
            logger.error(f"Error caching multiple prices: {e}")
            return False


async def init_redis_services(redis_url: str) -> dict:
    """
    Initialize Redis services.
    
    Args:
        redis_url: Redis connection URL (e.g., redis://localhost:6379)
        
    Returns:
        Dictionary with service instances
    """
    # Create Redis client
    redis_client = await aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=False  # We handle decoding manually
    )
    
    # Test connection
    await redis_client.ping()
    logger.info(f"Redis connection established: {redis_url}")
    
    # Create services
    feature_store = RedisFeatureStore(redis_client)
    cache_service = RedisCacheService(redis_client)
    price_cache = RedisPriceCache(redis_client)
    
    return {
        "feature_store": feature_store,
        "cache": cache_service,
        "price_cache": price_cache,
        "client": redis_client
    }
