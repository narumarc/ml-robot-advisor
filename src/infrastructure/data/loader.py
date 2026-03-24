"""Data Loader - Support multiple sources"""
import pandas as pd
from typing import List
from datetime import datetime


class DataLoader:
    """
    Charge les données de marché depuis différentes sources.
    Supporte: Stooq, Alpha Vantage, MongoDB (cache), CSV local
    """
    
    def __init__(self, source: str = 'stooq', cache_enabled: bool = True):
        self.source = source
        self.cache_enabled = cache_enabled
    
    async def load_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Charge les prix pour les tickers donnés.
        
        Returns:
            DataFrame avec colonnes = tickers, index = dates
        """
        
        # Check cache first
        if self.cache_enabled:
            cached_data = await self._load_from_cache(tickers, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Load from source
        if self.source == 'stooq':
            data = await self._load_from_stooq(tickers, start_date, end_date)
        elif self.source == 'alphavantage':
            data = await self._load_from_alphavantage(tickers, start_date, end_date)
        elif self.source == 'csv':
            data = await self._load_from_csv(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown source: {self.source}")
        
        # Cache for future use
        if self.cache_enabled:
            await self._save_to_cache(data, tickers, start_date, end_date)
        
        return data
    
    async def _load_from_stooq(self, tickers, start_date, end_date):
        """Load from Stooq API"""
        from pandas_datareader import data as web
        
        frames = []
        for ticker in tickers:
            stooq_ticker = f"{ticker}.US"
            df = web.DataReader(stooq_ticker, "stooq", start=start_date, end=end_date)
            df = df.sort_index()
            frames.append(df["Close"].rename(ticker))
        
        return pd.concat(frames, axis=1).dropna()
    
    async def _load_from_cache(self, tickers, start_date, end_date):
        """Load from MongoDB/Redis cache"""
        # TODO: Implement
        return None
    
    async def _save_to_cache(self, data, tickers, start_date, end_date):
        """Save to MongoDB/Redis cache"""
        # TODO: Implement
        pass