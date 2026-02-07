"""
Data Sources - Adapters pour les sources de données externes.
Implémente IMarketDataService.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.domain.ports.services.external_services_interface import IMarketDataService
from config.logging_config import get_logger

logger = get_logger(__name__)


class YFinanceDataSource(IMarketDataService):
    """
    yfinance adapter pour les données de marché.
    Implémente IMarketDataService.
    """
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize yfinance data source.
        
        Args:
            max_workers: Max concurrent requests
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"YFinance data source initialized (max_workers={max_workers})")
    
    async def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Run in executor to not block
            def _fetch():
                stock = yf.Ticker(ticker)
                info = stock.info
                return info.get('currentPrice') or info.get('regularMarketPrice')
            
            price = await loop.run_in_executor(self.executor, _fetch)
            
            if price is None:
                raise ValueError(f"Could not fetch price for {ticker}")
            
            logger.debug(f"Fetched current price for {ticker}: ${price}")
            return float(price)
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            raise
    
    async def get_historical_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _fetch():
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                return df
            
            df = await loop.run_in_executor(self.executor, _fetch)
            
            if df.empty:
                logger.warning(f"No historical data found for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(df)} historical prices for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            raise
    
    async def get_multiple_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of ticker -> price
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _fetch():
                # Use yfinance download for multiple tickers
                tickers_str = " ".join(tickers)
                data = yf.download(tickers_str, period="1d", progress=False)
                
                prices = {}
                if len(tickers) == 1:
                    prices[tickers[0]] = float(data['Close'].iloc[-1])
                else:
                    for ticker in tickers:
                        if ticker in data['Close'].columns:
                            prices[ticker] = float(data['Close'][ticker].iloc[-1])
                
                return prices
            
            prices = await loop.run_in_executor(self.executor, _fetch)
            
            logger.info(f"Fetched prices for {len(prices)}/{len(tickers)} tickers")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching multiple prices: {e}")
            raise
    
    async def get_company_info(self, ticker: str) -> dict:
        """
        Get company information.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with company info
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _fetch():
                stock = yf.Ticker(ticker)
                return stock.info
            
            info = await loop.run_in_executor(self.executor, _fetch)
            
            logger.debug(f"Fetched company info for {ticker}")
            return info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            raise
    
    async def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Get dividend history."""
        try:
            loop = asyncio.get_event_loop()
            
            def _fetch():
                stock = yf.Ticker(ticker)
                return stock.dividends
            
            dividends = await loop.run_in_executor(self.executor, _fetch)
            return dividends
            
        except Exception as e:
            logger.error(f"Error fetching dividends for {ticker}: {e}")
            raise
    
    async def get_options_chain(self, ticker: str) -> dict:
        """Get options chain data."""
        try:
            loop = asyncio.get_event_loop()
            
            def _fetch():
                stock = yf.Ticker(ticker)
                # Get available expiration dates
                exp_dates = stock.options
                if not exp_dates:
                    return {}
                
                # Get first expiration
                opt = stock.option_chain(exp_dates[0])
                return {
                    "calls": opt.calls,
                    "puts": opt.puts,
                    "expiration_dates": exp_dates
                }
            
            options = await loop.run_in_executor(self.executor, _fetch)
            return options
            
        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {e}")
            raise


class AlphaVantageDataSource(IMarketDataService):
    """
    Alpha Vantage API adapter.
    Alternative data source avec API key.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage data source.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        logger.info("Alpha Vantage data source initialized")
    
    async def get_current_price(self, ticker: str) -> float:
        """Get current price from Alpha Vantage."""
        import aiohttp
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if "Global Quote" not in data:
                        raise ValueError(f"Invalid response for {ticker}")
                    
                    price = float(data["Global Quote"]["05. price"])
                    logger.debug(f"Alpha Vantage: Fetched price for {ticker}: ${price}")
                    return price
                    
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
            raise
    
    async def get_historical_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical data from Alpha Vantage."""
        import aiohttp
        
        try:
            # Alpha Vantage function based on interval
            if interval == "1d":
                function = "TIME_SERIES_DAILY"
            elif interval == "1wk":
                function = "TIME_SERIES_WEEKLY"
            elif interval == "1mo":
                function = "TIME_SERIES_MONTHLY"
            else:
                function = "TIME_SERIES_DAILY"
            
            params = {
                "function": function,
                "symbol": ticker,
                "outputsize": "full",
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    # Parse time series data
                    time_series_key = f"Time Series ({interval})" if interval == "1d" else list(data.keys())[1]
                    time_series = data.get(time_series_key, {})
                    
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Rename columns
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = df.astype(float)
                    
                    # Filter by date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    logger.info(f"Alpha Vantage: Fetched {len(df)} prices for {ticker}")
                    return df
                    
        except Exception as e:
            logger.error(f"Alpha Vantage historical data error for {ticker}: {e}")
            raise
    
    async def get_multiple_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get multiple prices (sequential with rate limiting)."""
        prices = {}
        
        for ticker in tickers:
            try:
                price = await self.get_current_price(ticker)
                prices[ticker] = price
                # Rate limiting (5 API calls per minute for free tier)
                await asyncio.sleep(12)
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        return prices
    
    async def get_company_info(self, ticker: str) -> dict:
        """Get company overview from Alpha Vantage."""
        import aiohttp
        
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    logger.debug(f"Alpha Vantage: Fetched company info for {ticker}")
                    return data
                    
        except Exception as e:
            logger.error(f"Alpha Vantage company info error for {ticker}: {e}")
            raise


def create_market_data_source(
    source_type: str = "yfinance",
    **kwargs
) -> IMarketDataService:
    """
    Factory function to create market data source.
    
    Args:
        source_type: Type of data source ('yfinance' or 'alphavantage')
        **kwargs: Additional arguments for the data source
        
    Returns:
        IMarketDataService implementation
    """
    if source_type == "yfinance":
        return YFinanceDataSource(**kwargs)
    elif source_type == "alphavantage":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Alpha Vantage requires api_key")
        return AlphaVantageDataSource(api_key)
    else:
        raise ValueError(f"Unknown data source type: {source_type}")
