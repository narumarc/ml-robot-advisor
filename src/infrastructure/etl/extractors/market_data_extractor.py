"""
ETL Pipeline - Extractors.
Extraction des données depuis les sources externes.
"""
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

from src.infrastructure.data_sources.market_data import IMarketDataService
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Résultat d'une extraction."""
    success: bool
    data: pd.DataFrame
    metadata: Dict
    errors: List[str]


class MarketDataExtractor:
    """
    Extractor pour les données de marché.
    """
    
    def __init__(self, market_data_service: IMarketDataService):
        """
        Initialize extractor.
        
        Args:
            market_data_service: Service de données de marché
        """
        self.market_data = market_data_service
        logger.info("Market Data Extractor initialized")
    
    async def extract_historical_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> ExtractionResult:
        """
        Extract historical prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            ExtractionResult with combined DataFrame
        """
        logger.info(f"Extracting historical prices for {len(tickers)} tickers")
        
        all_data = {}
        errors = []
        
        for ticker in tickers:
            try:
                df = await self.market_data.get_historical_prices(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                
                if not df.empty:
                    all_data[ticker] = df
                    logger.debug(f"✓ Extracted {len(df)} rows for {ticker}")
                else:
                    errors.append(f"No data for {ticker}")
                    logger.warning(f"✗ No data for {ticker}")
                    
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
                logger.error(f"✗ Error extracting {ticker}: {e}")
        
        # Combine data into single DataFrame
        if all_data:
            # Create multi-index DataFrame
            combined_df = pd.concat(all_data, axis=1, keys=all_data.keys())
            
            metadata = {
                "extraction_time": datetime.now().isoformat(),
                "tickers": tickers,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": interval,
                "total_tickers": len(tickers),
                "successful_tickers": len(all_data),
                "failed_tickers": len(errors)
            }
            
            logger.info(f"Extraction complete: {len(all_data)}/{len(tickers)} successful")
            
            return ExtractionResult(
                success=len(all_data) > 0,
                data=combined_df,
                metadata=metadata,
                errors=errors
            )
        else:
            logger.error("No data extracted")
            return ExtractionResult(
                success=False,
                data=pd.DataFrame(),
                metadata={"errors": errors},
                errors=errors
            )
    
    async def extract_current_prices(self, tickers: List[str]) -> ExtractionResult:
        """
        Extract current prices for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            ExtractionResult with prices DataFrame
        """
        logger.info(f"Extracting current prices for {len(tickers)} tickers")
        
        try:
            prices = await self.market_data.get_multiple_prices(tickers)
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(prices, orient='index', columns=['price'])
            df.index.name = 'ticker'
            df['timestamp'] = datetime.now()
            
            metadata = {
                "extraction_time": datetime.now().isoformat(),
                "total_tickers": len(tickers),
                "successful_tickers": len(prices)
            }
            
            logger.info(f"Current prices extracted: {len(prices)}/{len(tickers)}")
            
            return ExtractionResult(
                success=True,
                data=df,
                metadata=metadata,
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error extracting current prices: {e}")
            return ExtractionResult(
                success=False,
                data=pd.DataFrame(),
                metadata={"error": str(e)},
                errors=[str(e)]
            )
    
    async def extract_company_fundamentals(self, tickers: List[str]) -> ExtractionResult:
        """
        Extract company fundamental data.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            ExtractionResult with fundamentals DataFrame
        """
        logger.info(f"Extracting fundamentals for {len(tickers)} companies")
        
        all_data = []
        errors = []
        
        for ticker in tickers:
            try:
                info = await self.market_data.get_company_info(ticker)
                
                # Extract relevant fields
                fundamental_data = {
                    "ticker": ticker,
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "extraction_date": datetime.now()
                }
                
                all_data.append(fundamental_data)
                logger.debug(f"✓ Extracted fundamentals for {ticker}")
                
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
                logger.error(f"✗ Error extracting fundamentals for {ticker}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            metadata = {
                "extraction_time": datetime.now().isoformat(),
                "total_tickers": len(tickers),
                "successful_tickers": len(all_data)
            }
            
            logger.info(f"Fundamentals extracted: {len(all_data)}/{len(tickers)}")
            
            return ExtractionResult(
                success=True,
                data=df,
                metadata=metadata,
                errors=errors
            )
        else:
            return ExtractionResult(
                success=False,
                data=pd.DataFrame(),
                metadata={},
                errors=errors
            )


class BatchExtractor:
    """
    Batch extractor for large-scale data extraction.
    """
    
    def __init__(self, market_data_service: IMarketDataService, batch_size: int = 10):
        """
        Initialize batch extractor.
        
        Args:
            market_data_service: Market data service
            batch_size: Number of tickers per batch
        """
        self.extractor = MarketDataExtractor(market_data_service)
        self.batch_size = batch_size
        logger.info(f"Batch Extractor initialized (batch_size={batch_size})")
    
    async def extract_in_batches(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> ExtractionResult:
        """
        Extract data in batches to avoid overwhelming the API.
        
        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined ExtractionResult
        """
        logger.info(f"Batch extraction for {len(tickers)} tickers (batch_size={self.batch_size})")
        
        all_dataframes = []
        all_errors = []
        
        # Split tickers into batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} tickers")
            
            result = await self.extractor.extract_historical_prices(
                tickers=batch,
                start_date=start_date,
                end_date=end_date
            )
            
            if result.success:
                all_dataframes.append(result.data)
            
            all_errors.extend(result.errors)
        
        # Combine all batches
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, axis=1)
            
            metadata = {
                "extraction_time": datetime.now().isoformat(),
                "total_tickers": len(tickers),
                "total_batches": total_batches,
                "total_errors": len(all_errors)
            }
            
            logger.info(f"Batch extraction complete: {len(combined_df.columns)} tickers")
            
            return ExtractionResult(
                success=True,
                data=combined_df,
                metadata=metadata,
                errors=all_errors
            )
        else:
            logger.error("Batch extraction failed: no data extracted")
            return ExtractionResult(
                success=False,
                data=pd.DataFrame(),
                metadata={},
                errors=all_errors
            )
