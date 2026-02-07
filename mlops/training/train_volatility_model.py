#!/usr/bin/env python
"""Train Volatility Predictor model."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.infrastructure.ml.models.volatility_predictor import GARCHVolatilityPredictor
from src.infrastructure.data_sources.market_data import YFinanceDataSource
from config.logging_config import setup_logging, get_logger
import pandas as pd
import mlflow

setup_logging()
logger = get_logger(__name__)

def main():
    logger.info("Training GARCH Volatility Model...")
    
    # Load data
    data_source = YFinanceDataSource()
    import asyncio
    prices = asyncio.run(data_source.get_historical_prices(
        'SPY', pd.to_datetime('2020-01-01'), pd.to_datetime('2024-01-01')
    ))
    returns = prices['Close'].pct_change().dropna()
    
    # Train
    model = GARCHVolatilityPredictor(p=1, q=1)
    
    mlflow.set_experiment("volatility_prediction")
    with mlflow.start_run():
        metrics = model.train(returns)
        mlflow.log_metrics(metrics)
        model.fitted_model.save('models/volatility/garch_model')
    
    logger.info(f"✅ Volatility model trained - AIC: {metrics['aic']:.2f}")

if __name__ == "__main__":
    main()
