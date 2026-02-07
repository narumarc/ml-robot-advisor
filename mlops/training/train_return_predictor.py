#!/usr/bin/env python
"""
Training script for Return Predictor model.
Usage: python mlops/training/train_return_predictor.py --config config.yaml
"""
import sys
import argparse
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.infrastructure.ml.models.return_predictor import XGBoostReturnPredictor
from src.infrastructure.ml.preprocessing.feature_engineer import FinancialFeatureEngineer
from src.infrastructure.ml.training.cross_validator import TimeSeriesCrossValidator
from src.infrastructure.ml.monitoring.data_quality_checker import DataQualityChecker
from src.infrastructure.data_sources.market_data import YFinanceDataSource
from config.logging_config import setup_logging, get_logger
import mlflow

setup_logging()
logger = get_logger(__name__)


def load_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load market data for training."""
    logger.info(f"Loading data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    data_source = YFinanceDataSource()
    all_data = {}
    
    import asyncio
    for ticker in tickers:
        try:
            df = asyncio.run(data_source.get_historical_prices(
                ticker=ticker,
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date)
            ))
            if not df.empty:
                all_data[ticker] = df['Close']
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
    
    # Combine into single DataFrame
    prices_df = pd.DataFrame(all_data)
    logger.info(f"Loaded data shape: {prices_df.shape}")
    return prices_df


def prepare_features(prices_df: pd.DataFrame) -> tuple:
    """Engineer features and prepare X, y."""
    logger.info("Engineering features...")
    
    engineer = FinancialFeatureEngineer()
    
    # Create features for each ticker
    all_features = []
    all_targets = []
    
    for ticker in prices_df.columns:
        # Create features
        features = engineer.create_technical_indicators(prices_df[[ticker]])
        features = features.dropna()
        
        # Target: next day return
        target = prices_df[ticker].pct_change().shift(-1)
        target = target.loc[features.index]
        
        # Combine
        features['ticker'] = ticker
        features['target'] = target
        
        all_features.append(features)
    
    # Concatenate all
    combined_df = pd.concat(all_features, axis=0).dropna()
    
    # Separate X and y
    X = combined_df.drop(['target', 'ticker'], axis=1)
    y = combined_df['target']
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def check_data_quality(X: pd.DataFrame, y: pd.Series) -> bool:
    """Run data quality checks."""
    logger.info("Running data quality checks...")
    
    # Split for quality check
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Add target back for leakage check
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    checker = DataQualityChecker()
    results = checker.run_all_checks(
        train_data=train_df,
        test_data=test_df,
        feature_cols=list(X.columns),
        target_col='target'
    )
    
    if results["summary"]["critical_issues_count"] > 0:
        logger.error("❌ CRITICAL DATA QUALITY ISSUES FOUND!")
        for issue in results["summary"]["critical_issues"]:
            logger.error(f"  - {issue['type']}: {issue['message']}")
        return False
    
    logger.info("✅ Data quality checks passed")
    return True


def train_model(X: pd.DataFrame, y: pd.Series, config: dict) -> XGBoostReturnPredictor:
    """Train the model."""
    logger.info("Training XGBoost Return Predictor...")
    
    # Initialize model
    model = XGBoostReturnPredictor(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 6),
        learning_rate=config.get('learning_rate', 0.1)
    )
    
    # Train with MLflow tracking
    mlflow.set_experiment("return_prediction")
    
    with mlflow.start_run(run_name=f"return_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(config)
        
        # Train
        metrics = model.train(X, y, validation_split=0.2)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv = TimeSeriesCrossValidator(n_splits=5)
        cv_results = cv.validate(model, X, y)
        
        mlflow.log_metric("cv_mean_score", cv_results['mean_cv_score'])
        mlflow.log_metric("cv_std_score", cv_results['std_cv_score'])
        
        logger.info(f"✅ Training complete - Val MSE: {metrics['val_mse']:.6f}")
        logger.info(f"✅ CV Mean Score: {cv_results['mean_cv_score']:.6f} ± {cv_results['std_cv_score']:.6f}")
    
    return model


def save_model(model: XGBoostReturnPredictor, output_dir: str) -> str:
    """Save trained model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_path / f"return_predictor_{timestamp}.pkl"
    
    model.save(str(model_path))
    logger.info(f"✅ Model saved to {model_path}")
    
    # Also save as "latest"
    latest_path = output_path / "return_predictor_latest.pkl"
    model.save(str(latest_path))
    logger.info(f"✅ Model saved as latest: {latest_path}")
    
    return str(model_path)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train Return Predictor model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="models/return_predictor", help="Output directory")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="MLflow tracking URI")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RETURN PREDICTOR TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    
    # Set MLflow URI
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    try:
        # 1. Load data
        prices_df = load_data(
            tickers=config['tickers'],
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        
        # 2. Prepare features
        X, y = prepare_features(prices_df)
        
        # 3. Data quality checks
        if not check_data_quality(X, y):
            logger.error("❌ Training aborted due to data quality issues")
            sys.exit(1)
        
        # 4. Train model
        model = train_model(X, y, config)
        
        # 5. Save model
        model_path = save_model(model, args.output_dir)
        
        logger.info("="*80)
        logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Model saved to: {model_path}")
        logger.info("="*80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
