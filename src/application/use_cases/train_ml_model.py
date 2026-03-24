"""
Use Case: Entraînement du Modèle ML
"""
import mlflow
from datetime import datetime
from typing import List

from src.infrastructure.data.loader import DataLoader
from src.infrastructure.preprocessing.feature_engineering import FeatureEngineer
from src.infrastructure.ml.models.model_factory import create_model
from src.infrastructure.ml.training.trainer import MLTrainer
from src.infrastructure.mlops.mlflow_tracker import MLflowTracker
from src.infrastructure.mlops.model_registry import ModelRegistry


class TrainMLModelUseCase:
    """Use Case pour entraîner les modèles ML de prédiction de rendements"""
    
    def __init__(
        self,
        data_loader: DataLoader,
        feature_engineer: FeatureEngineer,
        mlflow_tracker: MLflowTracker,
        model_registry: ModelRegistry
    ):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.mlflow_tracker = mlflow_tracker
        self.model_registry = model_registry
    
    async def execute(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        model_type: str = 'random_forest',
        target_horizon: int = 20
    ) -> dict:
        """
        Entraîne un modèle ML pour prédire les rendements futurs.
        
        Returns:
            dict avec métriques et model_id
        """
        
        with mlflow.start_run(run_name=f"train_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log params
            mlflow.log_params({
                'model_type': model_type,
                'tickers': ','.join(tickers),
                'target_horizon': target_horizon
            })
            
            results = {}
            
            # Train un modèle par ticker
            for ticker in tickers:
                print(f"\n Training model for {ticker}...")
                
                # Load data
                prices_df = await self.data_loader.load_prices(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Feature engineering
                features_df = self.feature_engineer.create_features(
                    prices_df, 
                    ticker,
                    target_horizon=target_horizon
                )
                
                # Separate X and y
                feature_cols = [col for col in features_df.columns if not col.startswith('target')]
                X = features_df[feature_cols]
                y = features_df['target_20d_return']
                
                #  Create model
                model = create_model(model_type)
                
                #  Train
                trainer = MLTrainer(model)
                metrics = trainer.train_with_cv(X, y, n_splits=5)
                
                #  Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{ticker}_{metric_name}", value)
                
                #  Save model to registry
                model_id = await self.model_registry.save_model(
                    model=model,
                    ticker=ticker,
                    metrics=metrics
                )
                
                results[ticker] = {
                    'model_id': model_id,
                    'metrics': metrics
                }
                
                print(f" {ticker}: CV Direction Accuracy = {metrics['cv_direction_accuracy']:.2%}")
            
            return results