"""
Use Case Principal: Optimisation de Portefeuille
"""
import mlflow
from datetime import datetime
from typing import List, Optional

from src.domain.entities.portfolio import Portfolio
from src.domain.value_objects.weights import Weights
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.models.return_predictor import ReturnPredictor
from src.infrastructure.optimization.solver_factory import create_optimizer
from src.infrastructure.data.repository import PortfolioRepository
from src.infrastructure.mlops.mlflow_tracker import MLflowTracker


class OptimizePortfolioUseCase:
    """
    Use Case pour optimiser un portefeuille.
    
    Flow complet:
    1. Load data (ETL)
    2. Predict returns (ML)
    3. Optimize (Solver)
    4. Save to MongoDB
    5. Track in MLflow
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        ml_predictor: ReturnPredictor,
        repository: PortfolioRepository,
        mlflow_tracker: MLflowTracker
    ):
        self.data_loader = data_loader
        self.ml_predictor = ml_predictor
        self.repository = repository
        self.mlflow_tracker = mlflow_tracker
    
    async def execute(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        strategy: str = 'markowitz',
        solver_type: str = 'highs',
        use_ml_predictions: bool = True
    ) -> Portfolio:
        """
        Exécute l'optimisation complète.
        
        Args:
            tickers: Liste des symboles d'actifs
            start_date: Date de début des données
            end_date: Date de fin
            strategy: 'markowitz', 'risk_parity', ou 'cvar'
            solver_type: 'highs', 'cvxpy', ou 'gurobi'
            use_ml_predictions: Si True, utilise ML pour prédire rendements
        
        Returns:
            Portfolio optimisé
        """
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"optimize_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params({
                'tickers': ','.join(tickers),
                'strategy': strategy,
                'solver': solver_type,
                'use_ml': use_ml_predictions
            })
            
            # 1. LOAD DATA
            print(" Loading market data...")
            prices_df = await self.data_loader.load_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            returns_df = prices_df.pct_change().dropna()
            cov_matrix = returns_df.cov() * 252  # Annualize
            
            # 2. PREDICT RETURNS (ML ou historique)
            if use_ml_predictions:
                print(" Predicting returns with ML...")
                expected_returns = await self.ml_predictor.predict(prices_df, tickers)
                mlflow.log_param("prediction_method", "ml")
            else:
                print(" Using historical returns...")
                expected_returns = returns_df.mean() * 252  # Annualize
                mlflow.log_param("prediction_method", "historical")
            
            # Log expected returns
            for ticker, ret in expected_returns.items():
                mlflow.log_metric(f"expected_return_{ticker}", ret)
            
            # 3. OPTIMIZE
            print(f"  Optimizing with {strategy} strategy...")
            optimizer = create_optimizer(solver_type)
            
            result = optimizer.optimize(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                strategy=strategy
            )
            
            # Log optimization results
            mlflow.log_metrics({
                'sharpe_ratio': result.sharpe_ratio,
                'expected_return': result.expected_return,
                'volatility': result.volatility,
                'solver_time': result.solver_time
            })
            
            # Log weights
            for ticker, weight in result.weights.items():
                mlflow.log_metric(f"weight_{ticker}", weight)
            
            # 4. CREATE PORTFOLIO ENTITY
            portfolio = Portfolio(
                portfolio_id=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"{strategy.capitalize()} Portfolio",
                tickers=tickers,
                weights=result.weights,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                strategy=strategy,
                expected_return=result.expected_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio
            )
            
            # 5. SAVE TO MONGODB
            print("💾 Saving to MongoDB...")
            await self.repository.save(portfolio)
            
            # 6. TRACK IN MLFLOW
            self.mlflow_tracker.log_portfolio(portfolio)
            
            print(f" Portfolio optimized successfully!")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.4f}")
            print(f"   Expected Return: {result.expected_return:.2%}")
            print(f"   Volatility: {result.volatility:.2%}")
            
            return portfolio