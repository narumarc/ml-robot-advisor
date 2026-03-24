#!/usr/bin/env python3
"""
ML-Powered Robo-Advisor - Script Complet
Entraîne des modèles ML pour prédire les rendements et optimise le portfolio
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

sys.path.append('/home/narisoa/LocInstall/robo-advisor-project/robo-advisor-project')

from config.settings import settings
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_technical_features(prices_df, ticker):
    """
    Crée des features techniques pour un ticker.
    
    Features créées:
    - Returns (1d, 5d, 20d, 60d)
    - Volatilité (rolling std)
    - Moving averages (SMA 20, 50, 200)
    - RSI
    - MACD
    - Momentum
    
    Returns:
        DataFrame avec features + target (future 20d return)
    """
    df = pd.DataFrame(index=prices_df.index)
    prices = prices_df[ticker]
    
    # Returns
    df['return_1d'] = prices.pct_change(1)
    df['return_5d'] = prices.pct_change(5)
    df['return_20d'] = prices.pct_change(20)
    df['return_60d'] = prices.pct_change(60)
    
    # Volatility
    df['vol_5d'] = df['return_1d'].rolling(5).std()
    df['vol_20d'] = df['return_1d'].rolling(20).std()
    df['vol_60d'] = df['return_1d'].rolling(60).std()
    
    # Moving Averages
    df['sma_20'] = prices.rolling(20).mean()
    df['sma_50'] = prices.rolling(50).mean()
    df['sma_200'] = prices.rolling(200).mean()
    
    # Price relative to MAs
    df['price_to_sma20'] = prices / df['sma_20']
    df['price_to_sma50'] = prices / df['sma_50']
    df['price_to_sma200'] = prices / df['sma_200']
    
    # RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Momentum
    df['momentum_5d'] = prices / prices.shift(5) - 1
    df['momentum_20d'] = prices / prices.shift(20) - 1
    
    # Bollinger Bands
    bb_middle = prices.rolling(20).mean()
    bb_std = prices.rolling(20).std()
    df['bb_upper'] = bb_middle + 2 * bb_std
    df['bb_lower'] = bb_middle - 2 * bb_std
    df['bb_position'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Target: Future 20-day return (annualized)
    df['target_20d_return'] = prices.pct_change(20).shift(-20)
    
    return df.dropna()


# ============================================================================
# ML MODEL TRAINING
# ============================================================================

def train_return_predictor(prices_df, ticker, save_model=True):
    """
    Entraîne un modèle Random Forest pour prédire les rendements futurs.
    
    Args:
        prices_df: DataFrame avec prix de tous les tickers
        ticker: Ticker à prédire
        save_model: Si True, sauvegarde dans MLflow
        
    Returns:
        dict avec model, feature_names, metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f" Entraînement ML - Return Predictor pour {ticker}")
    logger.info(f"{'='*70}")
    
    # Create features
    df = create_technical_features(prices_df, ticker)
    
    # Separate X and y
    feature_cols = [col for col in df.columns if not col.startswith('target')]
    X = df[feature_cols]
    y = df['target_20d_return']
    
    logger.info(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
    logger.info(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_results = []
    
    with mlflow.start_run(run_name=f"ml_returns_{ticker}", nested=True):
        
        # Log parameters
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("target", "20d_return")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Direction accuracy (très important pour trading!)
            direction_actual = (y_test > 0).astype(int)
            direction_pred = (y_pred > 0).astype(int)
            direction_acc = (direction_actual == direction_pred).mean()
            
            cv_results.append({
                'fold': fold + 1,
                'rmse': rmse,
                'mae': mae,
                'direction_acc': direction_acc
            })
            
            logger.info(f"  Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, "
                       f"Dir.Acc={direction_acc:.2%}")
        
        # Average metrics
        avg_rmse = np.mean([r['rmse'] for r in cv_results])
        avg_mae = np.mean([r['mae'] for r in cv_results])
        avg_dir_acc = np.mean([r['direction_acc'] for r in cv_results])
        
        # Log CV metrics
        mlflow.log_metric("cv_rmse", avg_rmse)
        mlflow.log_metric("cv_mae", avg_mae)
        mlflow.log_metric("cv_direction_accuracy", avg_dir_acc)
        
        logger.info(f"\n Cross-Validation Results:")
        logger.info(f"   RMSE: {avg_rmse:.4f}")
        logger.info(f"   MAE: {avg_mae:.4f}")
        logger.info(f"   Direction Accuracy: {avg_dir_acc:.2%}")
        
        # Train final model on all data
        final_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        final_model.fit(X, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"   {row['feature']:20s}: {row['importance']:.4f}")
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        
        # Save model
        if save_model:
            mlflow.sklearn.log_model(final_model, f"model_{ticker}")
            logger.info(f" Model saved in MLflow")
        
        return {
            'model': final_model,
            'features': feature_cols,
            'cv_rmse': avg_rmse,
            'cv_mae': avg_mae,
            'cv_direction_acc': avg_dir_acc,
            'feature_importance': feature_importance
        }


# ============================================================================
# PREDICTION & OPTIMIZATION
# ============================================================================

def predict_expected_returns(models_dict, prices_df, tickers):
    """
    Prédit les rendements espérés pour tous les tickers.
    
    Args:
        models_dict: Dict {ticker: model_info}
        prices_df: Prix actuels
        tickers: Liste des tickers
        
    Returns:
        pd.Series avec rendements espérés prédits
    """
    logger.info(f"\n{'='*70}")
    logger.info(f" Prédiction des Rendements Espérés")
    logger.info(f"{'='*70}")
    
    predicted_returns = {}
    
    for ticker in tickers:
        # Get latest features
        df_features = create_technical_features(prices_df, ticker)
        latest_features = df_features[models_dict[ticker]['features']].iloc[-1]
        
        # Predict 20-day return
        pred_return_20d = models_dict[ticker]['model'].predict([latest_features])[0]
        
        # Annualize (252 trading days / 20 days)
        annual_return = pred_return_20d * (252 / 20)
        
        predicted_returns[ticker] = annual_return
        
        logger.info(f"   {ticker:6s}: {annual_return:>7.2%}")
    
    return pd.Series(predicted_returns)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def load_local_csv_data(tickers, start_date, end_date, data_folder="notebooks/data"):
    """Charge les CSV locaux."""
    price_frames = []
    for ticker in tickers:
        file_path = os.path.join(data_folder, f"{ticker}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV pour {ticker} non trouvé: {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df[["Date", "Close"]]
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        df = df.set_index("Date").rename(columns={"Close": ticker})
        price_frames.append(df)
    
    prices = pd.concat(price_frames, axis=1).sort_index()
    return prices.ffill().bfill()


def main():
    """Workflow ML complet."""
    
    logger.info("="*80)
    logger.info(" ML-POWERED ROBO-ADVISOR")
    logger.info("="*80)
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ml-portfolio-optimization")
    
    with mlflow.start_run(run_name=f"ml_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # ================================================================
        # 1. LOAD DATA
        # ================================================================
        logger.info(f"\n{'='*70}")
        logger.info("CHARGEMENT DES DONNÉES")
        logger.info(f"{'='*70}")
        
        tickers = ['AAPL', 'MSFT', 'DIS', 'JNJ', 'JPM', 'BA', 'PG', 'XOM']
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        mlflow.log_param("tickers", ",".join(tickers))
        mlflow.log_param("ml_enabled", True)
        
        prices_df = load_local_csv_data(tickers, start_date, end_date)
        returns = prices_df.pct_change().dropna()
        
        logger.info(f" {len(prices_df)} jours, {len(tickers)} actifs")
        
        # ================================================================
        # 2. TRAIN ML MODELS
        # ================================================================
        logger.info(f"\n{'='*70}")
        logger.info("ENTRAÎNEMENT DES MODÈLES ML")
        logger.info(f"{'='*70}")
        
        models = {}
        for ticker in tickers:
            model_info = train_return_predictor(prices_df, ticker, save_model=True)
            models[ticker] = model_info
            
            # Log per-ticker metrics
            mlflow.log_metric(f"{ticker}_cv_direction_acc", model_info['cv_direction_acc'])
        
        avg_direction_acc = np.mean([m['cv_direction_acc'] for m in models.values()])
        mlflow.log_metric("avg_ml_direction_accuracy", avg_direction_acc)
        
        logger.info(f"\n Tous les modèles entraînés!")
        logger.info(f"   Direction Accuracy Moyenne: {avg_direction_acc:.2%}")
        
        # ================================================================
        # 3. PREDICT EXPECTED RETURNS
        # ================================================================
        ml_expected_returns = predict_expected_returns(models, prices_df, tickers)
        
        # Baseline (historical mean)
        baseline_returns = returns.tail(60).mean() * 252
        
        # Compare
        logger.info(f"\n{'='*70}")
        logger.info("COMPARAISON ML vs BASELINE")
        logger.info(f"{'='*70}")
        logger.info(f"{'Ticker':<8} {'ML Pred':>10} {'Baseline':>10} {'Diff':>10}")
        logger.info(f"{'-'*40}")
        for ticker in tickers:
            ml_ret = ml_expected_returns[ticker]
            base_ret = baseline_returns[ticker]
            diff = ml_ret - base_ret
            logger.info(f"{ticker:<8} {ml_ret:>9.2%} {base_ret:>9.2%} {diff:>9.2%}")
            
            # Log comparison
            mlflow.log_metric(f"{ticker}_ml_vs_baseline", diff)
        
        # ================================================================
        # 4. OPTIMIZATION WITH ML PREDICTIONS
        # ================================================================
        logger.info(f"\n{'='*70}")
        logger.info(" OPTIMISATION (ML-POWERED)")
        logger.info(f"{'='*70}")
        
        from src.infrastructure.optimization.solver_factory import create_optimizer
        optimizer = create_optimizer('highs')
        
        cov_matrix = returns.tail(252).cov() * 252
        
        # Optimize with ML predictions
        result_ml = optimizer.optimize_markowitz(
            expected_returns=ml_expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=0.02,
            constraints={
                'max_position_size': 0.40,
                'min_position_size': 0.05
            }
        )
        
        # Optimize with baseline
        result_baseline = optimizer.optimize_markowitz(
            expected_returns=baseline_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=0.02,
            constraints={
                'max_position_size': 0.40,
                'min_position_size': 0.05
            }
        )
        
        # Log results
        mlflow.log_metric("sharpe_ml", result_ml.sharpe_ratio)
        mlflow.log_metric("sharpe_baseline", result_baseline.sharpe_ratio)
        mlflow.log_metric("sharpe_improvement", 
                         result_ml.sharpe_ratio - result_baseline.sharpe_ratio)
        
        mlflow.log_metric("return_ml", result_ml.expected_return)
        mlflow.log_metric("return_baseline", result_baseline.expected_return)
        
        mlflow.log_metric("vol_ml", result_ml.volatility)
        mlflow.log_metric("vol_baseline", result_baseline.volatility)
        
        logger.info(f"\n RÉSULTATS OPTIMISATION:")
        logger.info(f"\n   ML-POWERED:")
        logger.info(f"      Sharpe: {result_ml.sharpe_ratio:.4f}")
        logger.info(f"      Return: {result_ml.expected_return:.2%}")
        logger.info(f"      Vol: {result_ml.volatility:.2%}")
        
        logger.info(f"\n   BASELINE:")
        logger.info(f"      Sharpe: {result_baseline.sharpe_ratio:.4f}")
        logger.info(f"      Return: {result_baseline.expected_return:.2%}")
        logger.info(f"      Vol: {result_baseline.volatility:.2%}")
        
        improvement = result_ml.sharpe_ratio - result_baseline.sharpe_ratio
        logger.info(f"\n   AMÉLIORATION: {improvement:+.4f} ({improvement/result_baseline.sharpe_ratio:+.1%})")
        
        # Weights comparison
        logger.info(f"\n{'='*70}")
        logger.info("ALLOCATION COMPARÉE")
        logger.info(f"{'='*70}")
        logger.info(f"{'Ticker':<8} {'ML':>10} {'Baseline':>10} {'Diff':>10}")
        logger.info(f"{'-'*40}")
        for ticker in tickers:
            ml_w = result_ml.weights.get(ticker, 0)
            base_w = result_baseline.weights.get(ticker, 0)
            diff = ml_w - base_w
            if ml_w > 0.01 or base_w > 0.01:
                logger.info(f"{ticker:<8} {ml_w:>9.2%} {base_w:>9.2%} {diff:>9.2%}")
        
        # ================================================================
        # SUMMARY
        # ================================================================
        logger.info(f"\n{'='*80}")
        logger.info(" WORKFLOW ML TERMINÉ")
        logger.info(f"{'='*80}")
        logger.info(f"Modèles entraînés: {len(models)}")
        logger.info(f" Direction Accuracy: {avg_direction_acc:.2%}")
        logger.info(f"Sharpe Improvement: {improvement:+.4f}")
        logger.info(f"\nMLflow UI: mlflow ui --port 5000")
        logger.info(f"http://localhost:5000")
        logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
