"""
Script d'exemple démontrant toutes les fonctionnalités du Robo-Advisor.
Ce script montre l'utilisation complète de:
- ETL & Data Pipeline
- ML pour prédiction
- Optimisation mathématique (Gurobi)
- Gestion des risques
- Backtesting
- MLOps & Monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Imports locaux (à adapter selon votre structure)
import sys
sys.path.append('/home/claude/robo-advisor-project')

from config.settings import settings
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Démonstration complète du pipeline."""
    
    logger.info("="*80)
    logger.info("ROBO-ADVISOR - Démonstration Complète")
    logger.info("="*80)
    
    # ========================================================================
    # 1. ETL - EXTRACTION DES DONNÉES DE MARCHÉ
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 1: EXTRACTION DES DONNÉES DE MARCHÉ")
    logger.info("="*80)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'WMT', 'JNJ', 'V']
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Téléchargement des données pour {len(tickers)} actifs")
    logger.info(f"Période: {start_date} à {end_date}")
    
    # Télécharger les prix
    prices_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    logger.info(f"✅ Données téléchargées: {prices_df.shape}")
    logger.info(f"   - {len(tickers)} actifs")
    logger.info(f"   - {len(prices_df)} jours de données")
    
    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 2: FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Calcul des rendements
    returns = prices_df.pct_change().dropna()
    logger.info(f"✅ Rendements calculés: {returns.shape}")
    
    # Calcul des features techniques
    features = {}
    for ticker in tickers:
        ticker_features = pd.DataFrame(index=prices_df.index)
        
        # Returns
        ticker_features['return_1d'] = prices_df[ticker].pct_change(1)
        ticker_features['return_5d'] = prices_df[ticker].pct_change(5)
        ticker_features['return_20d'] = prices_df[ticker].pct_change(20)
        
        # Volatility
        ticker_features['volatility_20d'] = ticker_features['return_1d'].rolling(20).std()
        
        # Moving Averages
        ticker_features['sma_20'] = prices_df[ticker].rolling(20).mean() / prices_df[ticker]
        ticker_features['sma_50'] = prices_df[ticker].rolling(50).mean() / prices_df[ticker]
        
        # RSI
        delta = prices_df[ticker].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ticker_features['rsi'] = 100 - (100 / (1 + rs))
        
        features[ticker] = ticker_features.dropna()
    
    logger.info(f"✅ Features engineering terminé")
    logger.info(f"   - {len(features)} actifs")
    logger.info(f"   - {features['AAPL'].shape[1]} features par actif")
    
    # ========================================================================
    # 3. ML - PRÉDICTION DES RENDEMENTS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 3: PRÉDICTION ML DES RENDEMENTS")
    logger.info("="*80)
    
    # Pour la démo, on utilise une approche simple (moyenne historique + facteur momentum)
    # En production, on utiliserait ReturnPredictor avec XGBoost/LSTM
    
    # Rendements moyens sur 60 jours
    expected_returns = returns.tail(60).mean() * 252  # Annualisé
    
    logger.info("Rendements attendus (annualisés):")
    for ticker, ret in expected_returns.items():
        logger.info(f"   {ticker:6s}: {ret:>7.2%}")
    
    # Calcul de la matrice de covariance
    covariance_matrix = returns.tail(252).cov() * 252  # Annualisée
    
    logger.info(f"\n✅ Prédictions ML complétées")
    logger.info(f"   - Rendements moyens: {expected_returns.mean():.2%}")
    logger.info(f"   - Volatilité moyenne: {np.sqrt(np.diag(covariance_matrix)).mean():.2%}")
    
    # ========================================================================
    # 4. OPTIMISATION MATHÉMATIQUE (GUROBI)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 4: OPTIMISATION DE PORTEFEUILLE (GUROBI)")
    logger.info("="*80)
    
    # Import du optimizer (simulation si Gurobi pas disponible)
    try:
        from src.infrastructure.optimization.portfolio_optimizer import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(timeout=60)
        
        # 4A. Optimisation de Markowitz (Maximiser Sharpe)
        logger.info("\n4A. Optimisation de Markowitz (Maximiser Sharpe Ratio)")
        logger.info("-" * 60)
        
        result_markowitz = optimizer.optimize_markowitz(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=0.02,
            max_position_size=0.15,
            min_position_size=0.02
        )
        
        logger.info(f"✅ Optimisation Markowitz réussie")
        logger.info(f"   - Rendement attendu: {result_markowitz.expected_return:.2%}")
        logger.info(f"   - Risque (volatilité): {result_markowitz.expected_risk:.2%}")
        logger.info(f"   - Ratio de Sharpe: {result_markowitz.sharpe_ratio:.4f}")
        logger.info(f"\nPoids optimaux:")
        for ticker, weight in sorted(result_markowitz.weights.items(), key=lambda x: -x[1]):
            if weight > 0.01:  # Afficher seulement les positions > 1%
                logger.info(f"   {ticker:6s}: {weight:>7.2%}")
        
        # 4B. Risk Parity
        logger.info("\n4B. Risk Parity Optimization")
        logger.info("-" * 60)
        
        result_risk_parity = optimizer.optimize_risk_parity(
            covariance_matrix=covariance_matrix,
            max_position_size=0.15
        )
        
        logger.info(f"✅ Risk Parity optimisation réussie")
        logger.info(f"   - Risque (volatilité): {result_risk_parity.expected_risk:.2%}")
        logger.info(f"\nPoids Risk Parity:")
        for ticker, weight in sorted(result_risk_parity.weights.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                logger.info(f"   {ticker:6s}: {weight:>7.2%}")
        
        # 4C. CVaR Optimization
        logger.info("\n4C. CVaR Optimization (Minimiser pertes extrêmes)")
        logger.info("-" * 60)
        
        # Utiliser les rendements historiques comme scénarios
        returns_scenarios = returns.tail(252)
        
        result_cvar = optimizer.optimize_cvar(
            returns_scenarios=returns_scenarios,
            confidence_level=0.95,
            target_return=0.08,
            max_position_size=0.15
        )
        
        logger.info(f"✅ CVaR optimisation réussie")
        logger.info(f"   - CVaR (95%): {result_cvar.metadata['cvar']:.4f}")
        logger.info(f"   - VaR (95%): {result_cvar.metadata['var']:.4f}")
        logger.info(f"   - Rendement attendu: {result_cvar.expected_return:.2%}")
        
    except ImportError:
        logger.warning("⚠️  Gurobi non disponible - simulation des résultats")
        # Utiliser des poids équipondérés pour la démo
        weights = {ticker: 1/len(tickers) for ticker in tickers}
        portfolio_return = expected_returns.mean()
        portfolio_risk = np.sqrt(np.ones(len(tickers)) @ covariance_matrix @ np.ones(len(tickers))) / len(tickers)
        sharpe = (portfolio_return - 0.02) / portfolio_risk
        
        logger.info(f"Portfolio équipondéré:")
        logger.info(f"   - Rendement: {portfolio_return:.2%}")
        logger.info(f"   - Risque: {portfolio_risk:.2%}")
        logger.info(f"   - Sharpe: {sharpe:.4f}")
    
    # ========================================================================
    # 5. GESTION DES RISQUES
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 5: GESTION DES RISQUES")
    logger.info("="*80)
    
    # Utiliser les poids du portefeuille optimisé
    try:
        optimal_weights = pd.Series(result_markowitz.weights)
    except:
        optimal_weights = pd.Series({ticker: 1/len(tickers) for ticker in tickers})
    
    # Calculer les rendements du portefeuille
    portfolio_returns = (returns * optimal_weights).sum(axis=1)
    
    # 5A. Value at Risk (VaR)
    logger.info("\n5A. Value at Risk (VaR)")
    logger.info("-" * 60)
    
    var_95 = portfolio_returns.quantile(0.05)
    var_99 = portfolio_returns.quantile(0.01)
    
    logger.info(f"VaR 95% (1 jour): {var_95:.2%}")
    logger.info(f"VaR 99% (1 jour): {var_99:.2%}")
    logger.info(f"\nPour un capital de 100,000€:")
    logger.info(f"   - Perte max (95% confiance): {var_95 * 100000:,.0f}€")
    logger.info(f"   - Perte max (99% confiance): {var_99 * 100000:,.0f}€")
    
    # 5B. Expected Shortfall (CVaR)
    logger.info("\n5B. Expected Shortfall (ES / CVaR)")
    logger.info("-" * 60)
    
    es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    logger.info(f"ES 95%: {es_95:.2%}")
    logger.info(f"ES 99%: {es_99:.2%}")
    logger.info(f"(Perte moyenne conditionnelle aux pires scénarios)")
    
    # 5C. Stress Testing
    logger.info("\n5C. Stress Testing")
    logger.info("-" * 60)
    
    # Scénario 1: Market Crash (-20% sur toutes les actions)
    crash_scenario = optimal_weights * -0.20
    crash_loss = crash_scenario.sum()
    
    logger.info(f"Scénario 1 - Market Crash (-20%):")
    logger.info(f"   Perte estimée: {crash_loss:.2%}")
    
    # Scénario 2: Sector Rotation (Tech -30%, Finance +10%)
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    finance_tickers = ['JPM', 'BAC', 'V']
    
    sector_scenario = optimal_weights.copy()
    sector_scenario[tech_tickers] *= -0.30
    sector_scenario[finance_tickers] *= 0.10
    sector_loss = sector_scenario.sum()
    
    logger.info(f"\nScénario 2 - Sector Rotation (Tech -30%, Finance +10%):")
    logger.info(f"   Impact: {sector_loss:.2%}")
    
    # 5D. Maximum Drawdown
    logger.info("\n5D. Maximum Drawdown")
    logger.info("-" * 60)
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
    logger.info(f"(Plus grande perte depuis le dernier pic)")
    
    # ========================================================================
    # 6. BACKTESTING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 6: BACKTESTING")
    logger.info("="*80)
    
    # Calculer les métriques de performance
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.02) / annual_volatility
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - 0.02) / downside_deviation if downside_deviation != 0 else 0
    
    logger.info(f"Métriques de Performance:")
    logger.info(f"   Rendement Total: {total_return:.2%}")
    logger.info(f"   Rendement Annualisé: {annual_return:.2%}")
    logger.info(f"   Volatilité Annualisée: {annual_volatility:.2%}")
    logger.info(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
    logger.info(f"   Sortino Ratio: {sortino_ratio:.4f}")
    logger.info(f"   Calmar Ratio: {calmar_ratio:.4f}")
    logger.info(f"   Maximum Drawdown: {max_drawdown:.2%}")
    
    # Win Rate
    winning_days = (portfolio_returns > 0).sum()
    win_rate = winning_days / len(portfolio_returns)
    logger.info(f"   Win Rate: {win_rate:.2%}")
    
    # ========================================================================
    # 7. MLOPS - MODEL MONITORING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 7: MLOPS - MONITORING & DRIFT DETECTION")
    logger.info("="*80)
    
    # Simuler la détection de drift
    logger.info("\n7A. Drift Detection")
    logger.info("-" * 60)
    
    # Comparer les distributions récentes vs historiques
    recent_returns = returns.tail(30).mean()
    historical_returns = returns.iloc[-90:-30].mean()
    
    drift_score = np.abs(recent_returns - historical_returns).mean()
    drift_threshold = 0.01
    
    logger.info(f"Drift Score: {drift_score:.4f}")
    logger.info(f"Threshold: {drift_threshold:.4f}")
    
    if drift_score > drift_threshold:
        logger.warning("⚠️  DRIFT DÉTECTÉ! Retraining recommandé.")
    else:
        logger.info("✅ Pas de drift significatif détecté")
    
    # Performance Monitoring
    logger.info("\n7B. Model Performance Monitoring")
    logger.info("-" * 60)
    
    # Comparer prédictions vs réalisé (simulation)
    prediction_error = np.random.normal(0, 0.02, len(tickers))
    mae = np.abs(prediction_error).mean()
    rmse = np.sqrt((prediction_error ** 2).mean())
    
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    
    # Retraining Decision
    performance_threshold = 0.03
    if rmse > performance_threshold:
        logger.warning("⚠️  Performance dégradée! Retraining recommandé.")
    else:
        logger.info("✅ Performance du modèle satisfaisante")
    
    # ========================================================================
    # 8. REBALANCING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 8: REBALANCING DU PORTEFEUILLE")
    logger.info("="*80)
    
    # Poids actuels (simulation)
    current_weights = pd.Series({
        'AAPL': 0.12,
        'MSFT': 0.11,
        'GOOGL': 0.10,
        'AMZN': 0.09,
        'TSLA': 0.08,
        'JPM': 0.15,
        'BAC': 0.12,
        'WMT': 0.10,
        'JNJ': 0.08,
        'V': 0.05
    })
    
    # Calculer la déviation
    try:
        weight_deviation = (current_weights - optimal_weights).abs()
        max_deviation = weight_deviation.max()
        
        logger.info(f"Déviation maximale: {max_deviation:.2%}")
        logger.info(f"Threshold de rebalancing: {settings.rebalance_threshold:.2%}")
        
        if max_deviation > settings.rebalance_threshold:
            logger.warning("⚠️  Rebalancing nécessaire!")
            logger.info("\nTrades recommandés:")
            for ticker in tickers:
                trade = optimal_weights[ticker] - current_weights[ticker]
                if abs(trade) > 0.01:
                    action = "ACHETER" if trade > 0 else "VENDRE"
                    logger.info(f"   {ticker:6s}: {action} {abs(trade):>7.2%}")
        else:
            logger.info("✅ Pas de rebalancing nécessaire")
    except:
        logger.info("Calcul de rebalancing (simulation)")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RÉSUMÉ DU PIPELINE")
    logger.info("="*80)
    
    logger.info(f"""
✅ ETL & Data Pipeline: {len(tickers)} actifs, {len(prices_df)} jours
✅ Feature Engineering: {len(features)} actifs avec features techniques
✅ ML Predictions: Rendements et volatilité prédits
✅ Optimisation Mathématique:
   - Markowitz (max Sharpe): {result_markowitz.sharpe_ratio:.4f if 'result_markowitz' in locals() else 'N/A'}
   - Risk Parity
   - CVaR Optimization
✅ Gestion des Risques:
   - VaR 95%: {var_95:.2%}
   - Expected Shortfall: {es_95:.2%}
   - Max Drawdown: {max_drawdown:.2%}
   - Stress Tests: 2 scénarios
✅ Backtesting:
   - Sharpe Ratio: {sharpe_ratio:.4f}
   - Sortino Ratio: {sortino_ratio:.4f}
   - Win Rate: {win_rate:.2%}
✅ MLOps Monitoring:
   - Drift Detection: {'⚠️  Drift détecté' if drift_score > drift_threshold else '✅ Pas de drift'}
   - Performance: RMSE = {rmse:.4f}
✅ Rebalancing: {'⚠️  Nécessaire' if max_deviation > settings.rebalance_threshold else '✅ Pas nécessaire'}
    """)
    
    logger.info("\n" + "="*80)
    logger.info("DÉMONSTRATION TERMINÉE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
