"""
Portfolio Endpoints - VERSION COMPLETE AVEC VRAIE OPTIMISATION
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

router = APIRouter()


# ============================================================================
# MODELS DE REQUEST/RESPONSE
# ============================================================================

class OptimizeRequest(BaseModel):
    """Corps de la requête pour optimisation de portfolio"""
    tickers: List[str]
    start_date: str
    end_date: str
    strategy: str = 'markowitz'  # 'markowitz', 'risk_parity', 'cvar'
    solver: str = 'highs'  # 'highs', 'cvxpy', 'gurobi'
    use_ml: bool = False  # Si True, utilise prédictions ML pour rendements espérés
    constraints: Optional[Dict] = None  # Contraintes optionnelles


class OptimizeResponse(BaseModel):
    """Corps de la réponse après optimisation"""
    portfolio_id: str
    weights: Dict[str, float]
    sharpe_ratio: float
    expected_return: float
    volatility: float
    strategy: str
    solver_time: float
    message: str


# ============================================================================
# HELPER FUNCTIONS - CHARGEMENT DONNÉES
# ============================================================================

def load_csv_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Charge les données depuis les fichiers CSV locaux.
    
    Args:
        tickers: Liste des symboles (ex: ['AAPL', 'MSFT'])
        start_date: Date début (format: 'YYYY-MM-DD')
        end_date: Date fin (format: 'YYYY-MM-DD')
    
    Returns:
        DataFrame avec colonnes = tickers, index = dates, valeurs = prix Close
    """
    import os
    
    # Chemin vers les CSV (ajuster selon ton setup)
    data_folder = os.path.join(os.getcwd(), 'data')
    
    price_frames = []
    
    for ticker in tickers:
        csv_path = os.path.join(data_folder, f"{ticker}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
        
        # Lire le CSV
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df = df[['Date', 'Close']]
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df = df.set_index('Date')
        df = df.rename(columns={'Close': ticker})
        
        price_frames.append(df)
    
    # Combiner tous les tickers
    prices = pd.concat(price_frames, axis=1)
    prices = prices.sort_index()
    
    # Remplir les valeurs manquantes (forward fill puis backward fill)
    prices = prices.ffill().bfill()
    
    return prices


# ============================================================================
# OPTIMISATION - MARKOWITZ
# ============================================================================

def optimize_markowitz(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    constraints: Optional[Dict] = None
) -> Dict:
    """
    Optimisation de Markowitz (Mean-Variance).
    Maximise le ratio de Sharpe.
    
    Args:
        expected_returns: Rendements espérés (pd.Series avec index = tickers)
        cov_matrix: Matrice de covariance (pd.DataFrame)
        risk_free_rate: Taux sans risque (annualisé)
        constraints: Dict avec 'max_position_size' et 'min_position_size'
    
    Returns:
        Dict avec 'weights', 'sharpe_ratio', 'expected_return', 'volatility', 'solver_time'
    """
    from scipy.optimize import minimize
    import time
    
    start_time = time.time()
    
    n_assets = len(expected_returns)
    tickers = expected_returns.index.tolist()
    
    # Contraintes par défaut
    max_weight = constraints.get('max_position_size', 0.40) if constraints else 0.40
    min_weight = constraints.get('min_position_size', 0.05) if constraints else 0.05
    
    # Fonction objectif: minimiser -Sharpe ratio
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe  # Négatif car on minimise
    
    # Contraintes
    constraints_list = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Somme des poids = 1
    ]
    
    # Bornes pour chaque poids
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    
    # Poids initiaux (équipondérés)
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Optimisation
    result = minimize(
        neg_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list
    )
    
    if not result.success:
        raise ValueError(f"Optimisation échouée: {result.message}")
    
    # Calculer les métriques finales
    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_std = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    
    solver_time = time.time() - start_time
    
    # Formater les poids en dictionnaire
    weights_dict = {ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)}
    
    return {
        'weights': weights_dict,
        'sharpe_ratio': float(sharpe_ratio),
        'expected_return': float(portfolio_return),
        'volatility': float(portfolio_std),
        'solver_time': solver_time
    }


# ============================================================================
# OPTIMISATION - RISK PARITY
# ============================================================================

def optimize_risk_parity(
    cov_matrix: pd.DataFrame,
    constraints: Optional[Dict] = None
) -> Dict:
    """
    Optimisation Risk Parity.
    Égalise la contribution au risque de chaque actif.
    
    Args:
        cov_matrix: Matrice de covariance
        constraints: Dict avec contraintes optionnelles
    
    Returns:
        Dict avec 'weights', 'sharpe_ratio', 'expected_return', 'volatility', 'solver_time'
    """
    from scipy.optimize import minimize
    import time
    
    start_time = time.time()
    
    n_assets = len(cov_matrix)
    tickers = cov_matrix.index.tolist()
    
    # Contraintes
    max_weight = constraints.get('max_position_size', 0.40) if constraints else 0.40
    min_weight = constraints.get('min_position_size', 0.05) if constraints else 0.05
    
    # Fonction objectif: minimiser la variance des contributions au risque
    def risk_parity_objective(weights):
        portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Contribution au risque de chaque actif
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_std
        
        # Contribution cible (équipondérée)
        target_contrib = portfolio_std / n_assets
        
        # Minimiser la somme des carrés des écarts
        return np.sum((risk_contrib - target_contrib) ** 2)
    
    # Contraintes
    constraints_list = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    
    # Bornes
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    
    # Poids initiaux
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Optimisation
    result = minimize(
        risk_parity_objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list
    )
    
    if not result.success:
        raise ValueError(f"Optimisation Risk Parity échouée: {result.message}")
    
    optimal_weights = result.x
    portfolio_std = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
    
    solver_time = time.time() - start_time
    
    weights_dict = {ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)}
    
    return {
        'weights': weights_dict,
        'sharpe_ratio': 0.0,  # Pas calculable sans rendements espérés
        'expected_return': 0.0,
        'volatility': float(portfolio_std),
        'solver_time': solver_time
    }


# ============================================================================
# OPTIMISATION - CVaR
# ============================================================================

def optimize_cvar(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    constraints: Optional[Dict] = None
) -> Dict:
    """
    Optimisation CVaR (Conditional Value at Risk).
    Minimise les pertes dans les pires scénarios.
    
    Args:
        returns: Rendements historiques (DataFrame avec colonnes = tickers)
        alpha: Niveau de confiance (0.95 = 95%)
        constraints: Dict avec contraintes
    
    Returns:
        Dict avec résultats
    """
    from scipy.optimize import minimize
    import time
    
    start_time = time.time()
    
    n_assets = returns.shape[1]
    n_scenarios = returns.shape[0]
    tickers = returns.columns.tolist()
    
    max_weight = constraints.get('max_position_size', 0.40) if constraints else 0.40
    min_weight = constraints.get('min_position_size', 0.05) if constraints else 0.05
    
    # Fonction objectif: CVaR
    def cvar_objective(weights):
        # Rendements du portfolio pour chaque scénario
        portfolio_returns = np.dot(returns.values, weights)
        
        # Pertes (opposé des rendements)
        losses = -portfolio_returns
        
        # VaR: quantile alpha des pertes
        var = np.percentile(losses, alpha * 100)
        
        # CVaR: moyenne des pertes >= VaR
        cvar = np.mean(losses[losses >= var])
        
        return cvar
    
    # Contraintes
    constraints_list = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Optimisation
    result = minimize(
        cvar_objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list
    )
    
    if not result.success:
        raise ValueError(f"Optimisation CVaR échouée: {result.message}")
    
    optimal_weights = result.x
    portfolio_returns = np.dot(returns.values, optimal_weights)
    
    expected_return = np.mean(portfolio_returns) * 252  # Annualisé
    volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualisé
    sharpe_ratio = expected_return / volatility if volatility > 0 else 0.0
    
    solver_time = time.time() - start_time
    
    weights_dict = {ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)}
    
    return {
        'weights': weights_dict,
        'sharpe_ratio': float(sharpe_ratio),
        'expected_return': float(expected_return),
        'volatility': float(volatility),
        'solver_time': solver_time
    }


# ============================================================================
# ENDPOINT PRINCIPAL
# ============================================================================

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_portfolio(request: OptimizeRequest):
    """
    Optimise un portefeuille selon la stratégie choisie.
    
    Stratégies disponibles:
    - markowitz: Maximise le ratio de Sharpe (rendement/risque)
    - risk_parity: Égalise la contribution au risque de chaque actif
    - cvar: Minimise les pertes dans les pires scénarios
    
    Exemple de requête:
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "strategy": "markowitz",
        "use_ml": false
    }
    """
    try:
        print(f"\n{'='*60}")
        print(f"📊 Optimisation de portfolio demandée")
        print(f"   Tickers: {request.tickers}")
        print(f"   Période: {request.start_date} à {request.end_date}")
        print(f"   Stratégie: {request.strategy}")
        print(f"{'='*60}\n")
        
        # 1. CHARGER LES DONNÉES
        print("📈 Chargement des données depuis CSV...")
        prices_df = load_csv_data(request.tickers, request.start_date, request.end_date)
        print(f"   ✅ {len(prices_df)} jours chargés")
        
        # 2. CALCULER LES RENDEMENTS
        print("📊 Calcul des rendements...")
        returns_df = prices_df.pct_change().dropna()
        
        # 3. CALCULER RENDEMENTS ESPÉRÉS
        if request.use_ml:
            # TODO: Charger les prédictions ML depuis MongoDB
            print("⚠️  ML non encore implémenté, utilisation moyenne historique")
            expected_returns = returns_df.mean() * 252  # Annualisé
        else:
            # Moyenne historique (annualisée)
            expected_returns = returns_df.mean() * 252
        
        # 4. CALCULER MATRICE DE COVARIANCE
        cov_matrix = returns_df.cov() * 252  # Annualisée
        
        # 5. OPTIMISER SELON LA STRATÉGIE
        print(f"⚙️  Optimisation avec {request.strategy}...")
        
        if request.strategy == 'markowitz':
            result = optimize_markowitz(
                expected_returns,
                cov_matrix,
                constraints=request.constraints
            )
        
        elif request.strategy == 'risk_parity':
            result = optimize_risk_parity(
                cov_matrix,
                constraints=request.constraints
            )
        
        elif request.strategy == 'cvar':
            result = optimize_cvar(
                returns_df,
                constraints=request.constraints
            )
        
        else:
            raise ValueError(f"Stratégie inconnue: {request.strategy}")
        
        # 6. CRÉER LA RÉPONSE
        portfolio_id = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"✅ Optimisation terminée!")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"   Rendement esp: {result['expected_return']:.2%}")
        print(f"   Volatilité: {result['volatility']:.2%}")
        print(f"   Temps calcul: {result['solver_time']:.3f}s")
        print(f"{'='*60}\n")
        
        # TODO: Sauvegarder dans MongoDB
        # TODO: Logger dans MLflow
        
        return OptimizeResponse(
            portfolio_id=portfolio_id,
            weights=result['weights'],
            sharpe_ratio=result['sharpe_ratio'],
            expected_return=result['expected_return'],
            volatility=result['volatility'],
            strategy=request.strategy,
            solver_time=result['solver_time'],
            message=f"✅ Portfolio optimisé avec succès ({request.strategy})"
        )
    
    except FileNotFoundError as e:
        print(f"❌ Erreur: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        print(f"❌ Erreur lors de l'optimisation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_portfolios():
    """Liste tous les portfolios sauvegardés"""
    # TODO: Récupérer depuis MongoDB
    return {
        "portfolios": [],
        "message": "📋 Fonction à implémenter (MongoDB)"
    }


@router.get("/{portfolio_id}")
async def get_portfolio(portfolio_id: str):
    """Récupère les détails d'un portfolio par son ID"""
    # TODO: Récupérer depuis MongoDB
    return {
        "portfolio_id": portfolio_id,
        "message": "📊 Fonction à implémenter (MongoDB)"
    }
