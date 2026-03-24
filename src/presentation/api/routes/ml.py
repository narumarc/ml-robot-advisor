"""
Machine Learning Endpoints - VERSION COMPLETE AVEC VRAI ENTRAINEMENT
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

router = APIRouter()


# ============================================================================
# MODELS DE REQUEST/RESPONSE
# ============================================================================

class TrainRequest(BaseModel):
    """Corps de la requête pour entraînement ML"""
    tickers: List[str]
    start_date: str
    end_date: str
    model_type: str = 'random_forest'  # Type de modèle ML
    target_horizon: int = 20  # Nombre de jours à prédire (défaut: 20 jours)
    n_estimators: int = 100  # Nombre d'arbres pour Random Forest
    max_depth: int = 10  # Profondeur max des arbres


class TrainResponse(BaseModel):
    """Corps de la réponse après entraînement"""
    models: Dict[str, Dict]  # Un modèle par ticker
    message: str
    training_time: float


# ============================================================================
# FEATURE ENGINEERING - CRÉATION DES FEATURES TECHNIQUES
# ============================================================================

def create_technical_features(prices: pd.Series, target_horizon: int = 20) -> pd.DataFrame:
    """
    Crée les features techniques pour le ML.
    
    Features créées:
    - Returns (rendements sur 1, 5, 20, 60 jours)
    - Volatilité (écart-type roulant sur 5, 20, 60 jours)
    - Moving Averages (moyennes mobiles 20, 50, 200 jours)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Momentum
    - Bollinger Bands
    
    Args:
        prices: Série des prix (pd.Series avec index = dates)
        target_horizon: Horizon de prédiction en jours
    
    Returns:
        DataFrame avec toutes les features + la target
    """
    df = pd.DataFrame(index=prices.index)
    
    # ========== RENDEMENTS (RETURNS) ==========
    df['return_1d'] = prices.pct_change(1)
    df['return_5d'] = prices.pct_change(5)
    df['return_20d'] = prices.pct_change(20)
    df['return_60d'] = prices.pct_change(60)
    
    # ========== VOLATILITÉ ==========
    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    df['volatility_60d'] = df['return_1d'].rolling(60).std()
    
    # ========== MOYENNES MOBILES (MOVING AVERAGES) ==========
    df['sma_20'] = prices.rolling(20).mean()
    df['sma_50'] = prices.rolling(50).mean()
    df['sma_200'] = prices.rolling(200).mean()
    
    # Position du prix par rapport aux moyennes mobiles
    df['price_to_sma20'] = prices / df['sma_20']
    df['price_to_sma50'] = prices / df['sma_50']
    df['price_to_sma200'] = prices / df['sma_200']
    
    # ========== RSI (RELATIVE STRENGTH INDEX) ==========
    # Mesure de survente/surachat (0-100)
    # RSI < 30 = survente (probable rebond), RSI > 70 = surachat (probable baisse)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ========== MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE) ==========
    # Indicateur de momentum et tendance
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # ========== MOMENTUM ==========
    df['momentum_5d'] = prices / prices.shift(5) - 1
    df['momentum_20d'] = prices / prices.shift(20) - 1
    
    # ========== BOLLINGER BANDS ==========
    # Bandes de volatilité autour de la moyenne mobile
    sma_20 = prices.rolling(20).mean()
    std_20 = prices.rolling(20).std()
    
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_position'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ========== TARGET (VARIABLE À PRÉDIRE) ==========
    # Rendement futur sur target_horizon jours
    df['target'] = prices.pct_change(target_horizon).shift(-target_horizon)
    
    # Supprimer les lignes avec valeurs manquantes
    df = df.dropna()
    
    return df


# ============================================================================
# ENTRAÎNEMENT DU MODÈLE
# ============================================================================

def train_random_forest_model(
    features_df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    n_splits: int = 5
) -> Dict:
    """
    Entraîne un modèle Random Forest avec cross-validation.
    
    Args:
        features_df: DataFrame avec features + target
        n_estimators: Nombre d'arbres dans la forêt
        max_depth: Profondeur maximale des arbres
        n_splits: Nombre de splits pour la cross-validation
    
    Returns:
        Dict avec le modèle entraîné et les métriques
    """
    import time
    start_time = time.time()
    
    # Séparer features (X) et target (y)
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols]
    y = features_df['target']
    
    print(f"   📊 Données: {len(X)} échantillons, {len(feature_cols)} features")
    
    # Cross-validation temporelle (Time Series Split)
    # Important: ne pas mélanger les données pour les séries temporelles!
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'rmse': [],
        'mae': [],
        'direction_accuracy': []  # Métrique importante pour le trading!
    }
    
    print(f"   🔄 Cross-validation avec {n_splits} folds...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Créer et entraîner le modèle
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1  # Utilise tous les cores CPU
        )
        
        model.fit(X_train, y_train)
        
        # Prédictions sur test set
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Direction accuracy: est-ce qu'on prédit correctement la direction (hausse/baisse)?
        direction_actual = (y_test > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_acc = (direction_actual == direction_pred).mean()
        
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        cv_scores['direction_accuracy'].append(direction_acc)
        
        print(f"      Fold {fold+1}: RMSE={rmse:.4f}, Direction Acc={direction_acc:.2%}")
    
    # Calculer les moyennes
    avg_metrics = {
        'cv_rmse': float(np.mean(cv_scores['rmse'])),
        'cv_rmse_std': float(np.std(cv_scores['rmse'])),
        'cv_mae': float(np.mean(cv_scores['mae'])),
        'cv_direction_accuracy': float(np.mean(cv_scores['direction_accuracy'])),
        'cv_direction_accuracy_std': float(np.std(cv_scores['direction_accuracy']))
    }
    
    print(f"   ✅ CV Direction Accuracy: {avg_metrics['cv_direction_accuracy']:.2%} "
          f"(±{avg_metrics['cv_direction_accuracy_std']:.2%})")
    
    # Entraîner le modèle final sur TOUTES les données
    print(f"   🎯 Entraînement du modèle final...")
    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(X, y)
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, final_model.feature_importances_))
    # Trier par importance décroissante
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Top 5 features
    top_5_features = dict(list(feature_importance.items())[:5])
    print(f"   📈 Top 5 features:")
    for feat, imp in top_5_features.items():
        print(f"      {feat}: {imp:.4f}")
    
    training_time = time.time() - start_time
    
    return {
        'model': final_model,
        'metrics': avg_metrics,
        'feature_importance': feature_importance,
        'feature_names': feature_cols,
        'training_time': training_time
    }


# ============================================================================
# CHARGEMENT DES DONNÉES CSV
# ============================================================================

def load_csv_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Charge les données depuis les fichiers CSV locaux.
    
    Args:
        tickers: Liste des symboles (ex: ['AAPL', 'MSFT'])
        start_date: Date début
        end_date: Date fin
    
    Returns:
        DataFrame avec colonnes = tickers, index = dates
    """
    import os
    
    data_folder = os.path.join(os.getcwd(), 'data')
    
    price_frames = []
    
    for ticker in tickers:
        csv_path = os.path.join(data_folder, f"{ticker}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
        
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df = df[['Date', 'Close']]
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df = df.set_index('Date')
        df = df.rename(columns={'Close': ticker})
        
        price_frames.append(df)
    
    prices = pd.concat(price_frames, axis=1)
    prices = prices.sort_index()
    prices = prices.ffill().bfill()
    
    return prices


# ============================================================================
# ENDPOINT PRINCIPAL - ENTRAÎNEMENT
# ============================================================================

@router.post("/train", response_model=TrainResponse)
async def train_models(request: TrainRequest):
    """
    Entraîne des modèles ML pour prédire les rendements futurs.
    
    Un modèle est entraîné pour CHAQUE ticker.
    
    Features utilisées:
    - Rendements (1d, 5d, 20d, 60d)
    - Volatilité (5d, 20d, 60d)
    - Moving Averages (20, 50, 200)
    - RSI, MACD, Momentum, Bollinger Bands
    
    Exemple de requête:
    {
        "tickers": ["AAPL", "MSFT"],
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "model_type": "random_forest",
        "target_horizon": 20,
        "n_estimators": 100,
        "max_depth": 10
    }
    """
    try:
        print(f"\n{'='*60}")
        print(f"🤖 Entraînement ML demandé")
        print(f"   Tickers: {request.tickers}")
        print(f"   Période: {request.start_date} à {request.end_date}")
        print(f"   Target horizon: {request.target_horizon} jours")
        print(f"{'='*60}\n")
        
        import time
        total_start_time = time.time()
        
        # Charger les données
        print("📊 Chargement des données...")
        prices_df = load_csv_data(request.tickers, request.start_date, request.end_date)
        print(f"   ✅ {len(prices_df)} jours chargés\n")
        
        # Entraîner un modèle pour chaque ticker
        all_models = {}
        
        for ticker in request.tickers:
            print(f"🎯 Entraînement pour {ticker}...")
            
            # Créer les features pour ce ticker
            prices_series = prices_df[ticker]
            features_df = create_technical_features(prices_series, request.target_horizon)
            
            print(f"   ✅ {len(features_df)} échantillons après feature engineering")
            
            # Entraîner le modèle
            result = train_random_forest_model(
                features_df,
                n_estimators=request.n_estimators,
                max_depth=request.max_depth
            )
            
            # Stocker les résultats
            all_models[ticker] = {
                'model_id': f"model_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'metrics': result['metrics'],
                'feature_importance': result['feature_importance'],
                'training_time': result['training_time']
            }
            
            # TODO: Sauvegarder le modèle dans MongoDB
            # TODO: Logger dans MLflow
            
            print(f"   ✅ Modèle pour {ticker} entraîné en {result['training_time']:.2f}s\n")
        
        total_training_time = time.time() - total_start_time
        
        print(f"{'='*60}")
        print(f"✅ Tous les modèles entraînés!")
        print(f"   Temps total: {total_training_time:.2f}s")
        print(f"{'='*60}\n")
        
        return TrainResponse(
            models=all_models,
            message=f"✅ {len(request.tickers)} modèles entraînés avec succès",
            training_time=total_training_time
        )
    
    except FileNotFoundError as e:
        print(f"❌ Erreur: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """Liste tous les modèles entraînés"""
    # TODO: Récupérer depuis MongoDB
    return {
        "models": [],
        "message": "🤖 Fonction à implémenter (MongoDB)"
    }


@router.get("/predict/{ticker}")
async def predict_returns(ticker: str):
    """
    Prédit les rendements futurs pour un ticker.
    Nécessite d'avoir entraîné un modèle d'abord.
    """
    # TODO: Charger modèle depuis MongoDB et faire prédiction
    return {
        "ticker": ticker,
        "predicted_return": 0.0,
        "confidence": 0.0,
        "message": "🔮 Fonction à implémenter (charger modèle + prédire)"
    }


@router.get("/drift")
async def check_drift(tickers: str = ""):
    """
    Vérifie le drift (dérive) des données.
    Compare la distribution récente vs historique.
    """
    # TODO: Implémenter avec Evidently
    ticker_list = tickers.split(',') if tickers else []
    
    return {
        "tickers": ticker_list,
        "drift_detected": False,
        "drift_score": 0.0,
        "message": "📊 Fonction à implémenter (Evidently)"
    }
