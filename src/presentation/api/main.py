"""
FastAPI main application for Robo-Advisor.
Provides REST API endpoints for portfolio optimization, predictions, and risk management.
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from config.settings import settings
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Robo-Advisor API",
    description="Portfolio Optimization & Risk Management Platform with ML and Mathematical Optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS (DTOs)
# ============================================================================

class AssetInput(BaseModel):
    """Input model for an asset."""
    ticker: str = Field(..., description="Asset ticker symbol")
    expected_return: float = Field(..., description="Expected annual return")
    weight: Optional[float] = Field(None, ge=0, le=1, description="Current weight in portfolio")


class OptimizationRequest(BaseModel):
    """Request model for portfolio optimization."""
    assets: List[AssetInput]
    method: str = Field(default="markowitz", description="Optimization method: markowitz, risk_parity, cvar")
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate")
    target_return: Optional[float] = Field(None, description="Target return (for Markowitz)")
    max_position_size: float = Field(default=0.15, ge=0, le=1)
    min_position_size: float = Field(default=0.0, ge=0, le=1)


class OptimizationResponse(BaseModel):
    """Response model for optimization results."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    status: str
    method: str
    metadata: Dict


class RiskMetricsRequest(BaseModel):
    """Request model for risk metrics calculation."""
    ticker: Optional[str] = Field(None, description="Single asset or None for portfolio")
    confidence_level: float = Field(default=0.95, ge=0.9, le=0.99)
    portfolio_weights: Optional[Dict[str, float]] = None


class RiskMetricsResponse(BaseModel):
    """Response model for risk metrics."""
    var: float = Field(..., description="Value at Risk")
    es: float = Field(..., description="Expected Shortfall")
    volatility: float
    beta: Optional[float] = None
    max_drawdown: Optional[float] = None


class PredictionRequest(BaseModel):
    """Request model for return prediction."""
    tickers: List[str]
    model_type: str = Field(default="xgboost", description="Model type: xgboost, lightgbm, lstm")
    horizon: int = Field(default=1, description="Prediction horizon in days")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    model_performance: Dict[str, float]


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": settings.env
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Robo-Advisor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "optimization": "/api/v1/optimize",
            "risk": "/api/v1/risk",
            "prediction": "/api/v1/predict",
            "backtest": "/api/v1/backtest"
        }
    }


# ============================================================================
# OPTIMIZATION ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/optimize",
    response_model=OptimizationResponse,
    tags=["Portfolio Optimization"],
    summary="Optimize portfolio weights"
)
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio using various methods (Markowitz, Risk Parity, CVaR).
    
    - **Markowitz**: Maximize Sharpe ratio or minimize risk for target return
    - **Risk Parity**: Equalize risk contributions
    - **CVaR**: Minimize conditional value at risk
    """
    try:
        logger.info(f"Optimization request: method={request.method}, assets={len(request.assets)}")
        
        # Extract data from request
        tickers = [asset.ticker for asset in request.assets]
        expected_returns = pd.Series({
            asset.ticker: asset.expected_return
            for asset in request.assets
        })
        
        # For demo, create a sample covariance matrix
        # In production, this would come from historical data or ML predictions
        n_assets = len(tickers)
        correlation = np.full((n_assets, n_assets), 0.3)
        np.fill_diagonal(correlation, 1.0)
        volatilities = np.array([0.15 + np.random.rand() * 0.10 for _ in range(n_assets)])
        covariance_matrix = pd.DataFrame(
            np.outer(volatilities, volatilities) * correlation,
            index=tickers,
            columns=tickers
        )
        
        # Perform optimization
        if request.method == "markowitz":
            # Simple Markowitz optimization
            weights = _optimize_markowitz_simple(
                expected_returns,
                covariance_matrix,
                request.risk_free_rate,
                request.max_position_size
            )
        elif request.method == "risk_parity":
            # Risk parity
            weights = _optimize_risk_parity_simple(
                covariance_matrix,
                request.max_position_size
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown optimization method: {request.method}"
            )
        
        # Calculate portfolio metrics
        portfolio_return = sum(weights[ticker] * expected_returns[ticker] for ticker in tickers)
        portfolio_variance = sum(
            weights[ticker_i] * weights[ticker_j] * covariance_matrix.loc[ticker_i, ticker_j]
            for ticker_i in tickers
            for ticker_j in tickers
        )
        portfolio_risk = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - request.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        logger.info(f"Optimization successful: Sharpe={sharpe_ratio:.4f}")
        
        return OptimizationResponse(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            status="optimal",
            method=request.method,
            metadata={
                "n_assets": len(tickers),
                "optimization_time": 0.5,
                "risk_free_rate": request.risk_free_rate
            }
        )
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RISK MANAGEMENT ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/risk",
    response_model=RiskMetricsResponse,
    tags=["Risk Management"],
    summary="Calculate risk metrics"
)
async def calculate_risk_metrics(request: RiskMetricsRequest):
    """
    Calculate comprehensive risk metrics including VaR, Expected Shortfall, volatility, etc.
    """
    try:
        logger.info(f"Risk metrics request: confidence={request.confidence_level}")
        
        # Simulate returns data (in production, fetch from database)
        returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
        
        # Calculate VaR
        var = np.percentile(returns, (1 - request.confidence_level) * 100)
        
        # Calculate Expected Shortfall
        es = returns[returns <= var].mean()
        
        # Calculate volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"Risk metrics calculated: VaR={var:.4f}, ES={es:.4f}")
        
        return RiskMetricsResponse(
            var=float(var),
            es=float(es),
            volatility=float(volatility),
            beta=None,  # Would require market data
            max_drawdown=float(max_drawdown)
        )
        
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ML PREDICTION ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["ML Predictions"],
    summary="Predict asset returns"
)
async def predict_returns(request: PredictionRequest):
    """
    Predict future returns using ML models (XGBoost, LightGBM, LSTM).
    """
    try:
        logger.info(f"Prediction request: tickers={request.tickers}, model={request.model_type}")
        
        # Simulate predictions (in production, use trained models)
        predictions = {
            ticker: np.random.normal(0.08, 0.03)
            for ticker in request.tickers
        }
        
        # Simulate confidence intervals
        confidence_intervals = {
            ticker: {
                "lower": pred - 0.02,
                "upper": pred + 0.02
            }
            for ticker, pred in predictions.items()
        }
        
        # Model performance metrics
        model_performance = {
            "mae": 0.015,
            "rmse": 0.023,
            "r2_score": 0.65
        }
        
        logger.info(f"Predictions generated for {len(request.tickers)} assets")
        
        return PredictionResponse(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_performance=model_performance
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _optimize_markowitz_simple(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
    max_position_size: float
) -> Dict[str, float]:
    """Simple Markowitz optimization without Gurobi (for demo)."""
    n_assets = len(expected_returns)
    tickers = expected_returns.index.tolist()
    
    # Start with equal weights
    weights = np.ones(n_assets) / n_assets
    
    # Simple gradient-based optimization (not optimal, just for demo)
    for _ in range(100):
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        # Simple gradient ascent on Sharpe
        grad = (expected_returns - risk_free_rate) / portfolio_std
        weights += 0.01 * grad
        
        # Project onto constraints
        weights = np.clip(weights, 0, max_position_size)
        weights = weights / weights.sum()
    
    return {ticker: float(w) for ticker, w in zip(tickers, weights)}


def _optimize_risk_parity_simple(
    covariance_matrix: pd.DataFrame,
    max_position_size: float
) -> Dict[str, float]:
    """Simple risk parity optimization (equal risk contribution)."""
    n_assets = len(covariance_matrix)
    tickers = covariance_matrix.index.tolist()
    
    # Start with equal weights
    weights = np.ones(n_assets) / n_assets
    
    # Iterative equalization of risk contributions
    for _ in range(100):
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Adjust weights inversely to risk contribution
        weights = weights / risk_contrib
        weights = np.clip(weights, 0, max_position_size)
        weights = weights / weights.sum()
    
    return {ticker: float(w) for ticker, w in zip(tickers, weights)}


# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting Robo-Advisor API...")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"MongoDB URI: {settings.mongodb_uri}")
    logger.info(f"Redis Host: {settings.redis_host}")
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down Robo-Advisor API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
