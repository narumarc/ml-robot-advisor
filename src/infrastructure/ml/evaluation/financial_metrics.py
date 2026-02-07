"""Financial evaluation metrics - Sharpe, Sortino, Information Ratio, etc."""
import numpy as np
import pandas as pd
from typing import Dict, Optional

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sharpe Ratio."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() == 0:
        return 0.0
    sharpe = (excess_returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
    return float(sharpe)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sortino Ratio (uses downside deviation)."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    sortino = (excess_returns.mean() * periods_per_year) / (downside_returns.std() * np.sqrt(periods_per_year))
    return float(sortino)

def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate Information Ratio."""
    active_returns = portfolio_returns - benchmark_returns
    if active_returns.std() == 0:
        return 0.0
    ir = active_returns.mean() / active_returns.std() * np.sqrt(252)
    return float(ir)

def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar Ratio (return / max drawdown)."""
    annual_return = returns.mean() * periods_per_year
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    if max_drawdown == 0:
        return 0.0
    return float(annual_return / max_drawdown)

def calculate_all_financial_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate all financial metrics."""
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'annual_return': float(returns.mean() * 252),
        'annual_volatility': float(returns.std() * np.sqrt(252))
    }
    
    if benchmark_returns is not None:
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns)
    
    return metrics
