"""
Base Solver - Abstract interface for portfolio optimization
All solvers must implement this interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    """
    Standard result format for all solvers.
    
    Attributes:
        success: Whether optimization succeeded
        weights: Portfolio weights {ticker: weight}
        expected_return: Expected portfolio return (annualized)
        volatility: Portfolio volatility (annualized)
        sharpe_ratio: Sharpe ratio = (return - rf) / volatility
        objective_value: Value of objective function at optimum
        solver_time: Time taken by solver (seconds)
        solver_name: Name of solver used
        message: Success or error message
        cvar: Conditional Value at Risk (if applicable)
    """
    success: bool
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective_value: float
    solver_time: float
    solver_name: str
    message: str = ""
    cvar: Optional[float] = None


class BaseSolver(ABC):
    """
    Abstract base class for portfolio optimization solvers.
    
    All concrete solvers (HiGHS, CVXPY, Gurobi, etc.) must inherit from this
    and implement the abstract methods.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize solver.
        
        Args:
            verbose: If True, print solver output
        """
        self.verbose = verbose
    
    @abstractmethod
    def optimize_markowitz(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        Markowitz Mean-Variance Optimization.
        
        Objective: Maximize Sharpe Ratio
        Math: max (w^T μ - r_f) / sqrt(w^T Σ w)
        
        Where:
            w = portfolio weights (decision variable)
            μ = expected returns vector
            Σ = covariance matrix
            r_f = risk-free rate
        
        Args:
            expected_returns: Expected return for each asset (annualized)
            cov_matrix: Covariance matrix of returns (annualized)
            risk_free_rate: Risk-free rate for Sharpe calculation
            constraints: Optional constraints dictionary:
                - max_position_size: Maximum weight per asset (default 1.0)
                - min_position_size: Minimum weight per asset (default 0.0)
        
        Returns:
            OptimizationResult with optimal portfolio weights and metrics
        """
        pass
    
    @abstractmethod
    def optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series
    ) -> OptimizationResult:
        """
        Risk Parity Optimization.
        
        Objective: Equalize risk contribution from each asset
        Math: min sum((RC_i - RC_target)^2)
        
        Where:
            RC_i = risk contribution of asset i
            RC_i = w_i * (Σw)_i / sqrt(w^T Σ w)
            RC_target = total_risk / n_assets
        
        Args:
            cov_matrix: Covariance matrix of returns
            expected_returns: Expected returns (for metrics calculation)
        
        Returns:
            OptimizationResult with risk parity portfolio
        """
        pass
    
    def optimize_cvar(
        self,
        returns_scenarios: pd.DataFrame,
        alpha: float = 0.95,
        target_return: Optional[float] = None
    ) -> OptimizationResult:
        """
        CVaR (Conditional Value at Risk) Optimization.
        
        Objective: Minimize CVaR (expected loss in worst α% cases)
        Math: min CVaR_α(w) = min E[Loss | Loss >= VaR_α]
        
        Where:
            VaR_α = Value at Risk at confidence level α
            CVaR_α = Conditional VaR (average loss beyond VaR)
        
        Args:
            returns_scenarios: Matrix of return scenarios
            alpha: Confidence level (e.g., 0.95 for 95%)
            target_return: Minimum required return (optional)
        
        Returns:
            OptimizationResult with minimum CVaR portfolio
        
        Note:
            Not all solvers implement this. May raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement CVaR optimization"
        )
    
    def _calculate_metrics(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> tuple:
        """
        Calculate portfolio performance metrics.
        
        Math:
            Portfolio return: r_p = w^T μ
            Portfolio variance: σ_p^2 = w^T Σ w
            Portfolio volatility: σ_p = sqrt(σ_p^2)
            Sharpe ratio: SR = (r_p - r_f) / σ_p
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
        
        Returns:
            Tuple of (portfolio_return, portfolio_volatility, sharpe_ratio)
        """
        # Portfolio return: w^T μ
        portfolio_return = np.dot(weights, expected_returns.values)
        
        # Portfolio variance: w^T Σ w
        portfolio_variance = np.dot(
            weights.T,
            np.dot(cov_matrix.values, weights)
        )
        
        # Portfolio volatility: sqrt(variance)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio: (return - rf) / volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def _clean_weights(
        self,
        weights: np.ndarray,
        threshold: float = 1e-6
    ) -> np.ndarray:
        """
        Clean small weights and renormalize.
        
        Args:
            weights: Raw weights from optimization
            threshold: Weights below this are set to 0
        
        Returns:
            Cleaned and normalized weights
        """
        # Set tiny weights to 0
        weights[weights < threshold] = 0
        
        # Renormalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights


if __name__ == "__main__":
    # This is an abstract class, cannot be instantiated directly
    print("BaseSolver: Abstract base class for portfolio optimizers")
    print("Concrete implementations: HiGHSSolver, CVXPYSolver, GurobiSolver")
