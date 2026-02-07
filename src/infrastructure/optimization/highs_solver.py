"""
HiGHS Solver - Free, fast optimization via scipy
Recommended for learning, prototyping, and production
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import time

from .base_solver import BaseSolver, OptimizationResult


class HiGHSSolver(BaseSolver):
    """
    Portfolio optimizer using HiGHS solver via scipy.
    
    HiGHS is a high-performance open-source optimization solver.
    Included in scipy >= 1.9.0 (FREE, no license needed).
    
    Advantages:
        - 100% FREE and open-source
        - Very fast (comparable to commercial solvers)
        - Production-ready
        - No dependencies beyond scipy
    
    When to use:
        - Learning and prototyping
        - Production (small to medium portfolios)
        - When budget is limited
        - When you want open-source solution
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize HiGHS solver.
        
        Args:
            verbose: Print solver output
        
        Raises:
            ImportError: If scipy not installed or version < 1.9
        """
        super().__init__(verbose)
        
        try:
            from scipy.optimize import minimize
            self.minimize = minimize
        except ImportError as e:
            raise ImportError(
                "HiGHSSolver requires scipy >= 1.9.0\n"
                "Install with: pip install 'scipy>=1.9.0'"
            ) from e
    
    def optimize_markowitz(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        Markowitz Mean-Variance Optimization using HiGHS.
        
        Mathematical Formulation:
        -------------------------
        Maximize: Sharpe Ratio = (w^T μ - r_f) / sqrt(w^T Σ w)
        
        Equivalent to minimizing:
            f(w) = -w^T μ + λ * w^T Σ w
        
        Where:
            w = [w_1, ..., w_n]  portfolio weights (decision variables)
            μ = [μ_1, ..., μ_n]  expected returns vector
            Σ = covariance matrix (n×n)
            r_f = risk-free rate
            λ = risk aversion parameter (controls risk-return tradeoff)
        
        Constraints:
            1. Fully invested: sum(w_i) = 1
            2. Long-only: w_i >= 0  (or min_position_size)
            3. Position limits: w_i <= max_position_size
        
        Args:
            expected_returns: Expected return for each asset
            cov_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate
            constraints: Optional constraints dict
        
        Returns:
            OptimizationResult with optimal portfolio
        """
        start_time = time.time()
        
        n_assets = len(expected_returns)
        tickers = expected_returns.index.tolist()
        
        # Parse constraints
        if constraints is None:
            constraints = {}
        
        max_position = constraints.get('max_position_size', 1.0)
        min_position = constraints.get('min_position_size', 0.0)
        
        # Risk aversion parameter
        # Higher λ = more conservative (lower risk, lower return)
        # Lower λ = more aggressive (higher risk, higher return)
        risk_aversion = 1.0
        
        def objective(weights):
            """
            Objective function to minimize.
            
            f(w) = -E[r_p] + λ * Var(r_p)
                 = -w^T μ + λ * w^T Σ w
            
            This approximates maximizing Sharpe ratio.
            """
            # Expected return: w^T μ
            expected_return = np.dot(weights, expected_returns.values)
            
            # Variance: w^T Σ w
            variance = np.dot(
                weights.T,
                np.dot(cov_matrix.values, weights)
            )
            
            # Objective: maximize return - risk_aversion * variance
            # (We minimize negative of this)
            return -(expected_return - risk_free_rate) + risk_aversion * variance
        
        def objective_gradient(weights):
            """
            Gradient of objective function.
            
            ∇f(w) = -μ + 2λ * Σ w
            
            Using gradient speeds up optimization significantly.
            """
            # Gradient of expected return: μ
            return_grad = expected_returns.values
            
            # Gradient of variance: 2 * Σ w
            variance_grad = 2 * np.dot(cov_matrix.values, weights)
            
            # Combined gradient
            return -return_grad + risk_aversion * variance_grad
        
        # Constraint: sum of weights = 1 (fully invested)
        def constraint_sum(weights):
            """Constraint: sum(w_i) = 1"""
            return np.sum(weights) - 1.0
        
        constraints_list = [
            {
                'type': 'eq',  # Equality constraint
                'fun': constraint_sum
            }
        ]
        
        # Bounds: min_position <= w_i <= max_position
        bounds = [(min_position, max_position) for _ in range(n_assets)]
        
        # Initial guess: equal weight portfolio
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize using SLSQP (Sequential Least Squares Programming)
        result = self.minimize(
            fun=objective,
            x0=initial_weights,
            method='SLSQP',  # Sequential Least Squares Programming
            jac=objective_gradient,  # Provide gradient for speed
            bounds=bounds,
            constraints=constraints_list,
            options={
                'ftol': 1e-9,  # Function tolerance
                'disp': self.verbose  # Display output
            }
        )
        
        solver_time = time.time() - start_time
        
        # Process results
        if result.success:
            # Clean weights (remove very small ones)
            weights = self._clean_weights(result.x)
            
            # Calculate portfolio metrics
            port_return, port_vol, sharpe = self._calculate_metrics(
                weights, expected_returns, cov_matrix, risk_free_rate
            )
            
            # Create weights dictionary
            weights_dict = {
                ticker: weight
                for ticker, weight in zip(tickers, weights)
            }
            
            return OptimizationResult(
                success=True,
                weights=weights_dict,
                expected_return=port_return,
                volatility=port_vol,
                sharpe_ratio=sharpe,
                objective_value=result.fun,
                solver_time=solver_time,
                solver_name="HiGHS (via scipy)",
                message="Optimization successful"
            )
        else:
            # Optimization failed
            return OptimizationResult(
                success=False,
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                solver_time=solver_time,
                solver_name="HiGHS (via scipy)",
                message=f"Optimization failed: {result.message}"
            )
    
    def optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series
    ) -> OptimizationResult:
        """
        Risk Parity Optimization using HiGHS.
        
        Mathematical Formulation:
        -------------------------
        Objective: Equalize risk contribution from each asset
        
        Risk Contribution of asset i:
            RC_i = w_i * (Σw)_i / sqrt(w^T Σ w)
            
        Where:
            (Σw)_i = marginal contribution to risk
            sqrt(w^T Σ w) = portfolio volatility
        
        Goal: RC_1 = RC_2 = ... = RC_n = total_risk / n
        
        Minimize:
            f(w) = sum((RC_i - RC_target)^2)
            
        Constraints:
            1. sum(w_i) = 1
            2. w_i >= min_weight (usually 0.01 to avoid 0 allocation)
            3. w_i <= max_weight (to ensure diversification)
        
        Args:
            cov_matrix: Covariance matrix
            expected_returns: Expected returns (for metrics)
        
        Returns:
            OptimizationResult with risk parity portfolio
        """
        start_time = time.time()
        
        n_assets = len(cov_matrix)
        tickers = cov_matrix.index.tolist()
        
        def calculate_risk_contribution(weights):
            """
            Calculate risk contribution for each asset.
            
            RC_i = w_i * (Σw)_i / σ_p
            
            Where:
                (Σw)_i = marginal contribution to variance
                σ_p = portfolio volatility
            """
            # Portfolio volatility
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix.values, weights))
            )
            
            # Marginal contribution to risk: Σw
            marginal_contrib = np.dot(cov_matrix.values, weights)
            
            # Risk contribution: w_i * (Σw)_i / σ_p
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            return risk_contrib
        
        def objective(weights):
            """
            Minimize variance of risk contributions.
            
            f(w) = sum((RC_i - RC_mean)^2)
            
            This ensures all assets contribute equally to risk.
            """
            rc = calculate_risk_contribution(weights)
            target_rc = np.mean(rc)  # Equal risk contribution
            return np.sum((rc - target_rc) ** 2)
        
        # Constraint: sum(w) = 1
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: reasonable range to ensure diversification
        bounds = [(0.01, 0.5) for _ in range(n_assets)]
        
        # Initial guess: equal weight
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = self.minimize(
            fun=objective,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'disp': self.verbose}
        )
        
        solver_time = time.time() - start_time
        
        if result.success:
            weights = self._clean_weights(result.x)
            
            # Calculate metrics
            port_return, port_vol, sharpe = self._calculate_metrics(
                weights, expected_returns, cov_matrix
            )
            
            weights_dict = {
                ticker: weight
                for ticker, weight in zip(tickers, weights)
            }
            
            return OptimizationResult(
                success=True,
                weights=weights_dict,
                expected_return=port_return,
                volatility=port_vol,
                sharpe_ratio=sharpe,
                objective_value=result.fun,
                solver_time=solver_time,
                solver_name="HiGHS (via scipy)",
                message="Risk Parity optimization successful"
            )
        else:
            return OptimizationResult(
                success=False,
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                solver_time=solver_time,
                solver_name="HiGHS (via scipy)",
                message=f"Optimization failed: {result.message}"
            )


if __name__ == "__main__":
    # Example usage
    print("HiGHSSolver: Free, fast portfolio optimization")
    print("Requires: scipy >= 1.9.0")
    print("\nUsage:")
    print("  from highs_solver import HiGHSSolver")
    print("  solver = HiGHSSolver()")
    print("  result = solver.optimize_markowitz(returns, cov_matrix)")
