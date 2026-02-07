"""
CVXPY Solver - Flexible convex optimization
Good alternative to HiGHS, more expressive syntax
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import time

from .base_solver import BaseSolver, OptimizationResult


class CVXPYSolver(BaseSolver):
    """
    Portfolio optimizer using CVXPY.
    
    CVXPY is a Python library for convex optimization.
    Can use different backends (ECOS, OSQP, SCS, etc.).
    
    Advantages:
        - FREE and open-source
        - Very expressive syntax
        - Supports many constraint types
        - Can switch backends easily
    
    When to use:
        - Need flexible constraint specification
        - Want readable optimization code
        - Prototyping complex strategies
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize CVXPY solver.
        
        Args:
            verbose: Print solver output
        
        Raises:
            ImportError: If cvxpy not installed
        """
        super().__init__(verbose)
        
        try:
            import cvxpy as cp
            self.cp = cp
        except ImportError as e:
            raise ImportError(
                "CVXPYSolver requires cvxpy\n"
                "Install with: pip install cvxpy"
            ) from e
    
    def optimize_markowitz(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        Markowitz optimization using CVXPY.
        
        Mathematical Formulation:
        -------------------------
        Decision Variable:
            w ∈ R^n  (portfolio weights)
        
        Objective:
            Maximize: w^T μ - λ/2 * w^T Σ w
            
        Which approximates:
            Maximize: Sharpe Ratio = (w^T μ - r_f) / sqrt(w^T Σ w)
        
        Constraints:
            - sum(w_i) = 1  (fully invested)
            - w_min <= w_i <= w_max  (position limits)
        
        CVXPY Syntax:
            >>> w = cp.Variable(n)
            >>> objective = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, Sigma))
            >>> constraints = [cp.sum(w) == 1, w >= 0, w <= max_w]
            >>> problem = cp.Problem(objective, constraints)
            >>> problem.solve()
        
        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            constraints: Optional constraints
        
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        n_assets = len(expected_returns)
        tickers = expected_returns.index.tolist()
        
        # Parse constraints
        if constraints is None:
            constraints = {}
        
        max_position = constraints.get('max_position_size', 1.0)
        min_position = constraints.get('min_position_size', 0.0)
        
        # Decision variable: portfolio weights
        weights = self.cp.Variable(n_assets)
        
        # Objective: Maximize return - 0.5 * variance
        # This is a quadratic approximation to maximizing Sharpe ratio
        portfolio_return = expected_returns.values @ weights
        portfolio_variance = self.cp.quad_form(weights, cov_matrix.values)
        
        objective = self.cp.Maximize(
            portfolio_return - 0.5 * portfolio_variance
        )
        
        # Constraints
        constraints_list = [
            self.cp.sum(weights) == 1,        # Fully invested
            weights >= min_position,           # Lower bound
            weights <= max_position            # Upper bound
        ]
        
        # Create and solve problem
        problem = self.cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(verbose=self.verbose)
        except Exception as e:
            return OptimizationResult(
                success=False,
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                solver_time=time.time() - start_time,
                solver_name="CVXPY",
                message=f"Solver error: {str(e)}"
            )
        
        solver_time = time.time() - start_time
        
        # Check if solution found
        if weights.value is not None and problem.status == 'optimal':
            w = self._clean_weights(weights.value)
            
            # Calculate metrics
            port_return, port_vol, sharpe = self._calculate_metrics(
                w, expected_returns, cov_matrix, risk_free_rate
            )
            
            weights_dict = {
                ticker: weight
                for ticker, weight in zip(tickers, w)
            }
            
            return OptimizationResult(
                success=True,
                weights=weights_dict,
                expected_return=port_return,
                volatility=port_vol,
                sharpe_ratio=sharpe,
                objective_value=problem.value,
                solver_time=solver_time,
                solver_name="CVXPY",
                message=f"Optimization successful (status: {problem.status})"
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
                solver_name="CVXPY",
                message=f"Optimization failed (status: {problem.status})"
            )
    
    def optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series
    ) -> OptimizationResult:
        """
        Risk Parity using CVXPY.
        
        Note: Risk Parity is non-convex, so we use a convex approximation.
        
        Approximation:
            Minimize: sum(log(w_i)) subject to w^T Σ w <= target_variance
        
        This tends to equalize weights in log-space, approximating risk parity.
        
        Args:
            cov_matrix: Covariance matrix
            expected_returns: Expected returns
        
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        n_assets = len(cov_matrix)
        tickers = cov_matrix.index.tolist()
        
        # Decision variable
        weights = self.cp.Variable(n_assets)
        
        # Target variance (chosen empirically)
        target_variance = 0.01  # 10% annual volatility
        
        # Objective: Encourage equal weights (in log space)
        # This is a convex approximation to risk parity
        objective = self.cp.Minimize(
            -self.cp.sum(self.cp.log(weights))
        )
        
        # Constraints
        constraints_list = [
            self.cp.sum(weights) == 1,
            weights >= 0.01,  # Minimum weight
            weights <= 0.5,   # Maximum weight
            self.cp.quad_form(weights, cov_matrix.values) <= target_variance
        ]
        
        # Solve
        problem = self.cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(verbose=self.verbose)
        except Exception as e:
            return OptimizationResult(
                success=False,
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                solver_time=time.time() - start_time,
                solver_name="CVXPY",
                message=f"Solver error: {str(e)}"
            )
        
        solver_time = time.time() - start_time
        
        if weights.value is not None and problem.status == 'optimal':
            w = self._clean_weights(weights.value)
            
            port_return, port_vol, sharpe = self._calculate_metrics(
                w, expected_returns, cov_matrix
            )
            
            weights_dict = {
                ticker: weight
                for ticker, weight in zip(tickers, w)
            }
            
            return OptimizationResult(
                success=True,
                weights=weights_dict,
                expected_return=port_return,
                volatility=port_vol,
                sharpe_ratio=sharpe,
                objective_value=problem.value,
                solver_time=solver_time,
                solver_name="CVXPY",
                message="Risk Parity (convex approximation) successful"
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
                solver_name="CVXPY",
                message=f"Optimization failed (status: {problem.status})"
            )


if __name__ == "__main__":
    print("CVXPYSolver: Flexible convex optimization")
    print("Requires: cvxpy")
    print("\nUsage:")
    print("  from cvxpy_solver import CVXPYSolver")
    print("  solver = CVXPYSolver()")
    print("  result = solver.optimize_markowitz(returns, cov_matrix)")
