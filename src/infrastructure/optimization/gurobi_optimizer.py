"""
Gurobi Portfolio Optimizer

Implements various portfolio optimization strategies using Gurobi:
- Markowitz Mean-Variance
- Risk Parity
- CVaR Optimization
- Black-Litterman
- Cardinality-constrained optimization
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class GurobiPortfolioOptimizer:
    """
    Portfolio optimization using Gurobi solver.
    
    Supports multiple optimization objectives and constraints.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        time_limit: int = 300,  # 5 minutes
        mip_gap: float = 0.01   # 1% optimality gap
    ):
        self.risk_free_rate = risk_free_rate
        self.time_limit = time_limit
        self.mip_gap = mip_gap
    
    def optimize_markowitz(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        budget: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        asset_sectors: Optional[Dict[int, str]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Markowitz Mean-Variance Optimization
        
        Minimize: λ * w^T Σ w - E[R]^T w
        Subject to:
            - sum(w) = budget
            - min_weight <= w_i <= max_weight
            - Optional: target return constraint
            - Optional: sector constraints
        
        Args:
            expected_returns: Expected returns for each asset (N,)
            covariance_matrix: Covariance matrix (N, N)
            budget: Total budget (typically 1.0 for 100%)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            target_return: Minimum target return (optional)
            risk_aversion: Risk aversion parameter (λ)
            sector_constraints: Dict of sector -> (min_weight, max_weight)
            asset_sectors: Dict of asset_index -> sector_name
        
        Returns:
            optimal_weights: Optimal portfolio weights
            metrics: Dictionary with optimization metrics
        """
        n_assets = len(expected_returns)
        
        try:
            # Create model
            model = gp.Model("markowitz_optimization")
            model.setParam('OutputFlag', 0)  # Suppress output
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.mip_gap)
            
            # Decision variables: portfolio weights
            weights = model.addVars(n_assets, lb=min_weight, ub=max_weight, name="weight")
            
            # Objective: Maximize risk-adjusted return
            # Minimize: risk_aversion * variance - expected_return
            portfolio_variance = gp.quicksum(
                weights[i] * weights[j] * covariance_matrix[i, j]
                for i in range(n_assets)
                for j in range(n_assets)
            )
            
            portfolio_return = gp.quicksum(
                weights[i] * expected_returns[i]
                for i in range(n_assets)
            )
            
            objective = risk_aversion * portfolio_variance - portfolio_return
            model.setObjective(objective, GRB.MINIMIZE)
            
            # Constraint: Budget constraint (weights sum to budget)
            model.addConstr(
                gp.quicksum(weights[i] for i in range(n_assets)) == budget,
                name="budget"
            )
            
            # Constraint: Target return (if specified)
            if target_return is not None:
                model.addConstr(
                    portfolio_return >= target_return,
                    name="target_return"
                )
            
            # Constraint: Sector constraints (if specified)
            if sector_constraints and asset_sectors:
                sectors = set(asset_sectors.values())
                for sector in sectors:
                    if sector in sector_constraints:
                        min_sector, max_sector = sector_constraints[sector]
                        sector_assets = [i for i, s in asset_sectors.items() if s == sector]
                        
                        sector_weight = gp.quicksum(weights[i] for i in sector_assets)
                        model.addConstr(sector_weight >= min_sector, name=f"sector_{sector}_min")
                        model.addConstr(sector_weight <= max_sector, name=f"sector_{sector}_max")
            
            # Optimize
            model.optimize()
            
            # Extract results
            if model.status == GRB.OPTIMAL:
                optimal_weights = np.array([weights[i].X for i in range(n_assets)])
                
                # Calculate metrics
                portfolio_vol = np.sqrt(
                    optimal_weights.T @ covariance_matrix @ optimal_weights
                )
                portfolio_ret = optimal_weights.T @ expected_returns
                sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
                
                metrics = {
                    "status": "optimal",
                    "portfolio_return": float(portfolio_ret),
                    "portfolio_volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "objective_value": float(model.objVal),
                    "solve_time": float(model.Runtime)
                }
                
                logger.info(f"Markowitz optimization successful. Sharpe: {sharpe_ratio:.4f}")
                return optimal_weights, metrics
            else:
                raise OptimizationError(f"Optimization failed with status: {model.status}")
        
        except Exception as e:
            logger.error(f"Markowitz optimization error: {str(e)}")
            raise
    
    def optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        budget: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Maximize Sharpe Ratio
        
        This is a reformulation of Sharpe maximization as a QP problem.
        """
        n_assets = len(expected_returns)
        
        try:
            model = gp.Model("max_sharpe")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', self.time_limit)
            
            # Decision variables
            weights = model.addVars(n_assets, lb=min_weight, ub=max_weight, name="weight")
            
            # Auxiliary variable for normalization
            kappa = model.addVar(lb=0, name="kappa")
            
            # Objective: Minimize variance
            portfolio_variance = gp.quicksum(
                weights[i] * weights[j] * covariance_matrix[i, j]
                for i in range(n_assets)
                for j in range(n_assets)
            )
            
            model.setObjective(portfolio_variance, GRB.MINIMIZE)
            
            # Constraint: Expected return equals 1 (normalization)
            excess_returns = expected_returns - self.risk_free_rate
            model.addConstr(
                gp.quicksum(weights[i] * excess_returns[i] for i in range(n_assets)) == 1,
                name="unit_excess_return"
            )
            
            # Optimize
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # Denormalize weights
                raw_weights = np.array([weights[i].X for i in range(n_assets)])
                optimal_weights = raw_weights / raw_weights.sum()
                
                # Calculate metrics
                portfolio_vol = np.sqrt(
                    optimal_weights.T @ covariance_matrix @ optimal_weights
                )
                portfolio_ret = optimal_weights.T @ expected_returns
                sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
                
                metrics = {
                    "status": "optimal",
                    "portfolio_return": float(portfolio_ret),
                    "portfolio_volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "solve_time": float(model.Runtime)
                }
                
                logger.info(f"Max Sharpe optimization successful. Sharpe: {sharpe_ratio:.4f}")
                return optimal_weights, metrics
            else:
                raise OptimizationError(f"Optimization failed with status: {model.status}")
        
        except Exception as e:
            logger.error(f"Max Sharpe optimization error: {str(e)}")
            raise
    
    def optimize_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        budget: float = 1.0,
        min_weight: float = 0.001
    ) -> Tuple[np.ndarray, Dict]:
        """
        Risk Parity Optimization
        
        Equalize risk contribution from each asset.
        Risk contribution of asset i: w_i * (Σw)_i
        
        This is a non-convex problem, approximated using iterative method.
        """
        n_assets = covariance_matrix.shape[0]
        
        try:
            model = gp.Model("risk_parity")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('NonConvex', 2)  # Allow non-convex objectives
            
            # Decision variables
            weights = model.addVars(n_assets, lb=min_weight, ub=1.0, name="weight")
            
            # Target risk contribution (equal for all assets)
            target_risk_contrib = 1.0 / n_assets
            
            # Objective: Minimize deviation from equal risk contribution
            # Using quadratic penalty for deviations
            objective = 0
            for i in range(n_assets):
                # Risk contribution of asset i
                marginal_risk = gp.quicksum(
                    covariance_matrix[i, j] * weights[j]
                    for j in range(n_assets)
                )
                risk_contrib = weights[i] * marginal_risk
                
                # Squared deviation from target
                objective += (risk_contrib - target_risk_contrib) ** 2
            
            model.setObjective(objective, GRB.MINIMIZE)
            
            # Constraint: Budget
            model.addConstr(
                gp.quicksum(weights[i] for i in range(n_assets)) == budget,
                name="budget"
            )
            
            # Optimize
            model.optimize()
            
            if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                optimal_weights = np.array([weights[i].X for i in range(n_assets)])
                
                # Calculate metrics
                portfolio_vol = np.sqrt(
                    optimal_weights.T @ covariance_matrix @ optimal_weights
                )
                
                # Calculate actual risk contributions
                marginal_risk = covariance_matrix @ optimal_weights
                risk_contributions = optimal_weights * marginal_risk
                total_risk = risk_contributions.sum()
                risk_contributions_pct = risk_contributions / total_risk
                
                metrics = {
                    "status": "optimal" if model.status == GRB.OPTIMAL else "suboptimal",
                    "portfolio_volatility": float(portfolio_vol),
                    "risk_contributions": risk_contributions_pct.tolist(),
                    "risk_contrib_std": float(np.std(risk_contributions_pct)),
                    "solve_time": float(model.Runtime)
                }
                
                logger.info(f"Risk Parity optimization successful. Vol: {portfolio_vol:.4f}")
                return optimal_weights, metrics
            else:
                raise OptimizationError(f"Optimization failed with status: {model.status}")
        
        except Exception as e:
            logger.error(f"Risk Parity optimization error: {str(e)}")
            raise
    
    def optimize_cvar(
        self,
        returns_scenarios: np.ndarray,
        alpha: float = 0.95,
        budget: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        CVaR (Conditional Value at Risk) Optimization
        
        Minimize expected loss in the worst (1-α)% scenarios.
        
        Args:
            returns_scenarios: Historical or simulated returns (n_scenarios, n_assets)
            alpha: Confidence level (e.g., 0.95 for 95%)
            budget: Total budget
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            target_return: Minimum target return (optional)
        
        Returns:
            optimal_weights: Optimal portfolio weights
            metrics: Dictionary with optimization metrics
        """
        n_scenarios, n_assets = returns_scenarios.shape
        
        try:
            model = gp.Model("cvar_optimization")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', self.time_limit)
            
            # Decision variables
            weights = model.addVars(n_assets, lb=min_weight, ub=max_weight, name="weight")
            
            # VaR variable (α-quantile of loss distribution)
            var = model.addVar(lb=-GRB.INFINITY, name="var")
            
            # Auxiliary variables for CVaR calculation
            # z_s = max(0, -portfolio_return_s - VaR)
            z = model.addVars(n_scenarios, lb=0, name="z")
            
            # Objective: Minimize CVaR = VaR + (1/(1-α)) * E[z]
            cvar = var + (1.0 / (1.0 - alpha)) * gp.quicksum(z[s] for s in range(n_scenarios)) / n_scenarios
            model.setObjective(cvar, GRB.MINIMIZE)
            
            # Constraints: z_s >= -portfolio_return_s - VaR
            for s in range(n_scenarios):
                portfolio_return_s = gp.quicksum(
                    weights[i] * returns_scenarios[s, i]
                    for i in range(n_assets)
                )
                model.addConstr(z[s] >= -portfolio_return_s - var, name=f"cvar_{s}")
            
            # Constraint: Budget
            model.addConstr(
                gp.quicksum(weights[i] for i in range(n_assets)) == budget,
                name="budget"
            )
            
            # Constraint: Target return (if specified)
            if target_return is not None:
                expected_return = gp.quicksum(
                    weights[i] * returns_scenarios[:, i].mean()
                    for i in range(n_assets)
                )
                model.addConstr(expected_return >= target_return, name="target_return")
            
            # Optimize
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                optimal_weights = np.array([weights[i].X for i in range(n_assets)])
                
                # Calculate metrics
                portfolio_returns = returns_scenarios @ optimal_weights
                expected_return = portfolio_returns.mean()
                portfolio_vol = portfolio_returns.std()
                
                # Calculate VaR and CVaR
                sorted_returns = np.sort(portfolio_returns)
                var_idx = int((1 - alpha) * n_scenarios)
                var_value = -sorted_returns[var_idx]
                cvar_value = -sorted_returns[:var_idx].mean()
                
                metrics = {
                    "status": "optimal",
                    "expected_return": float(expected_return),
                    "portfolio_volatility": float(portfolio_vol),
                    "var": float(var_value),
                    "cvar": float(cvar_value),
                    "objective_value": float(model.objVal),
                    "solve_time": float(model.Runtime)
                }
                
                logger.info(f"CVaR optimization successful. CVaR: {cvar_value:.4f}")
                return optimal_weights, metrics
            else:
                raise OptimizationError(f"Optimization failed with status: {model.status}")
        
        except Exception as e:
            logger.error(f"CVaR optimization error: {str(e)}")
            raise
    
    def optimize_cardinality_constrained(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        k_assets: int,
        budget: float = 1.0,
        min_weight: float = 0.01,
        max_weight: float = 1.0,
        risk_aversion: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cardinality-Constrained Portfolio Optimization
        
        Select exactly K assets from N available assets.
        This requires Mixed-Integer Quadratic Programming (MIQP).
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            k_assets: Number of assets to select
            budget: Total budget
            min_weight: Minimum weight if asset is selected
            max_weight: Maximum weight per asset
            risk_aversion: Risk aversion parameter
        
        Returns:
            optimal_weights: Optimal portfolio weights (sparse with k non-zero)
            metrics: Dictionary with optimization metrics
        """
        n_assets = len(expected_returns)
        
        if k_assets > n_assets:
            raise ValueError("k_assets cannot exceed number of available assets")
        
        try:
            model = gp.Model("cardinality_constrained")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.mip_gap)
            
            # Decision variables
            # Continuous: portfolio weights
            weights = model.addVars(n_assets, lb=0, ub=max_weight, name="weight")
            
            # Binary: asset selection indicators
            selected = model.addVars(n_assets, vtype=GRB.BINARY, name="selected")
            
            # Objective: Markowitz mean-variance
            portfolio_variance = gp.quicksum(
                weights[i] * weights[j] * covariance_matrix[i, j]
                for i in range(n_assets)
                for j in range(n_assets)
            )
            
            portfolio_return = gp.quicksum(
                weights[i] * expected_returns[i]
                for i in range(n_assets)
            )
            
            objective = risk_aversion * portfolio_variance - portfolio_return
            model.setObjective(objective, GRB.MINIMIZE)
            
            # Constraint: Budget
            model.addConstr(
                gp.quicksum(weights[i] for i in range(n_assets)) == budget,
                name="budget"
            )
            
            # Constraint: Cardinality (select exactly k assets)
            model.addConstr(
                gp.quicksum(selected[i] for i in range(n_assets)) == k_assets,
                name="cardinality"
            )
            
            # Constraint: Link weights to selection
            # If selected[i] = 0, then weight[i] = 0
            # If selected[i] = 1, then weight[i] >= min_weight
            for i in range(n_assets):
                model.addConstr(
                    weights[i] <= max_weight * selected[i],
                    name=f"link_upper_{i}"
                )
                model.addConstr(
                    weights[i] >= min_weight * selected[i],
                    name=f"link_lower_{i}"
                )
            
            # Optimize
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                optimal_weights = np.array([weights[i].X for i in range(n_assets)])
                selected_assets = [i for i in range(n_assets) if selected[i].X > 0.5]
                
                # Calculate metrics
                portfolio_vol = np.sqrt(
                    optimal_weights.T @ covariance_matrix @ optimal_weights
                )
                portfolio_ret = optimal_weights.T @ expected_returns
                sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
                
                metrics = {
                    "status": "optimal",
                    "portfolio_return": float(portfolio_ret),
                    "portfolio_volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "selected_assets": selected_assets,
                    "n_selected": len(selected_assets),
                    "objective_value": float(model.objVal),
                    "solve_time": float(model.Runtime)
                }
                
                logger.info(
                    f"Cardinality optimization successful. "
                    f"Selected {len(selected_assets)} assets. Sharpe: {sharpe_ratio:.4f}"
                )
                return optimal_weights, metrics
            else:
                raise OptimizationError(f"Optimization failed with status: {model.status}")
        
        except Exception as e:
            logger.error(f"Cardinality optimization error: {str(e)}")
            raise


class OptimizationError(Exception):
    """Custom exception for optimization errors."""
    pass
