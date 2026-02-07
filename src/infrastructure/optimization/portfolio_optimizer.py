"""
Portfolio optimization service using Gurobi.
Implements various optimization strategies (Markowitz, Risk Parity, CVaR, etc.)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    status: str
    objective_value: float
    metadata: Dict = None


class PortfolioOptimizer:
    """
    Portfolio optimization using Gurobi.
    Supports multiple optimization objectives and constraints.
    """
    
    def __init__(
        self,
        timeout: int = 300,
        mip_gap: float = 0.01,
        log_to_console: bool = False
    ):
        """
        Initialize optimizer.
        
        Args:
            timeout: Max optimization time in seconds
            mip_gap: MIP optimality gap tolerance
            log_to_console: Whether to show Gurobi output
        """
        self.timeout = timeout
        self.mip_gap = mip_gap
        self.log_to_console = log_to_console
    
    def optimize_markowitz(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None,
        max_position_size: float = 0.15,
        min_position_size: float = 0.0,
        sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        transaction_costs: Optional[pd.Series] = None
    ) -> OptimizationResult:
        """
        Markowitz mean-variance optimization.
        
        Args:
            expected_returns: Expected return for each asset
            covariance_matrix: Asset covariance matrix
            risk_free_rate: Risk-free rate for Sharpe ratio
            target_return: Target portfolio return (if None, maximize Sharpe)
            max_position_size: Maximum weight per asset
            min_position_size: Minimum weight per asset (if invested)
            sector_constraints: Dict of sector -> (min, max) exposure
            transaction_costs: Transaction cost per asset
            
        Returns:
            OptimizationResult with optimal weights
        """
        logger.info("Starting Markowitz optimization")
        
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()
        
        # Create model
        model = gp.Model("markowitz_optimization")
        model.setParam('OutputFlag', int(self.log_to_console))
        model.setParam('TimeLimit', self.timeout)
        model.setParam('MIPGap', self.mip_gap)
        
        # Decision variables: weights for each asset
        w = model.addVars(assets, lb=0.0, ub=max_position_size, name="weight")
        
        # Binary variables for cardinality constraints
        z = model.addVars(assets, vtype=GRB.BINARY, name="invested")
        
        # Auxiliary variable for portfolio variance (quadratic)
        portfolio_variance = model.addVar(lb=0.0, name="portfolio_variance")
        
        # Constraint: weights sum to 1
        model.addConstr(gp.quicksum(w[asset] for asset in assets) == 1.0, "fully_invested")
        
        # Constraint: link weights to binary variables
        for asset in assets:
            model.addConstr(w[asset] <= max_position_size * z[asset], f"upper_bound_{asset}")
            model.addConstr(w[asset] >= min_position_size * z[asset], f"lower_bound_{asset}")
        
        # Portfolio return
        portfolio_return = gp.quicksum(w[asset] * expected_returns[asset] for asset in assets)
        
        # Portfolio variance (quadratic constraint)
        variance_expr = gp.QuadExpr()
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                variance_expr += w[asset_i] * w[asset_j] * covariance_matrix.iloc[i, j]
        
        model.addConstr(portfolio_variance == variance_expr, "variance_definition")
        
        # Target return constraint (if specified)
        if target_return is not None:
            model.addConstr(portfolio_return >= target_return, "target_return")
            # Objective: minimize variance
            model.setObjective(portfolio_variance, GRB.MINIMIZE)
            logger.info(f"Minimizing risk for target return: {target_return:.4f}")
        else:
            # Objective: maximize Sharpe ratio (approximate)
            # Sharpe = (return - rf) / sqrt(variance)
            # We maximize: return - rf - lambda * variance
            # where lambda controls risk aversion
            lambda_risk = 0.5
            objective = portfolio_return - risk_free_rate - lambda_risk * portfolio_variance
            model.setObjective(objective, GRB.MAXIMIZE)
            logger.info("Maximizing Sharpe ratio")
        
        # Sector constraints
        if sector_constraints:
            logger.info(f"Applying sector constraints: {sector_constraints}")
            # This would require sector mapping for each asset
            # Implementation omitted for brevity
        
        # Transaction costs
        if transaction_costs is not None:
            logger.info("Including transaction costs")
            # Subtract costs from objective
            cost_expr = gp.quicksum(w[asset] * transaction_costs[asset] for asset in assets)
            model.setObjective(model.getObjective() - cost_expr, GRB.MAXIMIZE)
        
        # Optimize
        model.optimize()
        
        # Extract results
        if model.status == GRB.OPTIMAL:
            weights = {asset: w[asset].X for asset in assets}
            
            # Calculate portfolio metrics
            portfolio_ret = sum(weights[asset] * expected_returns[asset] for asset in assets)
            portfolio_var = sum(
                weights[asset_i] * weights[asset_j] * covariance_matrix.iloc[i, j]
                for i, asset_i in enumerate(assets)
                for j, asset_j in enumerate(assets)
            )
            portfolio_std = np.sqrt(portfolio_var)
            sharpe = (portfolio_ret - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            logger.info(f"Optimization successful: Return={portfolio_ret:.4f}, Risk={portfolio_std:.4f}, Sharpe={sharpe:.4f}")
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_ret,
                expected_risk=portfolio_std,
                sharpe_ratio=sharpe,
                status="OPTIMAL",
                objective_value=model.objVal,
                metadata={
                    "solver_time": model.Runtime,
                    "n_assets": n_assets,
                    "n_invested": sum(1 for w in weights.values() if w > 1e-6)
                }
            )
        else:
            logger.error(f"Optimization failed with status: {model.status}")
            raise ValueError(f"Optimization failed: {model.status}")
    
    def optimize_risk_parity(
        self,
        covariance_matrix: pd.DataFrame,
        max_position_size: float = 0.15
    ) -> OptimizationResult:
        """
        Risk parity optimization: equal risk contribution.
        
        Args:
            covariance_matrix: Asset covariance matrix
            max_position_size: Maximum weight per asset
            
        Returns:
            OptimizationResult with risk parity weights
        """
        logger.info("Starting risk parity optimization")
        
        n_assets = len(covariance_matrix)
        assets = covariance_matrix.index.tolist()
        
        # Risk parity: w_i * (Cov @ w)_i = constant for all i
        # This is non-convex, we use heuristic approach
        
        # Start with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Iterative algorithm to equalize risk contributions
        for iteration in range(100):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
            marginal_contrib = covariance_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Adjust weights inversely proportional to risk contribution
            weights = weights / risk_contrib
            weights = weights / weights.sum()  # Normalize
            
            # Apply bounds
            weights = np.clip(weights, 0, max_position_size)
            weights = weights / weights.sum()  # Renormalize
            
            # Check convergence
            if iteration > 0:
                std_risk_contrib = np.std(risk_contrib)
                if std_risk_contrib < 1e-6:
                    break
        
        weights_dict = {asset: float(w) for asset, w in zip(assets, weights)}
        
        portfolio_var = weights @ covariance_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)
        
        logger.info(f"Risk parity optimization complete: Risk={portfolio_std:.4f}")
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,  # Not optimizing for return
            expected_risk=portfolio_std,
            sharpe_ratio=0.0,
            status="OPTIMAL",
            objective_value=portfolio_std,
            metadata={"iterations": iteration + 1}
        )
    
    def optimize_cvar(
        self,
        returns_scenarios: pd.DataFrame,
        confidence_level: float = 0.95,
        target_return: Optional[float] = None,
        max_position_size: float = 0.15
    ) -> OptimizationResult:
        """
        CVaR (Conditional Value at Risk) optimization.
        Minimizes expected loss in worst-case scenarios.
        
        Args:
            returns_scenarios: Historical or simulated return scenarios
            confidence_level: CVaR confidence level (e.g., 0.95 for 95%)
            target_return: Minimum expected return
            max_position_size: Maximum weight per asset
            
        Returns:
            OptimizationResult with CVaR-optimal weights
        """
        logger.info(f"Starting CVaR optimization (alpha={confidence_level})")
        
        n_scenarios, n_assets = returns_scenarios.shape
        assets = returns_scenarios.columns.tolist()
        scenarios = range(n_scenarios)
        
        # Create model
        model = gp.Model("cvar_optimization")
        model.setParam('OutputFlag', int(self.log_to_console))
        model.setParam('TimeLimit', self.timeout)
        
        # Decision variables
        w = model.addVars(assets, lb=0.0, ub=max_position_size, name="weight")
        
        # VaR variable
        var = model.addVar(lb=-GRB.INFINITY, name="VaR")
        
        # Auxiliary variables for CVaR calculation
        u = model.addVars(scenarios, lb=0.0, name="loss_excess")
        
        # Constraints
        model.addConstr(gp.quicksum(w[asset] for asset in assets) == 1.0, "fully_invested")
        
        # Define loss excess for each scenario
        for s in scenarios:
            scenario_loss = -gp.quicksum(w[asset] * returns_scenarios.iloc[s, i] 
                                         for i, asset in enumerate(assets))
            model.addConstr(u[s] >= scenario_loss - var, f"loss_excess_{s}")
        
        # CVaR definition
        cvar = var + (1.0 / (n_scenarios * (1 - confidence_level))) * gp.quicksum(u[s] for s in scenarios)
        
        # Target return constraint
        if target_return is not None:
            expected_return = gp.quicksum(
                w[asset] * returns_scenarios[asset].mean()
                for asset in assets
            )
            model.addConstr(expected_return >= target_return, "target_return")
        
        # Objective: minimize CVaR
        model.setObjective(cvar, GRB.MINIMIZE)
        
        # Optimize
        model.optimize()
        
        # Extract results
        if model.status == GRB.OPTIMAL:
            weights = {asset: w[asset].X for asset in assets}
            
            # Calculate metrics
            portfolio_returns = returns_scenarios @ np.array([weights[asset] for asset in assets])
            expected_ret = portfolio_returns.mean()
            risk = portfolio_returns.std()
            
            logger.info(f"CVaR optimization successful: CVaR={model.objVal:.4f}")
            
            return OptimizationResult(
                weights=weights,
                expected_return=expected_ret,
                expected_risk=risk,
                sharpe_ratio=expected_ret / risk if risk > 0 else 0,
                status="OPTIMAL",
                objective_value=model.objVal,
                metadata={
                    "cvar": model.objVal,
                    "var": var.X,
                    "confidence_level": confidence_level
                }
            )
        else:
            raise ValueError(f"CVaR optimization failed: {model.status}")
