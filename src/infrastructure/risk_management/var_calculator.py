"""
Risk Management Module

Implements comprehensive risk measures:
- Value at Risk (VaR): Historical, Parametric, Monte Carlo
- Expected Shortfall (CVaR)
- Stress Testing
- Scenario Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_value: float
    confidence_level: float
    method: str
    portfolio_value: float
    var_percentage: float
    time_horizon_days: int


@dataclass
class StressTestResult:
    """Stress test result."""
    scenario_name: str
    portfolio_loss: float
    portfolio_loss_pct: float
    individual_losses: Dict[str, float]
    correlation_breakdown: Optional[Dict] = None


class VaRCalculator:
    """
    Value at Risk Calculator
    
    Implements three main VaR methodologies:
    1. Historical Simulation
    2. Parametric (Variance-Covariance)
    3. Monte Carlo Simulation
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        self.confidence_level = confidence_level
    
    def calculate_historical_var(
        self,
        portfolio_returns: np.ndarray,
        portfolio_value: float,
        time_horizon_days: int = 1
    ) -> VaRResult:
        """
        Calculate Historical VaR.
        
        Uses actual historical returns to estimate VaR.
        Simple and intuitive, but assumes history repeats.
        
        Args:
            portfolio_returns: Historical portfolio returns (1D array)
            portfolio_value: Current portfolio value
            time_horizon_days: Time horizon in days
        
        Returns:
            VaRResult object with VaR estimate
        """
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Find VaR at confidence level
        index = int((1 - self.confidence_level) * len(sorted_returns))
        var_return = sorted_returns[index]
        
        # Scale to time horizon if needed
        if time_horizon_days > 1:
            var_return = var_return * np.sqrt(time_horizon_days)
        
        # Convert to dollar value
        var_value = abs(var_return * portfolio_value)
        var_percentage = abs(var_return * 100)
        
        logger.info(
            f"Historical VaR ({self.confidence_level*100}%, {time_horizon_days}d): "
            f"${var_value:,.2f} ({var_percentage:.2f}%)"
        )
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method="historical",
            portfolio_value=portfolio_value,
            var_percentage=var_percentage,
            time_horizon_days=time_horizon_days
        )
    
    def calculate_parametric_var(
        self,
        portfolio_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_value: float,
        time_horizon_days: int = 1
    ) -> VaRResult:
        """
        Calculate Parametric VaR (Variance-Covariance method).
        
        Assumes returns are normally distributed.
        Fast but may underestimate tail risk.
        
        Args:
            portfolio_weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            portfolio_value: Current portfolio value
            time_horizon_days: Time horizon in days
        
        Returns:
            VaRResult object with VaR estimate
        """
        # Portfolio expected return
        portfolio_return = portfolio_weights.T @ expected_returns
        
        # Portfolio volatility
        portfolio_variance = portfolio_weights.T @ covariance_matrix @ portfolio_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # VaR calculation
        # VaR = μ - z*σ (for losses, we take the negative)
        var_return = -(portfolio_return + z_score * portfolio_volatility)
        
        # Scale to time horizon
        if time_horizon_days > 1:
            var_return = var_return * np.sqrt(time_horizon_days)
        
        # Convert to dollar value
        var_value = var_return * portfolio_value
        var_percentage = var_return * 100
        
        logger.info(
            f"Parametric VaR ({self.confidence_level*100}%, {time_horizon_days}d): "
            f"${var_value:,.2f} ({var_percentage:.2f}%)"
        )
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method="parametric",
            portfolio_value=portfolio_value,
            var_percentage=var_percentage,
            time_horizon_days=time_horizon_days
        )
    
    def calculate_monte_carlo_var(
        self,
        portfolio_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_value: float,
        time_horizon_days: int = 1,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.
        
        Simulates future portfolio returns using random sampling.
        Most flexible, can model non-normal distributions.
        
        Args:
            portfolio_weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            portfolio_value: Current portfolio value
            time_horizon_days: Time horizon in days
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        
        Returns:
            VaRResult object with VaR estimate
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_assets = len(portfolio_weights)
        
        # Portfolio statistics
        portfolio_return = portfolio_weights.T @ expected_returns
        portfolio_volatility = np.sqrt(
            portfolio_weights.T @ covariance_matrix @ portfolio_weights
        )
        
        # Simulate returns
        simulated_returns = np.random.normal(
            loc=portfolio_return * time_horizon_days,
            scale=portfolio_volatility * np.sqrt(time_horizon_days),
            size=n_simulations
        )
        
        # Calculate VaR
        sorted_returns = np.sort(simulated_returns)
        index = int((1 - self.confidence_level) * n_simulations)
        var_return = abs(sorted_returns[index])
        
        # Convert to dollar value
        var_value = var_return * portfolio_value
        var_percentage = var_return * 100
        
        logger.info(
            f"Monte Carlo VaR ({self.confidence_level*100}%, {time_horizon_days}d, "
            f"{n_simulations} sims): ${var_value:,.2f} ({var_percentage:.2f}%)"
        )
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method="monte_carlo",
            portfolio_value=portfolio_value,
            var_percentage=var_percentage,
            time_horizon_days=time_horizon_days
        )
    
    def calculate_expected_shortfall(
        self,
        portfolio_returns: np.ndarray,
        portfolio_value: float,
        time_horizon_days: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Average loss in worst (1-α)% cases.
        Provides better tail risk measure than VaR.
        
        Args:
            portfolio_returns: Historical portfolio returns
            portfolio_value: Current portfolio value
            time_horizon_days: Time horizon in days
        
        Returns:
            Tuple of (ES value, ES percentage)
        """
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Find threshold
        threshold_index = int((1 - self.confidence_level) * len(sorted_returns))
        
        # Expected Shortfall = average of returns below VaR
        es_return = np.mean(sorted_returns[:threshold_index])
        
        # Scale to time horizon
        if time_horizon_days > 1:
            es_return = es_return * np.sqrt(time_horizon_days)
        
        # Convert to dollar value
        es_value = abs(es_return * portfolio_value)
        es_percentage = abs(es_return * 100)
        
        logger.info(
            f"Expected Shortfall ({self.confidence_level*100}%, {time_horizon_days}d): "
            f"${es_value:,.2f} ({es_percentage:.2f}%)"
        )
        
        return es_value, es_percentage


class StressTester:
    """
    Portfolio Stress Testing
    
    Tests portfolio performance under adverse scenarios:
    - Historical crisis scenarios
    - Hypothetical shock scenarios
    - Factor-based stress tests
    """
    
    # Historical crisis scenarios
    CRISIS_SCENARIOS = {
        "2008_financial_crisis": {
            "description": "2008 Financial Crisis",
            "shocks": {
                "equity": -0.38,      # S&P 500 drop
                "bonds": 0.05,        # Flight to quality
                "commodities": -0.35,
                "real_estate": -0.30
            }
        },
        "2020_covid_crash": {
            "description": "2020 COVID-19 Market Crash",
            "shocks": {
                "equity": -0.34,
                "bonds": 0.08,
                "commodities": -0.25,
                "real_estate": -0.20
            }
        },
        "2022_inflation_shock": {
            "description": "2022 Inflation & Rate Hikes",
            "shocks": {
                "equity": -0.18,
                "bonds": -0.13,
                "commodities": 0.15,
                "real_estate": -0.25
            }
        },
        "1987_black_monday": {
            "description": "1987 Black Monday",
            "shocks": {
                "equity": -0.22,
                "bonds": 0.02,
                "commodities": -0.10,
                "real_estate": -0.05
            }
        },
        "dot_com_bubble": {
            "description": "2000-2002 Dot-com Bubble Burst",
            "shocks": {
                "equity": -0.49,
                "bonds": 0.10,
                "commodities": -0.15,
                "real_estate": 0.05
            }
        }
    }
    
    def __init__(self):
        """Initialize stress tester."""
        pass
    
    def run_historical_stress_test(
        self,
        portfolio_weights: np.ndarray,
        asset_classes: List[str],
        portfolio_value: float,
        scenario_name: str
    ) -> StressTestResult:
        """
        Run historical crisis scenario stress test.
        
        Args:
            portfolio_weights: Portfolio weights
            asset_classes: Asset class for each position
            portfolio_value: Current portfolio value
            scenario_name: Name of crisis scenario
        
        Returns:
            StressTestResult object
        """
        if scenario_name not in self.CRISIS_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.CRISIS_SCENARIOS[scenario_name]
        shocks = scenario["shocks"]
        
        # Calculate portfolio loss
        individual_losses = {}
        total_loss = 0
        
        for i, (weight, asset_class) in enumerate(zip(portfolio_weights, asset_classes)):
            if asset_class in shocks:
                shock = shocks[asset_class]
                position_value = weight * portfolio_value
                position_loss = position_value * shock
                
                individual_losses[f"position_{i}_{asset_class}"] = position_loss
                total_loss += position_loss
        
        loss_percentage = (total_loss / portfolio_value) * 100
        
        logger.info(
            f"Stress Test '{scenario['description']}': "
            f"Loss = ${abs(total_loss):,.2f} ({abs(loss_percentage):.2f}%)"
        )
        
        return StressTestResult(
            scenario_name=scenario['description'],
            portfolio_loss=total_loss,
            portfolio_loss_pct=loss_percentage,
            individual_losses=individual_losses
        )
    
    def run_custom_stress_test(
        self,
        portfolio_weights: np.ndarray,
        asset_returns_under_stress: np.ndarray,
        portfolio_value: float,
        scenario_name: str
    ) -> StressTestResult:
        """
        Run custom stress test with specified returns.
        
        Args:
            portfolio_weights: Portfolio weights
            asset_returns_under_stress: Shocked returns for each asset
            portfolio_value: Current portfolio value
            scenario_name: Name of scenario
        
        Returns:
            StressTestResult object
        """
        # Calculate portfolio return under stress
        portfolio_return = portfolio_weights @ asset_returns_under_stress
        portfolio_loss = portfolio_value * portfolio_return
        loss_percentage = portfolio_return * 100
        
        # Individual losses
        individual_losses = {
            f"asset_{i}": weight * portfolio_value * ret
            for i, (weight, ret) in enumerate(zip(portfolio_weights, asset_returns_under_stress))
        }
        
        logger.info(
            f"Custom Stress Test '{scenario_name}': "
            f"Loss = ${abs(portfolio_loss):,.2f} ({abs(loss_percentage):.2f}%)"
        )
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_loss=portfolio_loss,
            portfolio_loss_pct=loss_percentage,
            individual_losses=individual_losses
        )
    
    def run_factor_stress_test(
        self,
        portfolio_weights: np.ndarray,
        factor_exposures: np.ndarray,
        factor_shocks: Dict[str, float],
        portfolio_value: float
    ) -> StressTestResult:
        """
        Run factor-based stress test.
        
        Tests portfolio sensitivity to risk factor shocks.
        
        Args:
            portfolio_weights: Portfolio weights
            factor_exposures: Factor loadings for each asset (n_assets x n_factors)
            factor_shocks: Dictionary of factor shocks
            portfolio_value: Current portfolio value
        
        Returns:
            StressTestResult object
        """
        # Convert factor shocks to array
        shock_array = np.array(list(factor_shocks.values()))
        
        # Calculate asset returns from factor shocks
        asset_returns = factor_exposures @ shock_array
        
        # Portfolio loss
        portfolio_return = portfolio_weights @ asset_returns
        portfolio_loss = portfolio_value * portfolio_return
        loss_percentage = portfolio_return * 100
        
        logger.info(
            f"Factor Stress Test: Loss = ${abs(portfolio_loss):,.2f} "
            f"({abs(loss_percentage):.2f}%)"
        )
        
        return StressTestResult(
            scenario_name="Factor Stress Test",
            portfolio_loss=portfolio_loss,
            portfolio_loss_pct=loss_percentage,
            individual_losses={
                f"asset_{i}": weight * portfolio_value * ret
                for i, (weight, ret) in enumerate(zip(portfolio_weights, asset_returns))
            }
        )
    
    def run_all_crisis_scenarios(
        self,
        portfolio_weights: np.ndarray,
        asset_classes: List[str],
        portfolio_value: float
    ) -> List[StressTestResult]:
        """
        Run all historical crisis scenarios.
        
        Args:
            portfolio_weights: Portfolio weights
            asset_classes: Asset class for each position
            portfolio_value: Current portfolio value
        
        Returns:
            List of StressTestResult objects
        """
        results = []
        
        for scenario_name in self.CRISIS_SCENARIOS.keys():
            result = self.run_historical_stress_test(
                portfolio_weights,
                asset_classes,
                portfolio_value,
                scenario_name
            )
            results.append(result)
        
        # Sort by severity
        results.sort(key=lambda x: x.portfolio_loss)
        
        return results


class DrawdownCalculator:
    """Calculate drawdown metrics for portfolio performance."""
    
    @staticmethod
    def calculate_drawdowns(portfolio_values: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series.
        
        Args:
            portfolio_values: Time series of portfolio values
        
        Returns:
            Drawdown series (percentage from peak)
        """
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown
        drawdowns = (portfolio_values - running_max) / running_max
        
        return drawdowns
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values: Time series of portfolio values
        
        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        drawdowns = DrawdownCalculator.calculate_drawdowns(portfolio_values)
        
        # Find maximum drawdown
        max_dd = drawdowns.min()
        trough_idx = drawdowns.argmin()
        
        # Find peak (highest value before trough)
        peak_idx = portfolio_values[:trough_idx].argmax()
        
        return float(max_dd), int(peak_idx), int(trough_idx)
    
    @staticmethod
    def calculate_recovery_time(
        portfolio_values: np.ndarray,
        peak_index: int,
        trough_index: int
    ) -> Optional[int]:
        """
        Calculate recovery time (time to regain peak).
        
        Args:
            portfolio_values: Time series of portfolio values
            peak_index: Index of peak value
            trough_index: Index of trough value
        
        Returns:
            Number of periods to recovery, or None if not recovered
        """
        peak_value = portfolio_values[peak_index]
        
        # Search for recovery after trough
        for i in range(trough_index, len(portfolio_values)):
            if portfolio_values[i] >= peak_value:
                return i - trough_index
        
        return None  # Not recovered yet
