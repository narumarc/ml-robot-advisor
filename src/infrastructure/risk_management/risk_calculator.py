"""
Risk Management Module.
Calcul des métriques de risque (VaR, ES, stress testing, etc.)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Métriques de risque calculées."""
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class RiskCalculator:
    """
    Calculateur de métriques de risque.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk calculator.
        
        Args:
            risk_free_rate: Taux sans risque annualisé
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Risk Calculator initialized (rf={risk_free_rate})")
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.95 or 0.99)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR value (negative number representing loss)
        """
        if method == "historical":
            # Historical VaR: empirical quantile
            var = returns.quantile(1 - confidence_level)
            
        elif method == "parametric":
            # Parametric VaR: assume normal distribution
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            
        elif method == "monte_carlo":
            # Monte Carlo VaR: simulate returns
            mean = returns.mean()
            std = returns.std()
            simulations = np.random.normal(mean, std, 10000)
            var = np.percentile(simulations, (1 - confidence_level) * 100)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        logger.debug(f"VaR ({confidence_level:.0%}, {method}): {var:.4f}")
        return float(var)
    
    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (ES / CVaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            Expected shortfall (conditional expected loss)
        """
        var = self.calculate_var(returns, confidence_level, method="historical")
        
        # ES = mean of returns below VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) > 0:
            es = tail_returns.mean()
        else:
            es = var
        
        logger.debug(f"ES ({confidence_level:.0%}): {es:.4f}")
        return float(es)
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize (assumes daily returns)
            
        Returns:
            Volatility
        """
        vol = returns.std()
        
        if annualize:
            vol = vol * np.sqrt(252)  # Annualize for daily returns
        
        logger.debug(f"Volatility (annualized={annualize}): {vol:.4f}")
        return float(vol)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown (negative number)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        logger.debug(f"Max Drawdown: {max_dd:.4f}")
        return float(max_dd)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily rf
        
        if annualize:
            sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe = excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        logger.debug(f"Sharpe Ratio: {sharpe:.4f}")
        return float(sharpe)
    
    def calculate_sortino_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if annualize:
            sortino = (excess_returns.mean() * 252) / (downside_std * np.sqrt(252))
        else:
            sortino = excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        logger.debug(f"Sortino Ratio: {sortino:.4f}")
        return float(sortino)
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (systematic risk).
        
        Args:
            asset_returns: Returns of the asset
            market_returns: Returns of the market index
            
        Returns:
            Beta coefficient
        """
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        logger.debug(f"Beta: {beta:.4f}")
        return float(beta)
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate all risk metrics at once.
        
        Args:
            returns: Portfolio/asset returns
            market_returns: Optional market returns for beta calculation
            
        Returns:
            RiskMetrics dataclass with all metrics
        """
        logger.info("Calculating all risk metrics")
        
        metrics = RiskMetrics(
            var_95=self.calculate_var(returns, 0.95),
            var_99=self.calculate_var(returns, 0.99),
            expected_shortfall_95=self.calculate_expected_shortfall(returns, 0.95),
            expected_shortfall_99=self.calculate_expected_shortfall(returns, 0.99),
            volatility=self.calculate_volatility(returns),
            max_drawdown=self.calculate_max_drawdown(returns),
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            sortino_ratio=self.calculate_sortino_ratio(returns),
            skewness=float(returns.skew()),
            kurtosis=float(returns.kurtosis())
        )
        
        if market_returns is not None:
            metrics.beta = self.calculate_beta(returns, market_returns)
        
        logger.info("Risk metrics calculation complete")
        return metrics


class StressTester:
    """
    Stress testing de portefeuille.
    """
    
    def __init__(self):
        logger.info("Stress Tester initialized")
    
    def market_crash_scenario(
        self,
        portfolio_weights: Dict[str, float],
        shock_percentage: float = -0.20
    ) -> float:
        """
        Simulate market crash (all assets drop).
        
        Args:
            portfolio_weights: Dictionary of ticker -> weight
            shock_percentage: Percentage drop (e.g., -0.20 for -20%)
            
        Returns:
            Portfolio loss
        """
        total_loss = sum(portfolio_weights.values()) * shock_percentage
        logger.info(f"Market crash scenario ({shock_percentage:.0%}): Loss = {total_loss:.4f}")
        return total_loss
    
    def sector_shock_scenario(
        self,
        portfolio_weights: Dict[str, float],
        sector_mapping: Dict[str, str],
        shocked_sector: str,
        shock_percentage: float = -0.30
    ) -> float:
        """
        Simulate sector-specific shock.
        
        Args:
            portfolio_weights: Dictionary of ticker -> weight
            sector_mapping: Dictionary of ticker -> sector
            shocked_sector: Sector to shock (e.g., "TECHNOLOGY")
            shock_percentage: Shock magnitude
            
        Returns:
            Portfolio loss
        """
        loss = 0.0
        
        for ticker, weight in portfolio_weights.items():
            if sector_mapping.get(ticker) == shocked_sector:
                loss += weight * shock_percentage
        
        logger.info(f"Sector shock ({shocked_sector}, {shock_percentage:.0%}): Loss = {loss:.4f}")
        return loss
    
    def interest_rate_shock(
        self,
        portfolio_weights: Dict[str, float],
        durations: Dict[str, float],
        rate_change: float = 0.01
    ) -> float:
        """
        Simulate interest rate shock (affects bonds).
        
        Args:
            portfolio_weights: Dictionary of ticker -> weight
            durations: Dictionary of ticker -> duration
            rate_change: Change in interest rates (e.g., 0.01 for +1%)
            
        Returns:
            Portfolio impact
        """
        loss = 0.0
        
        for ticker, weight in portfolio_weights.items():
            duration = durations.get(ticker, 0)
            # Bond price change ≈ -Duration × ΔRate
            price_change = -duration * rate_change
            loss += weight * price_change
        
        logger.info(f"Interest rate shock (+{rate_change:.2%}): Impact = {loss:.4f}")
        return loss
    
    def run_all_scenarios(
        self,
        portfolio_weights: Dict[str, float],
        sector_mapping: Optional[Dict[str, str]] = None,
        durations: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Run all stress test scenarios.
        
        Args:
            portfolio_weights: Portfolio weights
            sector_mapping: Optional sector mapping
            durations: Optional duration mapping
            
        Returns:
            Dictionary of scenario -> impact
        """
        logger.info("Running all stress test scenarios")
        
        results = {
            "market_crash_20": self.market_crash_scenario(portfolio_weights, -0.20),
            "market_crash_30": self.market_crash_scenario(portfolio_weights, -0.30),
            "market_crash_40": self.market_crash_scenario(portfolio_weights, -0.40),
        }
        
        if sector_mapping:
            sectors = set(sector_mapping.values())
            for sector in sectors:
                scenario_name = f"sector_shock_{sector.lower()}_30"
                results[scenario_name] = self.sector_shock_scenario(
                    portfolio_weights,
                    sector_mapping,
                    sector,
                    -0.30
                )
        
        if durations:
            results["interest_rate_up_100bp"] = self.interest_rate_shock(
                portfolio_weights,
                durations,
                0.01
            )
            results["interest_rate_up_200bp"] = self.interest_rate_shock(
                portfolio_weights,
                durations,
                0.02
            )
        
        logger.info(f"Stress testing complete: {len(results)} scenarios")
        return results
