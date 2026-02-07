"""
Risk Metrics Value Object

Immutable value object representing portfolio risk measures.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class RiskMetrics:
    """
    Portfolio Risk Metrics
    
    Immutable value object containing various risk measures.
    All metrics are calculated and cannot be modified after creation.
    """
    
    # Volatility measures
    portfolio_volatility: Decimal  # Annualized standard deviation
    downside_volatility: Decimal   # Semi-deviation (downside only)
    
    # Value at Risk measures
    var_95: Optional[Decimal] = None  # 95% confidence VaR
    var_99: Optional[Decimal] = None  # 99% confidence VaR
    
    # Expected Shortfall (Conditional VaR)
    expected_shortfall_95: Optional[Decimal] = None
    expected_shortfall_99: Optional[Decimal] = None
    
    # Drawdown measures
    max_drawdown: Optional[Decimal] = None
    current_drawdown: Optional[Decimal] = None
    
    # Correlation and concentration
    average_correlation: Optional[Decimal] = None
    herfindahl_index: Optional[Decimal] = None  # Concentration measure
    
    # Greeks (if applicable)
    portfolio_beta: Optional[Decimal] = None
    portfolio_delta: Optional[Decimal] = None
    
    # Stress test results
    recession_scenario_loss: Optional[Decimal] = None
    inflation_scenario_loss: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate risk metrics after initialization."""
        if self.portfolio_volatility < Decimal("0"):
            raise ValueError("Portfolio volatility cannot be negative")
        if self.downside_volatility < Decimal("0"):
            raise ValueError("Downside volatility cannot be negative")
    
    def is_high_risk(self, threshold: Decimal = Decimal("0.20")) -> bool:
        """Check if portfolio is considered high risk (>20% volatility)."""
        return self.portfolio_volatility > threshold
    
    def var_exceeded(self, portfolio_value: Decimal, max_loss_pct: Decimal = Decimal("0.05")) -> bool:
        """Check if VaR exceeds acceptable loss threshold."""
        if self.var_95 is None:
            return False
        max_acceptable_loss = portfolio_value * max_loss_pct
        return abs(self.var_95) > max_acceptable_loss
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "portfolio_volatility": float(self.portfolio_volatility),
            "downside_volatility": float(self.downside_volatility),
            "var_95": float(self.var_95) if self.var_95 else None,
            "var_99": float(self.var_99) if self.var_99 else None,
            "expected_shortfall_95": float(self.expected_shortfall_95) if self.expected_shortfall_95 else None,
            "expected_shortfall_99": float(self.expected_shortfall_99) if self.expected_shortfall_99 else None,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown else None,
            "current_drawdown": float(self.current_drawdown) if self.current_drawdown else None,
            "average_correlation": float(self.average_correlation) if self.average_correlation else None,
            "herfindahl_index": float(self.herfindahl_index) if self.herfindahl_index else None,
            "portfolio_beta": float(self.portfolio_beta) if self.portfolio_beta else None,
            "portfolio_delta": float(self.portfolio_delta) if self.portfolio_delta else None,
            "recession_scenario_loss": float(self.recession_scenario_loss) if self.recession_scenario_loss else None,
            "inflation_scenario_loss": float(self.inflation_scenario_loss) if self.inflation_scenario_loss else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RiskMetrics":
        """Create from dictionary representation."""
        return cls(
            portfolio_volatility=Decimal(str(data["portfolio_volatility"])),
            downside_volatility=Decimal(str(data["downside_volatility"])),
            var_95=Decimal(str(data["var_95"])) if data.get("var_95") else None,
            var_99=Decimal(str(data["var_99"])) if data.get("var_99") else None,
            expected_shortfall_95=Decimal(str(data["expected_shortfall_95"])) if data.get("expected_shortfall_95") else None,
            expected_shortfall_99=Decimal(str(data["expected_shortfall_99"])) if data.get("expected_shortfall_99") else None,
            max_drawdown=Decimal(str(data["max_drawdown"])) if data.get("max_drawdown") else None,
            current_drawdown=Decimal(str(data["current_drawdown"])) if data.get("current_drawdown") else None,
            average_correlation=Decimal(str(data["average_correlation"])) if data.get("average_correlation") else None,
            herfindahl_index=Decimal(str(data["herfindahl_index"])) if data.get("herfindahl_index") else None,
            portfolio_beta=Decimal(str(data["portfolio_beta"])) if data.get("portfolio_beta") else None,
            portfolio_delta=Decimal(str(data["portfolio_delta"])) if data.get("portfolio_delta") else None,
            recession_scenario_loss=Decimal(str(data["recession_scenario_loss"])) if data.get("recession_scenario_loss") else None,
            inflation_scenario_loss=Decimal(str(data["inflation_scenario_loss"])) if data.get("inflation_scenario_loss") else None,
        )
