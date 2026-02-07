"""
Unit Tests for Domain Entities

Tests for Portfolio and Position entities following DDD principles.
"""

import pytest
from decimal import Decimal
from uuid import uuid4
from datetime import datetime

from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.value_objects.risk_metrics import RiskMetrics


class TestPosition:
    """Test suite for Position entity."""
    
    def test_create_position(self):
        """Test creating a valid position."""
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.25")
        )
        
        assert position.ticker == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.market_value == Decimal("17500.00")
        assert position.unrealized_pnl == Decimal("2500.00")
    
    def test_position_negative_quantity_raises_error(self):
        """Test that negative quantity raises error."""
        with pytest.raises(ValueError, match="Quantity cannot be negative"):
            Position(
                position_id=uuid4(),
                ticker="AAPL",
                quantity=Decimal("-100"),
                cost_basis=Decimal("150.00"),
                current_price=Decimal("175.00")
            )
    
    def test_position_zero_price_raises_error(self):
        """Test that zero price raises error."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Position(
                position_id=uuid4(),
                ticker="AAPL",
                quantity=Decimal("100"),
                cost_basis=Decimal("0"),
                current_price=Decimal("175.00")
            )
    
    def test_update_quantity(self):
        """Test updating position quantity."""
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00")
        )
        
        position.update_quantity(Decimal("150"))
        assert position.quantity == Decimal("150")
        assert position.market_value == Decimal("26250.00")
    
    def test_add_shares_updates_cost_basis(self):
        """Test that adding shares updates weighted average cost basis."""
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00")
        )
        
        # Add 50 shares at $160
        position.add_shares(Decimal("50"), Decimal("160.00"))
        
        # New cost basis should be weighted average
        expected_cost_basis = (Decimal("100") * Decimal("150") + Decimal("50") * Decimal("160")) / Decimal("150")
        assert position.quantity == Decimal("150")
        assert abs(position.cost_basis - expected_cost_basis) < Decimal("0.01")
    
    def test_remove_shares_calculates_realized_pnl(self):
        """Test that removing shares calculates correct realized P&L."""
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00")
        )
        
        # Sell 30 shares
        realized_pnl = position.remove_shares(Decimal("30"))
        
        # Realized P&L = 30 * (175 - 150) = 750
        assert realized_pnl == Decimal("750.00")
        assert position.quantity == Decimal("70")
    
    def test_remove_more_shares_than_owned_raises_error(self):
        """Test that selling more shares than owned raises error."""
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00")
        )
        
        with pytest.raises(ValueError, match="Cannot sell more shares than owned"):
            position.remove_shares(Decimal("150"))
    
    def test_position_serialization(self):
        """Test position to_dict and from_dict."""
        original = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.25")
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = Position.from_dict(data)
        
        assert restored.ticker == original.ticker
        assert restored.quantity == original.quantity
        assert restored.cost_basis == original.cost_basis
        assert restored.current_price == original.current_price


class TestPortfolio:
    """Test suite for Portfolio entity."""
    
    def test_create_portfolio(self):
        """Test creating a valid portfolio."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        assert portfolio.name == "Test Portfolio"
        assert portfolio.current_value == Decimal("100000")
        assert portfolio.risk_tolerance == Decimal("0.15")
    
    def test_risk_tolerance_out_of_range_raises_error(self):
        """Test that invalid risk tolerance raises error."""
        with pytest.raises(ValueError, match="Risk tolerance must be between 0 and 1"):
            Portfolio(
                portfolio_id=uuid4(),
                name="Test Portfolio",
                initial_capital=Decimal("100000"),
                risk_tolerance=Decimal("1.5"),  # Invalid
                owner_id="user123"
            )
    
    def test_negative_capital_raises_error(self):
        """Test that negative capital raises error."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(
                portfolio_id=uuid4(),
                name="Test Portfolio",
                initial_capital=Decimal("-100000"),
                risk_tolerance=Decimal("0.15"),
                owner_id="user123"
            )
    
    def test_add_position(self):
        """Test adding position to portfolio."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.17")
        )
        
        portfolio.add_position(position)
        assert "AAPL" in portfolio.positions
        assert len(portfolio.positions) == 1
    
    def test_add_duplicate_position_raises_error(self):
        """Test that adding duplicate position raises error."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        position1 = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.17")
        )
        
        position2 = Position(
            position_id=uuid4(),
            ticker="AAPL",  # Same ticker
            quantity=Decimal("50"),
            cost_basis=Decimal("160.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.09")
        )
        
        portfolio.add_position(position1)
        
        with pytest.raises(ValueError, match="Position AAPL already exists"):
            portfolio.add_position(position2)
    
    def test_add_position_exceeding_100_percent_raises_error(self):
        """Test that total allocation cannot exceed 100%."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        position1 = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.60")
        )
        
        position2 = Position(
            position_id=uuid4(),
            ticker="GOOGL",
            quantity=Decimal("50"),
            cost_basis=Decimal("100.00"),
            current_price=Decimal("120.00"),
            allocation=Decimal("0.50")  # Would exceed 100%
        )
        
        portfolio.add_position(position1)
        
        with pytest.raises(ValueError, match="Total allocation would exceed 100%"):
            portfolio.add_position(position2)
    
    def test_update_position(self):
        """Test updating existing position."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.17")
        )
        
        portfolio.add_position(position)
        portfolio.update_position("AAPL", Decimal("150"), Decimal("180.00"))
        
        updated_position = portfolio.positions["AAPL"]
        assert updated_position.quantity == Decimal("150")
        assert updated_position.current_price == Decimal("180.00")
    
    def test_remove_position(self):
        """Test removing position from portfolio."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.17")
        )
        
        portfolio.add_position(position)
        portfolio.remove_position("AAPL")
        
        assert "AAPL" not in portfolio.positions
        assert len(portfolio.positions) == 0
    
    def test_rebalance_emits_domain_event(self):
        """Test that rebalancing emits domain event."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        # Add initial positions
        position1 = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.50")
        )
        
        position2 = Position(
            position_id=uuid4(),
            ticker="GOOGL",
            quantity=Decimal("50"),
            cost_basis=Decimal("100.00"),
            current_price=Decimal("120.00"),
            allocation=Decimal("0.50")
        )
        
        portfolio.add_position(position1)
        portfolio.add_position(position2)
        
        # Rebalance
        new_allocations = {
            "AAPL": Decimal("0.60"),
            "GOOGL": Decimal("0.40")
        }
        
        portfolio.rebalance(new_allocations, transaction_cost=Decimal("0.001"))
        
        # Check events
        events = portfolio.get_domain_events()
        assert len(events) > 0
        assert events[0].__class__.__name__ == "PortfolioRebalanced"
    
    def test_update_risk_metrics_triggers_alert_on_threshold_exceeded(self):
        """Test that exceeding risk threshold triggers event."""
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),  # 15% max volatility
            owner_id="user123"
        )
        
        # Create risk metrics exceeding threshold
        risk_metrics = RiskMetrics(
            portfolio_volatility=Decimal("0.25"),  # Exceeds tolerance
            downside_volatility=Decimal("0.18"),
            var_95=Decimal("-8000"),
            var_99=Decimal("-12000")
        )
        
        portfolio.update_risk_metrics(risk_metrics)
        
        # Check for risk threshold exceeded event
        events = portfolio.get_domain_events()
        risk_events = [e for e in events if e.__class__.__name__ == "RiskThresholdExceeded"]
        assert len(risk_events) > 0
    
    def test_portfolio_serialization(self):
        """Test portfolio to_dict and from_dict."""
        original = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            risk_tolerance=Decimal("0.15"),
            owner_id="user123"
        )
        
        position = Position(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            cost_basis=Decimal("150.00"),
            current_price=Decimal("175.00"),
            allocation=Decimal("0.50")
        )
        
        original.add_position(position)
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = Portfolio.from_dict(data)
        
        assert restored.name == original.name
        assert restored.initial_capital == original.initial_capital
        assert restored.risk_tolerance == original.risk_tolerance
        assert "AAPL" in restored.positions


class TestRiskMetrics:
    """Test suite for RiskMetrics value object."""
    
    def test_create_risk_metrics(self):
        """Test creating valid risk metrics."""
        metrics = RiskMetrics(
            portfolio_volatility=Decimal("0.15"),
            downside_volatility=Decimal("0.10"),
            var_95=Decimal("-5000"),
            expected_shortfall_95=Decimal("-7000"),
            max_drawdown=Decimal("-0.12")
        )
        
        assert metrics.portfolio_volatility == Decimal("0.15")
        assert metrics.var_95 == Decimal("-5000")
    
    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises error."""
        with pytest.raises(ValueError, match="Portfolio volatility cannot be negative"):
            RiskMetrics(
                portfolio_volatility=Decimal("-0.15"),
                downside_volatility=Decimal("0.10")
            )
    
    def test_is_high_risk(self):
        """Test high risk detection."""
        metrics = RiskMetrics(
            portfolio_volatility=Decimal("0.25"),  # High
            downside_volatility=Decimal("0.18")
        )
        
        assert metrics.is_high_risk(threshold=Decimal("0.20")) is True
        assert metrics.is_high_risk(threshold=Decimal("0.30")) is False
    
    def test_var_exceeded(self):
        """Test VaR threshold exceeded detection."""
        metrics = RiskMetrics(
            portfolio_volatility=Decimal("0.15"),
            downside_volatility=Decimal("0.10"),
            var_95=Decimal("-8000")  # 8% loss
        )
        
        portfolio_value = Decimal("100000")
        
        # Should exceed 5% threshold
        assert metrics.var_exceeded(portfolio_value, max_loss_pct=Decimal("0.05")) is True
        
        # Should not exceed 10% threshold
        assert metrics.var_exceeded(portfolio_value, max_loss_pct=Decimal("0.10")) is False
    
    def test_risk_metrics_immutable(self):
        """Test that RiskMetrics is immutable."""
        metrics = RiskMetrics(
            portfolio_volatility=Decimal("0.15"),
            downside_volatility=Decimal("0.10")
        )
        
        # Attempting to modify should raise error
        with pytest.raises(Exception):  # dataclass frozen raises various errors
            metrics.portfolio_volatility = Decimal("0.20")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
