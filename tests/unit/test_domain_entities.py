"""
Unit tests for domain entities.
Testing Portfolio and Asset entities with DDD principles.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from src.domain.entities.portfolio import Portfolio, Position
from src.domain.entities.asset import Asset, AssetType, AssetSector


class TestAsset:
    """Test suite for Asset entity."""
    
    def test_create_asset(self):
        """Test creating a valid asset."""
        asset = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY,
            current_price=150.0
        )
        
        assert asset.ticker == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.asset_type == AssetType.STOCK
        assert asset.sector == AssetSector.TECHNOLOGY
        assert asset.current_price == 150.0
        assert isinstance(asset.id, UUID)
    
    def test_ticker_must_be_uppercase(self):
        """Test that ticker validation enforces uppercase."""
        with pytest.raises(ValueError, match="Ticker must be uppercase"):
            Asset(
                ticker="aapl",  # lowercase should fail
                name="Apple Inc.",
                asset_type=AssetType.STOCK,
                sector=AssetSector.TECHNOLOGY
            )
    
    def test_negative_price_validation(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Asset(
                ticker="AAPL",
                name="Apple Inc.",
                asset_type=AssetType.STOCK,
                sector=AssetSector.TECHNOLOGY,
                current_price=-10.0
            )
    
    def test_update_price(self):
        """Test updating asset price."""
        asset = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY,
            current_price=150.0
        )
        
        new_price = 155.0
        timestamp = datetime.utcnow()
        asset.update_price(new_price, timestamp)
        
        assert asset.current_price == new_price
        assert asset.last_price_update == timestamp
    
    def test_update_risk_metrics(self):
        """Test updating risk metrics."""
        asset = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY
        )
        
        asset.update_risk_metrics(
            beta=1.2,
            volatility=0.25,
            sharpe_ratio=1.5
        )
        
        assert asset.beta == 1.2
        assert asset.volatility == 0.25
        assert asset.sharpe_ratio == 1.5
    
    def test_negative_volatility_rejected(self):
        """Test that negative volatility is rejected."""
        asset = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY
        )
        
        with pytest.raises(ValueError, match="Volatility cannot be negative"):
            asset.update_risk_metrics(volatility=-0.1)
    
    def test_is_liquid(self):
        """Test liquidity check."""
        asset = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY
        )
        
        # No price update -> not liquid
        assert not asset.is_liquid()
        
        # Recent price update -> liquid
        asset.update_price(150.0, datetime.utcnow())
        assert asset.is_liquid()
    
    def test_asset_equality(self):
        """Test that assets are equal based on ID."""
        asset1 = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY
        )
        
        asset2 = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY
        )
        
        # Different IDs -> not equal
        assert asset1 != asset2
        
        # Same ID -> equal
        asset2.id = asset1.id
        assert asset1 == asset2
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=AssetSector.TECHNOLOGY,
            current_price=150.0,
            beta=1.2,
            volatility=0.25
        )
        
        # Convert to dict
        asset_dict = original.to_dict()
        
        # Recreate from dict
        recreated = Asset.from_dict(asset_dict)
        
        assert recreated.id == original.id
        assert recreated.ticker == original.ticker
        assert recreated.asset_type == original.asset_type
        assert recreated.current_price == original.current_price
        assert recreated.beta == original.beta


class TestPortfolio:
    """Test suite for Portfolio entity."""
    
    def test_create_portfolio(self):
        """Test creating a valid portfolio."""
        portfolio = Portfolio(
            name="Tech Portfolio",
            owner_id="user123",
            cash=Decimal("100000.0")
        )
        
        assert portfolio.name == "Tech Portfolio"
        assert portfolio.owner_id == "user123"
        assert portfolio.cash == Decimal("100000.0")
        assert isinstance(portfolio.id, UUID)
        assert len(portfolio.positions) == 0
    
    def test_portfolio_name_required(self):
        """Test that portfolio name is required."""
        with pytest.raises(ValueError, match="Portfolio must have a name"):
            Portfolio(
                name="",
                owner_id="user123"
            )
    
    def test_negative_cash_rejected(self):
        """Test that negative cash is rejected."""
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            Portfolio(
                name="Test Portfolio",
                owner_id="user123",
                cash=Decimal("-1000.0")
            )
    
    def test_get_total_value_empty_portfolio(self):
        """Test total value calculation for empty portfolio."""
        portfolio = Portfolio(
            name="Test Portfolio",
            owner_id="user123",
            cash=Decimal("50000.0")
        )
        
        assert portfolio.get_total_value() == Decimal("50000.0")
    
    def test_get_total_value_with_positions(self):
        """Test total value calculation with positions."""
        portfolio = Portfolio(
            name="Test Portfolio",
            owner_id="user123",
            cash=Decimal("50000.0")
        )
        
        # Add a position manually (simplified for testing)
        from uuid import uuid4
        asset_id = uuid4()
        position = Position(
            asset_id=asset_id,
            ticker="AAPL",
            quantity=Decimal("100"),
            average_cost=Decimal("150.0"),
            current_value=Decimal("15000.0")
        )
        portfolio.positions[asset_id] = position
        
        # Total = cash + position value
        expected_total = Decimal("50000.0") + Decimal("15000.0")
        assert portfolio.get_total_value() == expected_total
    
    def test_to_dict_and_from_dict(self):
        """Test portfolio serialization."""
        original = Portfolio(
            name="Test Portfolio",
            owner_id="user123",
            cash=Decimal("100000.0"),
            max_position_size=0.15,
            strategy="Balanced"
        )
        
        # Convert to dict
        portfolio_dict = original.to_dict()
        
        # Recreate from dict
        recreated = Portfolio.from_dict(portfolio_dict)
        
        assert recreated.id == original.id
        assert recreated.name == original.name
        assert recreated.owner_id == original.owner_id
        assert recreated.cash == original.cash
        assert recreated.strategy == original.strategy


class TestPosition:
    """Test suite for Position value object."""
    
    def test_create_position(self):
        """Test creating a position."""
        from uuid import uuid4
        
        asset_id = uuid4()
        position = Position(
            asset_id=asset_id,
            ticker="AAPL",
            quantity=Decimal("100"),
            average_cost=Decimal("150.0")
        )
        
        assert position.asset_id == asset_id
        assert position.ticker == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.0")
    
    def test_update_current_value(self):
        """Test updating position value."""
        from uuid import uuid4
        
        position = Position(
            asset_id=uuid4(),
            ticker="AAPL",
            quantity=Decimal("100"),
            average_cost=Decimal("150.0")
        )
        
        current_price = Decimal("160.0")
        position.update_current_value(current_price)
        
        expected_value = Decimal("100") * Decimal("160.0")
        expected_pnl = expected_value - (Decimal("100") * Decimal("150.0"))
        
        assert position.current_value == expected_value
        assert position.unrealized_pnl == expected_pnl
    
    def test_position_to_dict(self):
        """Test position serialization."""
        from uuid import uuid4
        
        asset_id = uuid4()
        position = Position(
            asset_id=asset_id,
            ticker="AAPL",
            quantity=Decimal("100"),
            average_cost=Decimal("150.0"),
            current_value=Decimal("16000.0"),
            weight=0.15,
            unrealized_pnl=Decimal("1000.0")
        )
        
        pos_dict = position.to_dict()
        
        assert pos_dict["asset_id"] == str(asset_id)
        assert pos_dict["ticker"] == "AAPL"
        assert pos_dict["quantity"] == 100.0
        assert pos_dict["weight"] == 0.15


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
