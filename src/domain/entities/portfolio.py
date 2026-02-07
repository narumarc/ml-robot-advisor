"""
Portfolio entity - represents an investment portfolio.
Core aggregate root in the domain model.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from decimal import Decimal


@dataclass
class Position:
    """Represents a position in an asset within the portfolio."""
    asset_id: UUID
    ticker: str
    quantity: Decimal
    average_cost: Decimal
    current_value: Optional[Decimal] = None
    weight: Optional[float] = None
    unrealized_pnl: Optional[Decimal] = None
    
    def update_current_value(self, current_price: Decimal) -> None:
        """Update position value based on current price."""
        self.current_value = current_price * self.quantity
        cost_basis = self.average_cost * self.quantity
        self.unrealized_pnl = self.current_value - cost_basis
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "asset_id": str(self.asset_id),
            "ticker": self.ticker,
            "quantity": float(self.quantity),
            "average_cost": float(self.average_cost),
            "current_value": float(self.current_value) if self.current_value else None,
            "weight": self.weight,
            "unrealized_pnl": float(self.unrealized_pnl) if self.unrealized_pnl else None,
        }


@dataclass
class Portfolio:
    """
    Portfolio aggregate root.
    
    Represents a complete investment portfolio with positions,
    cash, and associated metadata. Enforces business rules.
    """
    
    name: str
    owner_id: str
    id: UUID = field(default_factory=uuid4)
    positions: Dict[UUID, Position] = field(default_factory=dict)
    cash: Decimal = Decimal("0.0")
    currency: str = "USD"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    max_position_size: float = 0.15
    max_sector_exposure: float = 0.30
    min_cash_reserve: Decimal = Decimal("1000.0")
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    strategy: Optional[str] = None
    risk_profile: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def get_total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos.current_value or Decimal("0")
            for pos in self.positions.values()
        )
        return positions_value + self.cash
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "owner_id": self.owner_id,
            "positions": {str(k): v.to_dict() for k, v in self.positions.items()},
            "cash": float(self.cash),
            "currency": self.currency,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
