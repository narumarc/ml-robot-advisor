"""
Position Entity - Represents a single asset position in a portfolio
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional
from uuid import UUID, uuid4


class Position:
    """
    Position Entity
    
    Represents ownership of a specific asset within a portfolio.
    Tracks quantity, cost basis, current value, and performance.
    """
    
    def __init__(
        self,
        position_id: UUID,
        ticker: str,
        quantity: Decimal,
        cost_basis: Decimal,
        current_price: Decimal,
        allocation: Decimal = Decimal("0"),
        acquired_at: Optional[datetime] = None
    ):
        self._position_id = position_id
        self._ticker = ticker.upper()
        self._quantity = quantity
        self._cost_basis = cost_basis
        self._current_price = current_price
        self._allocation = allocation
        self._acquired_at = acquired_at or datetime.utcnow()
        self._updated_at = self._acquired_at
        
        self._validate_position()
    
    @property
    def position_id(self) -> UUID:
        return self._position_id
    
    @property
    def ticker(self) -> str:
        return self._ticker
    
    @property
    def quantity(self) -> Decimal:
        return self._quantity
    
    @property
    def cost_basis(self) -> Decimal:
        return self._cost_basis
    
    @property
    def current_price(self) -> Decimal:
        return self._current_price
    
    @property
    def allocation(self) -> Decimal:
        return self._allocation
    
    @allocation.setter
    def allocation(self, value: Decimal) -> None:
        if not (Decimal("0") <= value <= Decimal("1")):
            raise ValueError("Allocation must be between 0 and 1")
        self._allocation = value
    
    @property
    def market_value(self) -> Decimal:
        """Current market value of the position."""
        return self._quantity * self._current_price
    
    @property
    def total_cost(self) -> Decimal:
        """Total cost of acquiring the position."""
        return self._quantity * self._cost_basis
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized profit/loss."""
        return self.market_value - self.total_cost
    
    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Unrealized profit/loss as percentage."""
        if self.total_cost == Decimal("0"):
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * Decimal("100")
    
    def update_quantity(self, new_quantity: Decimal) -> None:
        """Update position quantity."""
        if new_quantity < Decimal("0"):
            raise ValueError("Quantity cannot be negative")
        
        self._quantity = new_quantity
        self._updated_at = datetime.utcnow()
    
    def update_price(self, new_price: Decimal) -> None:
        """Update current market price."""
        if new_price <= Decimal("0"):
            raise ValueError("Price must be positive")
        
        self._current_price = new_price
        self._updated_at = datetime.utcnow()
    
    def add_shares(self, quantity: Decimal, price: Decimal) -> None:
        """
        Add shares to position (buy more).
        Updates cost basis using weighted average.
        """
        if quantity <= Decimal("0"):
            raise ValueError("Quantity must be positive")
        if price <= Decimal("0"):
            raise ValueError("Price must be positive")
        
        # Calculate new weighted average cost basis
        old_total_cost = self._quantity * self._cost_basis
        new_total_cost = quantity * price
        new_total_quantity = self._quantity + quantity
        
        self._cost_basis = (old_total_cost + new_total_cost) / new_total_quantity
        self._quantity = new_total_quantity
        self._updated_at = datetime.utcnow()
    
    def remove_shares(self, quantity: Decimal) -> Decimal:
        """
        Remove shares from position (sell).
        Returns realized P&L from the sale.
        """
        if quantity <= Decimal("0"):
            raise ValueError("Quantity must be positive")
        if quantity > self._quantity:
            raise ValueError("Cannot sell more shares than owned")
        
        # Calculate realized P&L
        realized_pnl = quantity * (self._current_price - self._cost_basis)
        
        # Update quantity
        self._quantity -= quantity
        self._updated_at = datetime.utcnow()
        
        return realized_pnl
    
    def _validate_position(self) -> None:
        """Validate position invariants."""
        if self._quantity < Decimal("0"):
            raise ValueError("Quantity cannot be negative")
        if self._cost_basis <= Decimal("0"):
            raise ValueError("Cost basis must be positive")
        if self._current_price <= Decimal("0"):
            raise ValueError("Current price must be positive")
        if not self._ticker:
            raise ValueError("Ticker cannot be empty")
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary representation."""
        return {
            "position_id": str(self._position_id),
            "ticker": self._ticker,
            "quantity": float(self._quantity),
            "cost_basis": float(self._cost_basis),
            "current_price": float(self._current_price),
            "allocation": float(self._allocation),
            "market_value": float(self.market_value),
            "unrealized_pnl": float(self.unrealized_pnl),
            "unrealized_pnl_percent": float(self.unrealized_pnl_percent),
            "acquired_at": self._acquired_at.isoformat(),
            "updated_at": self._updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Position":
        """Create position from dictionary representation."""
        position = cls(
            position_id=UUID(data["position_id"]),
            ticker=data["ticker"],
            quantity=Decimal(str(data["quantity"])),
            cost_basis=Decimal(str(data["cost_basis"])),
            current_price=Decimal(str(data["current_price"])),
            allocation=Decimal(str(data.get("allocation", "0"))),
            acquired_at=datetime.fromisoformat(data["acquired_at"])
        )
        position._updated_at = datetime.fromisoformat(data["updated_at"])
        return position
    
    def __repr__(self) -> str:
        return (
            f"Position(ticker='{self._ticker}', quantity={self._quantity}, "
            f"value={self.market_value}, pnl={self.unrealized_pnl_percent:.2f}%)"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Position):
            return False
        return self._position_id == other._position_id
    
    def __hash__(self) -> int:
        return hash(self._position_id)
