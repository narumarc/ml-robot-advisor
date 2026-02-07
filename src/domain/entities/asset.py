"""
Asset entity - represents a financial asset in the portfolio.
Follows DDD principles with rich domain model.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum
from uuid import UUID, uuid4


class AssetType(str, Enum):
    """Types of financial assets."""
    STOCK = "STOCK"
    BOND = "BOND"
    ETF = "ETF"
    COMMODITY = "COMMODITY"
    CRYPTO = "CRYPTO"
    CASH = "CASH"


class AssetSector(str, Enum):
    """Market sectors for classification."""
    TECHNOLOGY = "TECHNOLOGY"
    FINANCE = "FINANCE"
    HEALTHCARE = "HEALTHCARE"
    ENERGY = "ENERGY"
    CONSUMER = "CONSUMER"
    INDUSTRIAL = "INDUSTRIAL"
    REAL_ESTATE = "REAL_ESTATE"
    UTILITIES = "UTILITIES"
    MATERIALS = "MATERIALS"
    TELECOMMUNICATIONS = "TELECOMMUNICATIONS"
    OTHER = "OTHER"


@dataclass
class Asset:
    """
    Asset entity representing a tradeable financial instrument.
    
    This is a core domain entity with business logic and invariants.
    """
    
    ticker: str
    name: str
    asset_type: AssetType
    sector: AssetSector
    id: UUID = field(default_factory=uuid4)
    currency: str = "USD"
    exchange: Optional[str] = None
    market_cap: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)
    
    # Price data
    current_price: Optional[float] = None
    last_price_update: Optional[datetime] = None
    
    # Risk metrics
    beta: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Validate business rules after initialization."""
        self._validate_invariants()
    
    def _validate_invariants(self) -> None:
        """Ensure domain invariants are maintained."""
        if not self.ticker:
            raise ValueError("Asset must have a ticker symbol")
        
        if self.ticker != self.ticker.upper():
            raise ValueError("Ticker must be uppercase")
        
        if self.current_price is not None and self.current_price < 0:
            raise ValueError("Price cannot be negative")
        
        if self.volatility is not None and self.volatility < 0:
            raise ValueError("Volatility cannot be negative")
    
    def update_price(self, price: float, timestamp: datetime) -> None:
        """
        Update asset price with business validation.
        
        Args:
            price: New price
            timestamp: Time of price update
            
        Raises:
            ValueError: If price is invalid
        """
        if price < 0:
            raise ValueError(f"Invalid price for {self.ticker}: {price}")
        
        self.current_price = price
        self.last_price_update = timestamp
        self.updated_at = datetime.utcnow()
    
    def update_risk_metrics(
        self, 
        beta: Optional[float] = None,
        volatility: Optional[float] = None,
        sharpe_ratio: Optional[float] = None
    ) -> None:
        """Update risk metrics with validation."""
        if beta is not None:
            self.beta = beta
        if volatility is not None:
            if volatility < 0:
                raise ValueError("Volatility cannot be negative")
            self.volatility = volatility
        if sharpe_ratio is not None:
            self.sharpe_ratio = sharpe_ratio
        
        self.updated_at = datetime.utcnow()
    
    def is_liquid(self) -> bool:
        """Check if asset is liquid (has recent price data)."""
        if not self.last_price_update:
            return False
        
        # Consider liquid if price was updated in last 24 hours
        age = datetime.utcnow() - self.last_price_update
        return age.total_seconds() < 86400
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for persistence."""
        return {
            "id": str(self.id),
            "ticker": self.ticker,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "sector": self.sector.value,
            "currency": self.currency,
            "exchange": self.exchange,
            "market_cap": self.market_cap,
            "current_price": self.current_price,
            "last_price_update": self.last_price_update.isoformat() if self.last_price_update else None,
            "beta": self.beta,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Asset":
        """Create Asset from dictionary."""
        data = data.copy()
        data["id"] = UUID(data["id"]) if isinstance(data["id"], str) else data["id"]
        data["asset_type"] = AssetType(data["asset_type"])
        data["sector"] = AssetSector(data["sector"])
        
        # Parse datetime strings
        for field_name in ["created_at", "updated_at", "last_price_update"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)
    
    def __eq__(self, other) -> bool:
        """Assets are equal if they have the same ID."""
        if not isinstance(other, Asset):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        return f"Asset(ticker={self.ticker}, type={self.asset_type.value}, price={self.current_price})"
