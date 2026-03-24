"""
Portfolio Entity - Core Business Object
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal


@dataclass
class Portfolio:
    """
    Portfolio entity représentant un portefeuille d'investissement.
    
    C'est l'objet métier central qui contient:
    - Les actifs et leurs poids
    - Les métriques de performance
    - L'historique des rebalancements
    """
    
    portfolio_id: str
    name: str
    tickers: List[str]
    weights: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    
    # Optional metadata
    strategy: Optional[str] = None
    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Validation après initialisation"""
        self._validate_weights()
    
    def _validate_weights(self):
        """Valide que les poids sont cohérents"""
        if not self.weights:
            return
        
        # Vérifier que tous les tickers ont un poids
        missing_tickers = set(self.tickers) - set(self.weights.keys())
        if missing_tickers:
            raise ValueError(f"Missing weights for tickers: {missing_tickers}")
        
        # Vérifier que les poids somment à 1 (±0.01 pour tolérance)
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Vérifier que tous les poids sont positifs
        negative_weights = {k: v for k, v in self.weights.items() if v < 0}
        if negative_weights:
            raise ValueError(f"Negative weights not allowed: {negative_weights}")
    
    def get_weight(self, ticker: str) -> float:
        """Retourne le poids d'un actif"""
        return self.weights.get(ticker, 0.0)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Met à jour les poids du portefeuille"""
        self.weights = new_weights
        self.updated_at = datetime.now()
        self._validate_weights()
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour sérialisation"""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'tickers': self.tickers,
            'weights': self.weights,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'strategy': self.strategy,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Portfolio':
        """Crée un Portfolio depuis un dictionnaire"""
        return cls(
            portfolio_id=data['portfolio_id'],
            name=data['name'],
            tickers=data['tickers'],
            weights=data['weights'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            strategy=data.get('strategy'),
            expected_return=data.get('expected_return'),
            volatility=data.get('volatility'),
            sharpe_ratio=data.get('sharpe_ratio')
        )


@dataclass
class PortfolioSnapshot:
    """
    Snapshot d'un portefeuille à un moment donné.
    Utilisé pour l'historique et le backtesting.
    """
    
    portfolio_id: str
    timestamp: datetime
    weights: Dict[str, float]
    value: float
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'portfolio_id': self.portfolio_id,
            'timestamp': self.timestamp.isoformat(),
            'weights': self.weights,
            'value': self.value,
            'metrics': self.metrics
        }
