"""Weights Value Object - Immutable"""
from typing import Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    """
    Value Object immuable représentant les poids d'un portefeuille.
    Frozen=True garantit l'immutabilité.
    """
    
    values: Dict[str, float]
    
    def __post_init__(self):
        # Validation
        total = sum(self.values.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def get(self, ticker: str) -> float:
        return self.values.get(ticker, 0.0)
    
    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)