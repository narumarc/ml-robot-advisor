"""Returns Value Object"""
import pandas as pd
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Returns:
    """Value Object pour les rendements"""
    
    expected: Dict[str, float]  # Rendements espérés
    historical: pd.DataFrame    # Rendements historiques
    
    def get_expected(self, ticker: str) -> float:
        return self.expected.get(ticker, 0.0)
    
    def get_mean_return(self) -> float:
        return sum(self.expected.values()) / len(self.expected)