from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd
from src.core.models.entities import StockData

class StockRepository(ABC):
    @abstractmethod
    def get_historical_data(self, symbol: str, market: str, interval: str = '1d') -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_financial_info(self, symbol: str) -> Dict:
        pass
    
    @abstractmethod
    def get_news(self, symbol: str) -> List:
        pass
    
    @abstractmethod
    def save_historical_data(self, symbol: str, data: pd.DataFrame, market: str, interval: str = '1d') -> None:
        pass
    
    @abstractmethod
    def save_financial_info(self, symbol: str, info: Dict) -> None:
        pass