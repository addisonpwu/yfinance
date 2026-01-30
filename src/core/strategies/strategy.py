from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from src.core.models.entities import StrategyResult

class StrategyContext:
    def __init__(self, 
                 hist: pd.DataFrame, 
                 info: Dict, 
                 market_return: float = 0.0, 
                 is_market_healthy: bool = True,
                 additional_data: Optional[Dict] = None):
        self.hist = hist
        self.info = info
        self.market_return = market_return
        self.is_market_healthy = is_market_healthy
        self.additional_data = additional_data or {}

class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, context: StrategyContext) -> StrategyResult:
        pass
    
    def calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """
        計算平均真實波幅 (ATR)

        :param hist: 歷史數據
        :param period: ATR 週期，默認 14
        :return: ATR 值
        """
        import numpy as np
        high_low = hist['High'] - hist['Low']
        high_close = np.abs(hist['High'] - hist['Close'].shift())
        low_close = np.abs(hist['Low'] - hist['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]

    def calculate_stop_loss(self, hist: pd.DataFrame, entry_price: float, atr_multiplier: float = 2.0) -> float:
        """
        計算基於 ATR 的止損價格

        :param hist: 歷史數據
        :param entry_price: 入場價格
        :param atr_multiplier: ATR 倍數，默認 2.0
        :return: 止損價格
        """
        atr = self.calculate_atr(hist)
        return entry_price - (atr * atr_multiplier)

class StrategyEngine:
    def __init__(self, strategies: list):
        self.strategies = strategies
    
    def execute_all(self, context: StrategyContext) -> list:
        results = []
        for strategy in self.strategies:
            try:
                result = strategy.execute(context)
                results.append(result)
            except Exception as e:
                print(f"执行策略 {strategy.name} 时出错: {e}")
                results.append(StrategyResult(passed=False, details={"error": str(e)}))
        return results