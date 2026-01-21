
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """
    策略的抽象基礎類別
    """
    @property
    @abstractmethod
    def name(self):
        """策略的名稱"""
        pass

    @abstractmethod
    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        """
        執行策略分析

        :param hist: 包含股票歷史數據的 pandas DataFrame
        :param kwargs: 其他可能的參數 (例如大盤數據)
        :return: 如果股票符合策略，返回 True，否則返回 False
        """
        pass

    def calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """
        計算平均真實波幅 (ATR)

        :param hist: 歷史數據
        :param period: ATR 週期，默認 14
        :return: ATR 值
        """
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
