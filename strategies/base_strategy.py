
from abc import ABC, abstractmethod
import pandas as pd

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
