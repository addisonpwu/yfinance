from src.core.strategies.strategy import BaseStrategy
from src.core.models.entities import StrategyResult
import pandas as pd
import numpy as np

class MomentumBreakoutStrategy(BaseStrategy):
    """
    動量爆發策略：僅實現價格突破、量能爆發、動量強度邏輯
    """
    
    @property
    def name(self) -> str:
        return "動量爆發策略"
    
    @property
    def category(self) -> str:
        return "動量策略"
    
    def execute(self, context):
        """
        只檢查動量爆發條件
        """
        hist = context.hist
        
        if hist is None or len(hist) < 21:  # 需要至少21天數據來計算20日最高價
            return StrategyResult(passed=False, details={"reason": "數據不足"})
            
        # 獲取數據
        current_price = hist['Close'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        
        # 1. 價格突破：當前收盤價 > 近20日最高價 × 1.01
        high_20 = hist['High'].rolling(window=20).max().iloc[-1]
        price_breakout = current_price > high_20 * 1.01
        
        # 2. 量能爆發：當日成交量 > 近20日均量 × 2.0
        vol_20_avg = hist['Volume'].rolling(window=20).mean().iloc[-1]
        volume_burst = current_volume > vol_20_avg * 2.0 if vol_20_avg > 0 else False
        
        # 3. 動量強度：5日漲幅 > 3% 且 20日漲幅 > 5%
        if len(hist) >= 6:
            price_change_5d = (current_price / hist['Close'].iloc[-6] - 1)
        else:
            price_change_5d = 0
            
        if len(hist) >= 21:
            price_change_20d = (current_price / hist['Close'].iloc[-21] - 1)
        else:
            price_change_20d = 0
            
        momentum_strength = price_change_5d > 0.03 and price_change_20d > 0.05
        
        # 返回結果
        passed = price_breakout and volume_burst and momentum_strength
        
        return StrategyResult(
            passed=passed,
            confidence=0.8 if passed else 0.1,
            details={
                "price_breakout": price_breakout,
                "volume_burst": volume_burst,
                "momentum_strength": momentum_strength
            }
        )