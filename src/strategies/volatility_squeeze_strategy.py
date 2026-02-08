from src.core.strategies.strategy import BaseStrategy
from src.core.models.entities import StrategyResult
import pandas as pd
import numpy as np

class VolatilitySqueezeStrategy(BaseStrategy):
    """
    波動率壓縮策略：僅實現布林帶壓縮、突破確認、量能配合邏輯
    """
    
    @property
    def name(self) -> str:
        return "波動率壓縮策略"
    
    @property
    def category(self) -> str:
        return "波動率策略"
    
    def execute(self, context):
        """
        只檢查波動率壓縮條件
        """
        hist = context.hist
        market_health = context.is_market_healthy
        
        if hist is None or len(hist) < 100:
            return StrategyResult(passed=False, details={"reason": "數據不足"})
        
        # 計算當前布林帶
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(hist, 20, 2)
        current_price = hist['Close'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        
        # 1. 擠壓識別：布林帶寬度 < 近100日最小10%分位
        squeeze_condition = self._check_squeeze(hist)
        
        # 2. 突破確認：收盤價 > 20日均線 且 當日漲幅 > 2%
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        price_above_ma = current_price > ma_20 if pd.notna(ma_20) else False
        daily_change = (current_price / hist['Close'].iloc[-2] - 1) if len(hist) >= 2 else 0
        breakout_confirmation = price_above_ma and daily_change > 0.02
        
        # 3. 量能配合：當日成交量 > 50日均量 × 1.5
        vol_50_avg = hist['Volume'].rolling(window=50).mean().iloc[-1]
        volume_support = current_volume > vol_50_avg * 1.5 if vol_50_avg > 0 else False
        
        # 市場條件：市場健康度
        market_condition_ok = market_health
        
        # 返回結果
        passed = squeeze_condition and breakout_confirmation and volume_support and market_condition_ok
        
        return StrategyResult(
            passed=passed,
            confidence=0.8 if passed else 0.1,
            details={
                "squeeze_condition": squeeze_condition,
                "breakout_confirmation": breakout_confirmation,
                "volume_support": volume_support,
                "market_condition_ok": market_condition_ok
            }
        )
    
    def calculate_bollinger_bands(self, hist: pd.DataFrame, period: int = 20, std_dev: int = 2):
        """計算布林帶"""
        sma = hist['Close'].rolling(window=period).mean()
        std = hist['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        middle_band = sma
        lower_band = sma - (std * std_dev)
        return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1]
    
    def _check_squeeze(self, hist: pd.DataFrame) -> bool:
        """檢查布林帶擠壓條件"""
        bb_width_history = []
        for i in range(min(100, len(hist)-20)):
            temp_hist = hist.iloc[-(i+21):-(i+1)] if i > 0 else hist.iloc[-21:]
            if len(temp_hist) >= 20:
                temp_sma = temp_hist['Close'].rolling(window=20).mean()
                temp_std = temp_hist['Close'].rolling(window=20).std()
                if not pd.isna(temp_sma.iloc[-1]) and not pd.isna(temp_std.iloc[-1]):
                    temp_bb_width = (temp_sma.iloc[-1] + 2 * temp_std.iloc[-1] - (temp_sma.iloc[-1] - 2 * temp_std.iloc[-1])) / temp_sma.iloc[-1]
                    bb_width_history.append(temp_bb_width)
        
        if len(bb_width_history) < 10:
            return False
        
        bb_width_percentile = np.percentile(bb_width_history, 10)
        current_bb = hist['Close'].rolling(window=20).mean()
        current_std = hist['Close'].rolling(window=20).std()
        if pd.isna(current_bb.iloc[-1]) or pd.isna(current_std.iloc[-1]):
            return False
        current_bb_width = (current_bb.iloc[-1] + 2 * current_std.iloc[-1] - (current_bb.iloc[-1] - 2 * current_std.iloc[-1])) / current_bb.iloc[-1]
        
        return current_bb_width < bb_width_percentile