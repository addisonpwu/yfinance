from src.core.strategies.strategy import BaseStrategy
from src.core.models.entities import StrategyResult
import pandas as pd
import numpy as np

class AccumulationAccelerationStrategy(BaseStrategy):
    """
    主力吸籌加速策略：僅實現橫盤幅度、量能趨勢、突破信號、RSI動態上穿邏輯
    """
    
    @property
    def name(self) -> str:
        return "主力吸籌加速策略"
    
    @property
    def category(self) -> str:
        return "吸籌策略"
    
    def execute(self, context):
        """
        只檢查主力吸籌加速條件（包含動態RSI上穿）
        """
        hist = context.hist
        
        if hist is None or len(hist) < 30:
            return StrategyResult(passed=False, details={"reason": "數據不足"})
        
        # 獲取數據
        current_price = hist['Close'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        
        # 1. 吸籌期：近30日價格波動幅度 < 15%
        accumulation_period = self._check_accumulation_period(hist)
        
        # 2. 量能趨勢：近30日成交量呈上升趨勢
        volume_trend = self._check_volume_trend(hist)
        
        # 3. 加速信號：當前價 > 橫盤上沿 × 1.015 且 量比 > 2.5
        acceleration_signal = self._check_acceleration_signal(hist, current_price, current_volume)
        
        # 4. 動態RSI條件：前一日RSI在40-60區間，當日RSI > 65
        rsi_momentum = self._check_dynamic_rsi_condition(hist)
        
        # 返回結果
        passed = accumulation_period and volume_trend and acceleration_signal and rsi_momentum
        
        return StrategyResult(
            passed=passed,
            confidence=0.8 if passed else 0.1,
            details={
                "accumulation_period": accumulation_period,
                "volume_trend": volume_trend,
                "acceleration_signal": acceleration_signal,
                "rsi_momentum": rsi_momentum
            }
        )
    
    def _check_accumulation_period(self, hist: pd.DataFrame) -> bool:
        """檢查吸籌期：近30日價格波動幅度 < 15%"""
        if len(hist) < 30:
            return False
        
        price_range_30d = hist['High'].iloc[-30:].max() - hist['Low'].iloc[-30:].min()
        avg_price_30d = hist['Close'].iloc[-30:].mean()
        volatility_30d = price_range_30d / avg_price_30d if avg_price_30d != 0 else float('inf')
        return volatility_30d < 0.15
    
    def _check_volume_trend(self, hist: pd.DataFrame) -> bool:
        """檢查量能趨勢：近30日成交量呈上升趨勢"""
        if len(hist) < 30:
            return False
        
        vol_early_30d = hist['Volume'].iloc[-30:-15].mean() if len(hist) >= 30 else 0
        vol_late_30d = hist['Volume'].iloc[-15:].mean() if len(hist) >= 15 else 0
        return vol_late_30d > vol_early_30d if vol_early_30d != 0 else True
    
    def _check_acceleration_signal(self, hist: pd.DataFrame, current_price: float, current_volume: float) -> bool:
        """檢查加速信號：當前價 > 橫盤上沿 × 1.015 且 量比 > 2.5"""
        if len(hist) < 30:
            return False
        
        range_high = hist['High'].iloc[-30:].max()
        vol_20_avg = hist['Volume'].rolling(window=20).mean().iloc[-1]
        vol_ratio = current_volume / vol_20_avg if vol_20_avg != 0 else 0
        
        return current_price > range_high * 1.015 and vol_ratio > 2.5
    
    def _check_dynamic_rsi_condition(self, hist: pd.DataFrame) -> bool:
        """檢查動態RSI條件：前一日RSI在40-60區間，當日RSI > 65"""
        if len(hist) < 2:
            return False
        
        # 計算前一日RSI
        hist_prev = hist.iloc[:-1]  # 除最後一日外的所有數據
        if len(hist_prev) < 15:  # 需要至少15天數據計算14日RSI
            return False
        
        rsi_prev = self.calculate_rsi(hist_prev, 14)
        rsi_current = self.calculate_rsi(hist, 14)
        
        # 檢查前一日RSI是否在40-60區間，當日RSI是否>65
        prev_in_range = 40 <= rsi_prev <= 60
        current_above_threshold = rsi_current > 65
        
        return prev_in_range and current_above_threshold
    
    def calculate_rsi(self, hist: pd.DataFrame, period: int = 14) -> float:
        """計算RSI指標"""
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]