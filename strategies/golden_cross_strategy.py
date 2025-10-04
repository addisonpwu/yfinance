
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class GoldenCrossStrategy(BaseStrategy):
    """
    優化版黃金交叉策略
    結合大盤濾網、趨勢強度、成交量、RSI、乖離率、K線形態等多重過濾器
    """
    @property
    def name(self):
        return "優化版黃金交叉"

    def __init__(self,
                 short_window=50,
                 long_window=200,
                 rsi_period=14,
                 vol_avg_period=20,
                 ma_slope_period=10,
                 cross_lookback_period=3,
                 vol_multiplier=1.5,
                 rsi_threshold=70,
                 max_bias_ratio=0.15,
                 min_close_pos=0.5):
        # Periods
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.vol_avg_period = vol_avg_period
        self.ma_slope_period = ma_slope_period
        self.cross_lookback_period = cross_lookback_period
        # Thresholds
        self.vol_multiplier = vol_multiplier
        self.rsi_threshold = rsi_threshold
        self.max_bias_ratio = max_bias_ratio
        self.min_close_pos = min_close_pos

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """手動計算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 0. 大盤濾網 ---
        is_market_healthy = kwargs.get('is_market_healthy', True) # 默認為True以保持兼容性
        if not is_market_healthy:
            return False

        # --- 1. 基本數據長度檢查 ---
        if len(hist) < self.long_window + self.ma_slope_period:
            return False

        # --- 2. 計算所需指標 ---
        hist['MA50'] = hist['Close'].rolling(window=self.short_window).mean()
        hist['MA200'] = hist['Close'].rolling(window=self.long_window).mean()
        
        y = hist['MA200'].dropna()
        if len(y) < self.ma_slope_period: return False
        x = np.arange(len(y))
        ma200_slope = np.polyfit(x[-self.ma_slope_period:], y[-self.ma_slope_period:], 1)[0]

        hist['Vol_avg20'] = hist['Volume'].rolling(window=self.vol_avg_period).mean()
        hist['RSI14'] = self._calculate_rsi(hist['Close'], self.rsi_period)

        latest = hist.iloc[-1]
        
        required_cols = ['MA50', 'MA200', 'Vol_avg20', 'RSI14']
        if hist.iloc[-self.cross_lookback_period:][required_cols].isnull().values.any():
            return False

        # --- 3. 執行策略條件篩選 ---

        # 核心條件：目前為金叉狀態，且在N天內發生過交叉
        in_golden_cross_state = latest['MA50'] > latest['MA200']
        cross_occurred_recently = (hist['MA50'].iloc[-self.cross_lookback_period:] <= hist['MA200'].iloc[-self.cross_lookback_period:]).any()
        if not (in_golden_cross_state and cross_occurred_recently):
            return False

        # 閘門 1: 趨勢強度 (200MA本身要走平或向上)
        if ma200_slope <= 0:
            return False

        # 閘門 2: 成交量 (金叉當日要放量)
        if latest['Volume'] < latest['Vol_avg20'] * self.vol_multiplier:
            return False

        # 閘門 3: 短線不追買 (RSI)
        if latest['RSI14'] > self.rsi_threshold:
            return False
            
        # 閘門 4: 價格離 200MA 太遠不追 (乖離率)
        bias_ratio = abs(latest['Close'] / latest['MA200'] - 1)
        if bias_ratio > self.max_bias_ratio:
            return False
            
        # 閘門 5: 價格行為確認 (K線形態)
        daily_range = latest['High'] - latest['Low']
        if daily_range > 0:
            close_position_in_range = (latest['Close'] - latest['Low']) / daily_range
            if close_position_in_range < self.min_close_pos:
                return False
        # 如果 daily_range is 0, 只要是上漲就接受 (一字漲停)
        elif not (latest['Close'] > hist.iloc[-2]['Close']):
             return False

        # --- 所有條件均滿足 ---
        return True
