import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MainForceAccumulationStrategy(BaseStrategy):
    """
    優化版主力吸籌策略
    結合多重過濾器，尋找趨勢健康、資金流入、波動受控且買盤強勁的股票。
    """
    @property
    def name(self):
        return "優化版主力吸籌"

    def __init__(self,
                 long_ma_period=100,
                 short_ma_period=20,
                 vol_ma_period=60,
                 obv_slope_period=10,
                 obv_ma_period=10,
                 atr_period=14,
                 vol_multiplier=2.0,
                 max_atr_ratio=0.05,
                 min_close_pos=0.5,
                 max_ma_bias_ratio=0.20):
        # Periods
        self.long_ma_period = long_ma_period
        self.short_ma_period = short_ma_period
        self.vol_ma_period = vol_ma_period
        self.obv_slope_period = obv_slope_period
        self.obv_ma_period = obv_ma_period
        self.atr_period = atr_period
        # Thresholds
        self.vol_multiplier = vol_multiplier
        self.max_atr_ratio = max_atr_ratio
        self.min_close_pos = min_close_pos
        self.max_ma_bias_ratio = max_ma_bias_ratio

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """更簡潔且使用span的ATR計算"""
        prev_close = close.shift(1)
        tr_df = pd.DataFrame({
            'tr1': high - low,
            'tr2': abs(high - prev_close),
            'tr3': abs(low - prev_close)
        })
        tr = tr_df.max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 1. 基本數據長度檢查 ---
        if len(hist) < self.long_ma_period:
            return False

        # --- 2. 計算所需指標 ---
        hist['MA100'] = hist['Close'].rolling(window=self.long_ma_period).mean()
        hist['MA20'] = hist['Close'].rolling(window=self.short_ma_period).mean()
        hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()
        hist['OBV_MA10'] = hist['OBV'].rolling(window=self.obv_ma_period).mean()
        hist['Avg_Vol_60'] = hist['Volume'].rolling(window=self.vol_ma_period).mean()
        hist['ATR14'] = self._calculate_atr(hist['High'], hist['Low'], hist['Close'], self.atr_period)
        
        y = hist['OBV'].dropna()
        if len(y) < self.obv_slope_period: return False
        x = np.arange(len(y))
        obv_slope = np.polyfit(x[-self.obv_slope_period:], y[-self.obv_slope_period:], 1)[0]

        latest = hist.iloc[-1]
        previous = hist.iloc[-2]

        required_cols = ['MA100', 'MA20', 'OBV_MA10', 'Avg_Vol_60', 'ATR14']
        if hist.iloc[-1:][required_cols].isnull().values.any():
            return False

        # --- 3. 執行策略條件篩選 ---

        # 閘門 1.1: 長期趨勢 (股價在100MA之上)
        if latest['Close'] < latest['MA100']:
            return False
        
        # 閘門 1.2: 長期趨勢 (乖離率不能過大)
        if abs(latest['Close'] / latest['MA100'] - 1) > self.max_ma_bias_ratio:
            return False

        # 閘門 1.3: 短期趨勢 (股價在20MA之上)
        if latest['Close'] < latest['MA20']:
            return False

        # 閘門 2.1: OBV趨勢向上 (斜率)
        if obv_slope <= 0:
            return False
            
        # 閘門 2.2: OBV位於均線之上 (雙重確認)
        if latest['OBV'] < latest['OBV_MA10']:
            return False

        # 閘門 3: 近期成交量放大
        if latest['Volume'] < latest['Avg_Vol_60'] * self.vol_multiplier:
            return False

        # 閘門 4: 價格波動受控
        atr_ratio = latest['ATR14'] / latest['Close']
        if atr_ratio > self.max_atr_ratio:
            return False

        # 閘門 5: 當日買盤強勁
        daily_range = latest['High'] - latest['Low']
        if daily_range == 0:
            if not (latest['Close'] > previous['Close']):
                return False # 如果是一字跌停或平盤，則不滿足
        else:
            close_position_in_range = (latest['Close'] - latest['Low']) / daily_range
            if close_position_in_range < self.min_close_pos:
                return False

        # --- 所有條件均滿足 ---
        return True