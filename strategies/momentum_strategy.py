
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """
    基本的「價漲量增」動能突破策略
    """
    @property
    def name(self):
        return "價漲量增"

    def __init__(self, volume_increase_ratio=3.0, price_increase_lower_bound=2.0, price_increase_upper_bound=4.0):
        self.volume_increase_ratio = volume_increase_ratio
        self.price_increase_lower_bound = price_increase_lower_bound
        self.price_increase_upper_bound = price_increase_upper_bound

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # 確保有足夠數據
        if hist.empty or len(hist) < 101:
            return False

        # --- 計算技術指標 ---
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA100'] = hist['Close'].rolling(window=100).mean()
        hist['Avg_Volume_20'] = hist['Volume'].rolling(window=20).mean().shift(1)

        latest_data = hist.iloc[-1]
        previous_data = hist.iloc[-2]
        
        if pd.isna(latest_data['MA50']) or pd.isna(latest_data['MA100']) or pd.isna(latest_data['Avg_Volume_20']) or latest_data['Avg_Volume_20'] == 0:
            return False

        # --- 開始進行條件篩選 ---
        
        # 條件一：中期趨勢向上
        is_trend_up = (latest_data['Close'] > latest_data['MA50']) or \
                      (latest_data['Close'] > latest_data['MA100'])
        
        if not is_trend_up:
            return False

        # 條件二：成交量顯著放大
        is_volume_surge = latest_data['Volume'] > (latest_data['Avg_Volume_20'] * self.volume_increase_ratio)

        if not is_volume_surge:
            return False

        # 條件三：價格同步上漲
        is_red_candle = latest_data['Close'] > latest_data['Open']
        price_change_percent = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100
        is_moderate_price_up = (self.price_increase_lower_bound < price_change_percent < self.price_increase_upper_bound)

        if not (is_red_candle and is_moderate_price_up):
            return False

        return True


class OptimizedMomentumStrategy(BaseStrategy):
    """
    優化版的動能突破策略 (主要用於港股)
    """
    @property
    def name(self):
        return "優化價漲量增(含相對強度)"

    def __init__(self, volume_increase_ratio=2.0, price_increase_lower_bound=2.0, price_increase_upper_bound=4.0, relative_strength_threshold=1.0):
        self.volume_increase_ratio = volume_increase_ratio
        self.price_increase_lower_bound = price_increase_lower_bound
        self.price_increase_upper_bound = price_increase_upper_bound
        self.relative_strength_threshold = relative_strength_threshold

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 數據品質過濾 ---
        if hist.empty or len(hist) < 101:
            return False
        hist = hist[hist['Volume'] > 0].copy()
        hist['Close'] = hist['Close'].ffill()
        if len(hist) < 101:
            return False

        latest_data = hist.iloc[-1]
        previous_data = hist.iloc[-2]

        # --- 過濾市調機制 ---
        if (latest_data['High'] / previous_data['Close'] - 1) * 100 > 9.5:
            return False

        # --- 計算指標 ---
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA100'] = hist['Close'].rolling(window=100).mean()
        hist['Avg_Volume_20'] = hist['Volume'].replace(0, np.nan).rolling(20).mean().shift(1)
        hist['HH10'] = hist['High'].rolling(10).max().shift(1)
        hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).cumsum()
        
        # After adding new columns, the iloc[-1] may not have them if they are added in place
        # It's safer to get the latest data again from the dataframe
        latest_data = hist.iloc[-1]

        if pd.isna(latest_data['Avg_Volume_20']) or latest_data['Avg_Volume_20'] == 0 or len(hist) < 20:
            return False
        
        if len(hist) < 5:
            return False
        obv_slope = np.polyfit(range(5), hist['OBV'].iloc[-5:], 1)[0]

        # --- 基礎條件篩選 ---
        is_trend_up = (latest_data['Close'] > latest_data['MA50']) or (latest_data['Close'] > latest_data['MA100'])
        if not is_trend_up: return False

        is_volume_surge = latest_data['Volume'] > (latest_data['Avg_Volume_20'] * self.volume_increase_ratio)
        if not is_volume_surge: return False

        is_red_candle = latest_data['Close'] > latest_data['Open']
        price_change_percent = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100
        is_moderate_price_up = (self.price_increase_lower_bound < price_change_percent < self.price_increase_upper_bound)
        if not (is_red_candle and is_moderate_price_up): return False

        # --- 優化條件篩選 ---
        market_return = kwargs.get('market_return', 0.0)
        alpha = price_change_percent - market_return
        if alpha < self.relative_strength_threshold: return False

        if pd.isna(latest_data['HH10']): return False
        is_breakout = latest_data['Close'] > latest_data['HH10']
        if not is_breakout: return False

        if obv_slope < 0: return False

        return True
