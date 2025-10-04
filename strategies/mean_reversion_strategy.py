
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    優化版均值回歸策略
    結合布林通道、趨勢過濾、止穩訊號、威廉指標與成交量分析
    """
    @property
    def name(self):
        return "優化版均值回歸"

    def __init__(self,
                 period=20,
                 std_dev=2,
                 rsv_period=14,
                 long_ma_period=200,
                 slope_period=5,
                 vol_avg_period=20,
                 willr_lower=-95,
                 willr_upper=-80,
                 vol_shrink_ratio=0.7):
        # Bollinger Band settings
        self.period = period
        self.std_dev = std_dev
        # Williams %R settings
        self.rsv_period = rsv_period
        self.willr_lower = willr_lower
        self.willr_upper = willr_upper
        # Trend and Volume settings
        self.long_ma_period = long_ma_period
        self.slope_period = slope_period
        self.vol_avg_period = vol_avg_period
        self.vol_shrink_ratio = vol_shrink_ratio

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 1. 基本數據長度檢查 ---
        if len(hist) < self.long_ma_period:
            return False

        # --- 2. 計算所需指標 ---
        hist['SMA20'] = hist['Close'].rolling(window=self.period).mean()
        hist['StdDev'] = hist['Close'].rolling(window=self.period).std()
        hist['LowerBand'] = hist['SMA20'] - (hist['StdDev'] * self.std_dev)
        
        hist['SMA200'] = hist['Close'].rolling(window=self.long_ma_period).mean()
        
        y = hist['SMA20'].dropna()
        if len(y) < self.slope_period: return False
        x = np.arange(len(y))
        slope = np.polyfit(x[-self.slope_period:], y[-self.slope_period:], 1)[0]

        low_min = hist['Low'].rolling(self.rsv_period).min()
        high_max = hist['High'].rolling(self.rsv_period).max()
        will_r = -100 * (high_max - hist['Close']) / (high_max - low_min)
        hist['WillR'] = will_r

        hist['Avg_Volume_20'] = hist['Volume'].rolling(window=self.vol_avg_period).mean()

        latest = hist.iloc[-1]
        previous = hist.iloc[-2]

        required_cols = ['SMA20', 'LowerBand', 'SMA200', 'WillR', 'Avg_Volume_20']
        if hist.iloc[-4:][required_cols].isnull().values.any():
            return False

        # --- 3. 執行策略條件篩選 ---

        # 閘門 1: 過濾「空頭加速」階段
        if latest['Close'] < latest['SMA200'] and slope < 0:
            return False

        # 閘門 2: 確認「止跌」而不是「繼續破」 (增加紅K確認)
        is_below_band = latest['Close'] < latest['LowerBand']
        is_higher_low = latest['Low'] > previous['Low']
        is_green_candle = latest['Close'] > latest['Open']
        if not (is_below_band and is_higher_low and is_green_candle):
            return False

        # 閘門 3: 價格「太遠」不追 (使用威廉指標)
        if not (self.willr_lower < latest['WillR'] < self.willr_upper):
            return False

        # 閘門 4: 量縮→量增 確認「最後一跌」
        is_volume_up = latest['Volume'] > latest['Avg_Volume_20']
        was_volume_down = (hist['Volume'].iloc[-4:-1] < self.vol_shrink_ratio * hist['Avg_Volume_20'].iloc[-4:-1]).any()
        if not (is_volume_up and was_volume_down):
            return False

        # --- 所有條件均滿足 ---
        return True
