
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class BollingerSqueezeStrategy(BaseStrategy):
    """
    布林帶擠壓突破策略 (Bollinger Band Squeeze Breakout)
    旨在捕捉股價從低波動盤整期，向上突破進入趨勢行情（單邊行情）的瞬間。
    """
    @property
    def name(self):
        return "布林帶擠壓突破"

    def __init__(self, 
                 bb_period=20, 
                 bb_std_dev=2, 
                 squeeze_lookback=120, 
                 squeeze_percentile=0.10, # 擠壓程度定義為帶寬處於過去N天的最低10%
                 prolonged_squeeze_period=5, # 擠壓需要持續的天數
                 long_trend_period=200,
                 ma_slope_period=5,
                 volume_period=50,
                 volume_multiplier=1.5):
        # Bollinger Band settings
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        # Squeeze settings
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.prolonged_squeeze_period = prolonged_squeeze_period
        # Filters
        self.long_trend_period = long_trend_period
        self.ma_slope_period = ma_slope_period
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 1. 基本數據長度檢查 ---
        if len(hist) < max(self.squeeze_lookback, self.long_trend_period):
            return False

        # --- 2. 計算所需指標 ---
        hist['SMA'] = hist['Close'].rolling(window=self.bb_period).mean()
        hist['StdDev'] = hist['Close'].rolling(window=self.bb_period).std()
        hist['UpperBand'] = hist['SMA'] + (hist['StdDev'] * self.bb_std_dev)
        hist['LowerBand'] = hist['SMA'] - (hist['StdDev'] * self.bb_std_dev)
        hist['BandWidth'] = (hist['UpperBand'] - hist['LowerBand']) / hist['SMA']
        
        hist['SMA200'] = hist['Close'].rolling(window=self.long_trend_period).mean()
        hist['SMA200_slope'] = hist['SMA200'].diff(self.ma_slope_period)
        hist['Avg_Volume'] = hist['Volume'].rolling(window=self.volume_period).mean()

        # --- 3. 識別數據點並檢查有效性 ---
        latest = hist.iloc[-1]

        required_cols = ['UpperBand', 'BandWidth', 'SMA200', 'SMA200_slope', 'Avg_Volume']
        if hist.iloc[-self.prolonged_squeeze_period-1:][required_cols].isnull().values.any():
            return False

        # --- 4. 執行策略條件篩選 (根據專家建議精緻化) ---

        # 閘門 1: 識別持續的擠壓狀態
        squeeze_threshold = hist['BandWidth'].rolling(window=self.squeeze_lookback).quantile(self.squeeze_percentile).iloc[-1]
        is_in_prolonged_squeeze = (hist['BandWidth'].iloc[-self.prolonged_squeeze_period-1:-1] <= squeeze_threshold).all()
        if not is_in_prolonged_squeeze:
            return False

        # 閘門 1.5: 價格振幅也應收縮
        hist['price_range'] = (hist['High'] - hist['Low']) / hist['Close']
        hist['price_range_ma'] = hist['price_range'].rolling(20).mean()
        is_price_range_contracting = (hist['price_range'].iloc[-self.prolonged_squeeze_period-1:-1] < hist['price_range_ma'].iloc[-self.prolonged_squeeze_period-1:-1].mean()).all()
        if not is_price_range_contracting:
            return False

        # 閘門 2: 長期趨勢過濾 (趨勢向上且價格在趨勢之上)
        is_long_trend_healthy = (latest['Close'] > latest['SMA200']) and (latest['SMA200_slope'] >= 0)
        if not is_long_trend_healthy:
            return False

        # 閘門 3: 向上突破的品質 (核心觸發條件) - 增加突破力度檢測
        is_breakout = latest['Close'] > latest['UpperBand']
        is_strong_candle = latest['Close'] > latest['Open'] # 要求是紅K
        breakout_strength = (latest['Close'] - latest['UpperBand']) / latest['UpperBand'] if latest['UpperBand'] > 0 else 0
        if not (is_breakout and is_strong_candle and breakout_strength >= 0.01):
            return False

        # 閘門 4: 成交量確認（連續2天放量）
        if latest['Volume'] < latest['Avg_Volume'] * self.volume_multiplier:
            return False
        if hist['Volume'].iloc[-2] < hist['Avg_Volume'].iloc[-2] * self.volume_multiplier:
            return False

        # 閘門 5: 假突破過濾（檢查突破後是否快速回落）
        close_to_band = (latest['UpperBand'] - latest['Close']) / latest['UpperBand'] if latest['UpperBand'] > 0 else 1
        if close_to_band < 0.005:  # 收盤價太接近上軌
            return False

        # --- 所有條件均滿足 ---
        return True
