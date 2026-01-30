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
                 obv_slope_period=10,
                 obv_ma_period=10,
                 # New BB Squeeze params
                 bb_period=20,
                 bb_squeeze_lookback=120,
                 bb_squeeze_percentile=0.1,
                 # Volume params
                 vol_ma_period=60,
                 vol_shrink_lookback=10,
                 vol_lower_multiplier=1.1,
                 vol_upper_multiplier=1.8,
                 # Price behavior params
                 max_ma_bias_ratio=0.10, # Tightened from 0.20
                 min_close_pos=0.5):
        # Periods
        self.long_ma_period = long_ma_period
        self.short_ma_period = short_ma_period
        self.obv_slope_period = obv_slope_period
        self.obv_ma_period = obv_ma_period
        self.bb_period = bb_period
        self.bb_squeeze_lookback = bb_squeeze_lookback
        self.vol_ma_period = vol_ma_period
        self.vol_shrink_lookback = vol_shrink_lookback
        # Thresholds
        self.bb_squeeze_percentile = bb_squeeze_percentile
        self.vol_lower_multiplier = vol_lower_multiplier
        self.vol_upper_multiplier = vol_upper_multiplier
        self.max_ma_bias_ratio = max_ma_bias_ratio
        self.min_close_pos = min_close_pos


    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 1. 基本數據長度檢查 ---
        if len(hist) < max(self.long_ma_period, self.bb_squeeze_lookback):
            return False

        # --- 2. 計算所需指標 ---
        # 趨勢指標
        hist['MA100'] = hist['Close'].rolling(window=self.long_ma_period).mean()
        hist['MA20'] = hist['Close'].rolling(window=self.short_ma_period).mean()
        
        # 資金流指標
        hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()
        hist['OBV_MA10'] = hist['OBV'].rolling(window=self.obv_ma_period).mean()
        
        # 波動率指標
        hist['SMA_BB'] = hist['Close'].rolling(window=self.bb_period).mean()
        hist['StdDev'] = hist['Close'].rolling(window=self.bb_period).std()
        hist['UpperBand'] = hist['SMA_BB'] + (hist['StdDev'] * 2)
        hist['LowerBand'] = hist['SMA_BB'] - (hist['StdDev'] * 2)
        hist['BandWidth'] = (hist['UpperBand'] - hist['LowerBand']) / hist['SMA_BB']

        # 成交量指標
        hist['Avg_Vol_60'] = hist['Volume'].rolling(window=self.vol_ma_period).mean()
        
        # --- 3. 識別數據點並檢查有效性 ---
        latest = hist.iloc[-1]
        previous = hist.iloc[-2]

        required_cols = ['MA100', 'MA20', 'OBV_MA10', 'BandWidth', 'Avg_Vol_60']
        if hist.iloc[-self.vol_shrink_lookback-1:][required_cols].isnull().values.any():
            return False

        # --- 4. 執行策略條件篩選 (專注於"吸籌"特徵) ---

        # 閘門 0: 波動率必須處於收縮狀態 (檢查前一天)
        squeeze_threshold = hist['BandWidth'].rolling(window=self.bb_squeeze_lookback).quantile(self.bb_squeeze_percentile).iloc[-1]
        if previous['BandWidth'] > squeeze_threshold:
            return False

        # 閘門 1: 長期與短期趨勢健康
        if not (latest['Close'] > latest['MA100'] and latest['Close'] > latest['MA20']):
            return False
        if abs(latest['Close'] / latest['MA100'] - 1) > self.max_ma_bias_ratio:
            return False

        # 閘門 2: 資金持續流入 (OBV) - 使用加權回歸
        y = hist['OBV'].dropna()
        if len(y) < self.obv_slope_period: return False
        x = np.arange(len(y))
        # 使用加權回歸，近期權重更高
        weights = np.exp(np.linspace(-1, 0, len(y[-self.obv_slope_period:])))
        obv_slope = np.polyfit(x[-self.obv_slope_period:], y[-self.obv_slope_period:], 1, w=weights)[0]
        if obv_slope <= 0 or latest['OBV'] < latest['OBV_MA10']:
            return False

        # 閘門 2.5: 檢測吸籌結束信號（OBV加速度）
        hist['OBV_acceleration'] = hist['OBV'].diff().diff()
        if hist['OBV_acceleration'].iloc[-1] < 0:  # OBV加速度轉負
            return False

        # 閘門 2.6: 筹码锁定检测（放量不涨）
        hist['price_change_per_volume'] = hist['Close'].pct_change() / hist['Volume']
        if hist['price_change_per_volume'].iloc[-5:].mean() < 0:
            return False  # 放量不涨，可能主力出貨

        # 閘門 3: 成交量呈現「萎縮後溫和放量」 - 根據波動率動態調整
        was_volume_down = (hist['Volume'].iloc[-self.vol_shrink_lookback:-1] < hist['Avg_Vol_60'].iloc[-self.vol_shrink_lookback:-1]).any()

        # 根據波動率動態調整成交量放大上限
        volatility_ratio = previous['BandWidth'] / squeeze_threshold
        current_vol_upper_multiplier = self.vol_upper_multiplier
        if volatility_ratio < 0.8:  # 波動率仍很低
            current_vol_upper_multiplier = 2.0  # 允許更大的放量

        is_volume_gentle_up = (latest['Avg_Vol_60'] * self.vol_lower_multiplier < latest['Volume'] < latest['Avg_Vol_60'] * current_vol_upper_multiplier)
        if not (was_volume_down and is_volume_gentle_up):
            return False

        # 閘門 4: 當日買盤強勁 (收在K線上半部)
        daily_range = latest['High'] - latest['Low']
        if daily_range > 0:
            close_position_in_range = (latest['Close'] - latest['Low']) / daily_range
            if close_position_in_range < self.min_close_pos:
                return False
        elif not (latest['Close'] > previous['Close']):
             return False # 針對一字板情況

        # --- 所有條件均滿足 ---
        return True