import pandas as pd
import numpy as np
from src.core.strategies.strategy import BaseStrategy, StrategyContext, StrategyResult

class VCP_PocketPivotStrategy(BaseStrategy):
    """
    VCP + 口袋支點整合策略 (修正版)
    在VCP形態的末端，尋找口袋支點作為早期介入訊號。
    """
    @property
    def name(self):
        return "VCP口袋支點"
    
    @property
    def category(self):
        return "技术分析"

    def __init__(self,
                 # VCP Params
                 ma_periods=[50, 150, 200],
                 volatility_windows=[50, 20, 10],
                 volume_avg_period=50,
                 max_final_volatility=0.025,
                 min_dist_from_52w_high=0.85,
                 # Pocket Pivot Params
                 pp_lookback_period=10,
                 pp_vol_multiplier=1.2, # 提高到 1.2 倍
                 pp_max_bias_ratio=0.08):
        
        self.ma_periods = sorted(ma_periods, reverse=True)
        self.volatility_windows = sorted(volatility_windows, reverse=True)
        self.volume_avg_period = volume_avg_period
        self.max_final_volatility = max_final_volatility
        self.min_dist_from_52w_high = min_dist_from_52w_high
        self.pp_lookback_period = pp_lookback_period
        self.pp_vol_multiplier = pp_vol_multiplier
        self.pp_max_bias_ratio = pp_max_bias_ratio

    def execute(self, context: StrategyContext) -> StrategyResult:
        hist = context.hist
        info = context.info
        market_return = context.market_return
        
        # --- 1. 基本數據長度檢查 ---
        if len(hist) < 252: # 需要52週數據
            return StrategyResult(passed=False)

        # --- 2. 計算所需指標 ---
        for period in self.ma_periods:
            hist[f'MA{period}'] = hist['Close'].rolling(window=period).mean()
        hist['MA200_slope'] = hist['MA200'].diff(5)

        hist['pct_change'] = hist['Close'].pct_change(fill_method=None)
        for window in self.volatility_windows:
            hist[f'volatility_{window}'] = hist['pct_change'].rolling(window=window).std()

        hist['Avg_Vol_50'] = hist['Volume'].rolling(window=self.volume_avg_period).mean()
        hist['Avg_Vol_10'] = hist['Volume'].rolling(window=10).mean()
        hist['52w_high'] = hist['Close'].rolling(window=252).max()

        # 新增：價格振幅檢測
        hist['price_range'] = (hist['High'] - hist['Low']) / hist['Close']
        hist['price_range_ma'] = hist['price_range'].rolling(20).mean()

        # --- 3. 定義數據點並檢查有效性 ---
        latest = hist.iloc[-1]
        previous = hist.iloc[-2]

        # 獲取市場環境
        if market_return > 2:  # 大盤強勢
            current_min_dist = 0.90
        else:
            current_min_dist = self.min_dist_from_52w_high
        
        required_cols = [f'MA{p}' for p in self.ma_periods] + ['MA200_slope'] + \
                        [f'volatility_{w}' for w in self.volatility_windows] + \
                        ['Avg_Vol_50', 'Avg_Vol_10', '52w_high']
        # 檢查最近兩筆數據，確保 previous 的指標也有效
        if hist.iloc[-2:][required_cols].isnull().values.any():
            return StrategyResult(passed=False)

        # --- 4. VCP 背景條件篩選 ---
        # 閘門 0: 接近52週高點 (根據市場環境動態調整)
        if latest['Close'] < (latest['52w_high'] * current_min_dist):
            return StrategyResult(passed=False)

        # 閘門 0.5: 價格振幅也應收縮
        if latest['price_range'] > latest['price_range_ma']:
            return StrategyResult(passed=False)

        # 閘門 1: 強勢趨勢 (均線多頭排列 + 長期均線向上)
        ma50, ma150, ma200 = latest['MA50'], latest['MA150'], latest['MA200']
        is_ma_aligned = (latest['Close'] > ma50 > ma150 > ma200)
        is_long_trend_up = latest['MA200_slope'] > 0
        if not (is_ma_aligned and is_long_trend_up):
             return StrategyResult(passed=False)

        # 閘門 2: 波動收縮
        vola_values = [latest[f'volatility_{w}'] for w in self.volatility_windows]
        is_vola_contracting = (vola_values[2] < vola_values[1] < vola_values[0])
        is_vola_low = vola_values[2] < self.max_final_volatility
        if not (is_vola_contracting and is_vola_low):
            return StrategyResult(passed=False)

        # 閘門 3: 量能萎縮 (修正：檢查前一天的狀態)
        if previous['Avg_Vol_10'] >= previous['Avg_Vol_50']:
            return StrategyResult(passed=False)

        # --- 5. 口袋支點觸發條件 ---
        # 閘門 5.1: 口袋支點成交量訊號 (壓倒近期賣壓 + 絕對強度)
        lookback_data = hist.iloc[-self.pp_lookback_period-1:-1]
        down_days_volume = lookback_data[lookback_data['Close'] < lookback_data['Open']]['Volume']

        volume_check_passed = False
        if down_days_volume.empty: # 如果近期沒有下跌日，本身就是極強勢
            volume_check_passed = True
        else:
            max_down_volume = down_days_volume.max()
            if latest['Volume'] > max_down_volume * 1.2:  # 要求壓倒近期最大下跌日成交量1.2倍
                volume_check_passed = True

        if not (volume_check_passed and latest['Volume'] > latest['Avg_Vol_50'] * self.pp_vol_multiplier):
            return StrategyResult(passed=False)

        # 閘門 5.2: 價格行為 (紅K + 獲取支撐)
        is_green_candle = latest['Close'] > latest['Open']
        is_supported = latest['Low'] > ma50 # 最低價高於50MA
        if not (is_green_candle and is_supported):
            return StrategyResult(passed=False)

        # 閘門 5.3: 避免過度乖離
        bias_ratio = (latest['Close'] - ma50) / ma50
        if bias_ratio > self.pp_max_bias_ratio:
            return StrategyResult(passed=False)

        # 閘門 5.4: 支點確認機制（如果數據足夠）
        if len(hist) > 3:
            pivot_high = hist.iloc[-3]['High']
            confirmation_close = hist.iloc[-2]['Close']
            if confirmation_close < pivot_high * 0.98:  # 允許2%的回撤
                return StrategyResult(passed=False)

        # --- 所有條件均滿足 ---
        return StrategyResult(passed=True, confidence=0.9)