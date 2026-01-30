
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class InsideDayStrategy(BaseStrategy):
    """
    內部日反轉策略 (Inside Day / Harami Reversal)
    在下跌趨勢的末端，尋找一個內部日（Harami）形態，並在次日出現向上突破時確認信號。
    這是一個捕捉趨勢轉折點的右側交易策略。
    """
    @property
    def name(self):
        return "內部日反轉"

    def __init__(self,
                 rsi_period=14,
                 rsi_oversold_threshold=35,
                 trend_period=20,
                 trend_slope_period=5,
                 long_trend_period=200):
        # Periods
        self.rsi_period = rsi_period
        self.trend_period = trend_period
        self.trend_slope_period = trend_slope_period
        self.long_trend_period = long_trend_period
        # Thresholds
        self.rsi_oversold_threshold = rsi_oversold_threshold

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """計算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 1. 基本數據長度檢查 ---
        if len(hist) < self.long_trend_period:
            return False

        # --- 2. 計算所需指標 -- -
        hist['SMA20'] = hist['Close'].rolling(window=self.trend_period).mean()
        hist['SMA200'] = hist['Close'].rolling(window=self.long_trend_period).mean()
        hist['RSI14'] = self._calculate_rsi(hist['Close'], self.rsi_period)

        # --- 3. 識別K線與檢查指標有效性 ---
        confirmation_day = hist.iloc[-1]
        inside_day = hist.iloc[-2]
        engulfing_day = hist.iloc[-3]

        required_cols = ['SMA20', 'SMA200', 'RSI14']
        if hist.iloc[-3:][required_cols].isnull().values.any():
            return False
            
        # --- 4. 執行策略條件篩選 (根據專家建議精緻化) ---

        # 閘門 1: 識別內部日形態（實體也在內部）
        is_inside_day = (inside_day['High'] < engulfing_day['High']) and \
                        (inside_day['Low'] > engulfing_day['Low'])
        is_body_inside = (
            inside_day['Close'] < engulfing_day['Close'] and
            inside_day['Open'] > engulfing_day['Open']
        )
        if not (is_inside_day and is_body_inside):
            return False

        # 新增閘門 1.5: 母線必須為黑K (經典Harami定義)
        if engulfing_day['Close'] >= engulfing_day['Open']:
            return False

        # 閘門 1.6: 內部日影線長度過濾
        inside_body_size = abs(inside_day['Close'] - inside_day['Open'])
        inside_upper_shadow = inside_day['High'] - inside_day[['Open', 'Close']].max()
        inside_lower_shadow = inside_day[['Open', 'Close']].min() - inside_day['Low']
        if inside_upper_shadow > inside_body_size * 2 or inside_lower_shadow > inside_body_size * 2:
            return False

        # 閘門 2: 必須發生在短期下跌趨勢中
        y = hist['SMA20'].dropna()
        if len(y) < self.trend_slope_period * 2: return False
        x = np.arange(len(y))
        trend_slope = np.polyfit(x[-(self.trend_slope_period+3):-3], y[-(self.trend_slope_period+3):-3], 1)[0]
        if trend_slope >= 0:
            return False

        # 新增閘門 2.5: 但必須處於長期上升趨勢中 (牛市回檔)
        if inside_day['Close'] < inside_day['SMA200']:
            return False

        # 閘門 3: 內部日成交量萎縮
        if inside_day['Volume'] >= engulfing_day['Volume']:
            return False

        # 閘門 4 (優化): 在賣壓高潮(母線)時處於RSI超賣區
        if engulfing_day['RSI14'] > self.rsi_oversold_threshold:
            return False

        # 閘門 5: 突破確認 (核心觸發條件) - 增加突破力度檢測
        is_breakout_confirmed = confirmation_day['Close'] > inside_day['High']
        inside_day_range = inside_day['High'] - inside_day['Low']
        breakout_magnitude = (confirmation_day['Close'] - inside_day['High']) / inside_day_range if inside_day_range > 0 else 0
        if not (is_breakout_confirmed and breakout_magnitude >= 0.5):
            return False

        # 閘門 6: 突破時成交量放大
        if confirmation_day['Volume'] <= inside_day['Volume']:
            return False

        # 閘門 7: 假突破過濾（檢查確認日是否留下長上影線）
        confirmation_body = abs(confirmation_day['Close'] - confirmation_day['Open'])
        confirmation_upper_shadow = confirmation_day['High'] - confirmation_day[['Open', 'Close']].max()
        if confirmation_upper_shadow > confirmation_body:
            return False

        # --- 所有條件均滿足 ---
        return True
