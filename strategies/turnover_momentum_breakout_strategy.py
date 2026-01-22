import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class TurnoverMomentumBreakoutStrategy(BaseStrategy):
    """
    換手率動量突破策略
    尋找換手率近期有顯著放大，且股價處於短期上升趨勢的股票。
    """
    @property
    def name(self):
        return "換手率動量突破"

    def __init__(self,
                 turnover_ma_period=20,
                 turnover_change_threshold=20.0, # in percent
                 min_turnover_rate=3.0, # in percent
                 trend_period=5,
                 min_market_cap=1_000_000_000, # 1 billion
                 min_price=1.0):
        self.turnover_ma_period = turnover_ma_period
        self.turnover_change_threshold = turnover_change_threshold
        self.min_turnover_rate = min_turnover_rate
        self.trend_period = trend_period
        self.min_market_cap = min_market_cap
        self.min_price = min_price

    def run(self, hist: pd.DataFrame, **kwargs) -> bool:
        # --- 1. 獲取 info 對象 ---
        info = kwargs.get('info', {})
        if not info:
            # 如果没有 info 数据，跳过此策略
            return False

        # --- 2. 基本數據長度檢查 ---
        if len(hist) < self.turnover_ma_period + 1:
            return False

        # --- 3. 核心指標計算 ---
        # 獲取流通股本
        float_shares = hist['FloatShares'].iloc[-1]
        if float_shares is None or float_shares == 0 or np.isnan(float_shares):
            return False

        # 計算換手率
        hist['TurnoverRate'] = hist['Volume'] / float_shares * 100
        
        # 計算換手率20日移動平均
        hist['TurnoverRate_MA20'] = hist['TurnoverRate'].rolling(window=self.turnover_ma_period).mean()

        latest = hist.iloc[-1]
        
        # 檢查指標有效性
        if pd.isna(latest['TurnoverRate_MA20']):
            return False

        # 計算換手率變化率 - 使用Z-Score標準化
        turnover_std = hist['TurnoverRate'].rolling(20).std()
        if pd.notna(turnover_std.iloc[-1]) and turnover_std.iloc[-1] > 0:
            turnover_zscore = (latest['TurnoverRate'] - latest['TurnoverRate_MA20']) / turnover_std.iloc[-1]
        else:
            turnover_zscore = 0

        # --- 4. 執行選股條件 ---
        # 條件 1: 換手率Z-Score > 2（要求至少2個標準差）
        if turnover_zscore < 2:
            return False

        # 條件 2: 當日換手率 > 3%
        if latest['TurnoverRate'] <= self.min_turnover_rate:
            return False

        # 條件 2.5: 換手率持續性檢測（不是單日爆發）
        hist['TurnoverRate_MA5'] = hist['TurnoverRate'].rolling(5).mean()
        if latest['TurnoverRate_MA5'] < latest['TurnoverRate_MA20'] * 1.5:
            return False

        # 條件 3: 收盤價 > 5日移動平均
        hist['MA5'] = hist['Close'].rolling(window=self.trend_period).mean()
        if pd.isna(hist['MA5'].iloc[-1]) or latest['Close'] <= hist['MA5'].iloc[-1]:
            return False

        # 條件 4: 市值過濾（優化：5億-500億USD）
        market_cap = info.get('marketCap')
        if market_cap is None or market_cap < 500_000_000:  # 小於5億USD，流動性差
            return False
        elif market_cap > 50_000_000_000:  # 大於500億USD，難以拉升
            return False

        # 條件 5: 價格過濾
        if latest['Close'] < self.min_price:
            return False

        # 條件 6: 價格位置過濾（避免追高）
        hist['52w_high'] = hist['Close'].rolling(252).max()
        distance_to_52w_high = (hist['52w_high'].iloc[-1] - latest['Close']) / hist['52w_high'].iloc[-1]
        if distance_to_52w_high < 0.1:  # 距離52周高點小於10%
            return False

        # 條件 7: 量價配合檢測
        price_change = latest['Close'].pct_change()
        if price_change < 0.03:  # 價格漲幅小於3%
            return False

        # --- 所有條件均滿足 ---
        return True
