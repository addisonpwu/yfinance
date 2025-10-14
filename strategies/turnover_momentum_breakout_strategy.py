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
            
        # 計算換手率變化率
        if latest['TurnoverRate_MA20'] == 0:
            turnover_change_rate = float('inf') # 如果均值是0，任何正的換手率都是無限大增長
        else:
            turnover_change_rate = (latest['TurnoverRate'] - latest['TurnoverRate_MA20']) / latest['TurnoverRate_MA20'] * 100

        # --- 4. 執行選股條件 ---
        # 條件 1: 換手率變化率 > 20%
        if turnover_change_rate <= self.turnover_change_threshold:
            return False

        # 條件 2: 當日換手率 > 3%
        if latest['TurnoverRate'] <= self.min_turnover_rate:
            return False

        # 條件 3: 收盤價 > 5日移動平均
        hist['MA5'] = hist['Close'].rolling(window=self.trend_period).mean()
        if pd.isna(hist['MA5'].iloc[-1]) or latest['Close'] <= hist['MA5'].iloc[-1]:
            return False

        # 條件 4: 市值過濾
        market_cap = info.get('marketCap')
        if market_cap is None or market_cap < self.min_market_cap:
            return False

        # 條件 5: 價格過濾
        # yfinance返回的價格總是對應於上市地的貨幣，所以可以直接比較
        if latest['Close'] < self.min_price:
            return False

        # --- 所有條件均滿足 ---
        return True
