"""成交量异常放大策略

该策略识别成交量异常放大但股价波动有限的股票，通常预示着主力资金介入，
且市场情绪尚未过热，为潜在的上涨机会提供信号。
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.core.strategies.strategy import BaseStrategy, StrategyContext, StrategyResult


class VolumeAnomalyStrategy(BaseStrategy):
    """成交量异常放大策略"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        # 从配置中获取参数，如果不提供则使用默认值
        self.volume_multiplier_threshold = self.config.get('volume_multiplier_threshold', 2.5)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.05)  # 5%
        self.close_volatility_threshold = self.config.get('close_volatility_threshold', 0.03)  # 3%
        self.price_efficiency_ratio = self.config.get('price_efficiency_ratio', 0.3)
        self.rsi_lower = self.config.get('rsi_lower', 35)
        self.rsi_upper = self.config.get('rsi_upper', 60)
        self.price_position_threshold = self.config.get('price_position_threshold', 0.4)  # 40%分位以下
        self.ma_deviation_threshold = self.config.get('ma_deviation_threshold', 0.05)  # ±5%
        self.mfi_threshold = self.config.get('mfi_threshold', 65)
        self.min_trading_days = self.config.get('min_trading_days', 20)
        
    @property
    def name(self):
        return "成交量异常放大策略"
    
    @property
    def category(self):
        return "成交量分析"
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """计算资金流量指数(MFI)"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = pd.Series(index=typical_price.index, data=0.0)
        negative_flow = pd.Series(index=typical_price.index, data=0.0)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = raw_money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """计算布林带"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, ma, lower_band
        
    def calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, pd.Series]:
        """計算技術指標"""
        indicators = {}
        
        # 計算 RSI
        indicators['rsi'] = self.calculate_rsi(hist['Close'])
        
        # 計算 MFI
        indicators['mfi'] = self.calculate_mfi(hist['High'], hist['Low'], 
                                              hist['Close'], hist['Volume'])
        
        # 計算布林帶
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(hist['Close'])
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # 計算移動平均線
        indicators['ma_5'] = hist['Close'].rolling(window=5).mean()
        indicators['ma_10'] = hist['Close'].rolling(window=10).mean()
        indicators['ma_20'] = hist['Close'].rolling(window=20).mean()
        indicators['ma_50'] = hist['Close'].rolling(window=50).mean()
        
        return indicators
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """執行策略
        
        Args:
            context: 策略執行上下文，包含歷史數據、基本面信息等
        
        Returns:
            StrategyResult: 策略執行結果
        """
        hist = context.hist
        info = context.info
        market_return = context.market_return
        is_market_healthy = context.is_market_healthy
        
        if len(hist) < self.min_trading_days:
            return StrategyResult(passed=False)
            
        # 獲取最新數據
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # 檢查是否有必要數據
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_columns):
            return StrategyResult(passed=False)
        
        # 計算技術指標
        indicators = self.calculate_technical_indicators(hist)
        
        # 一、成交量異常放大條件
        # 1.1 成交量倍數篩選
        if len(hist) < 20:
            return StrategyResult(passed=False)
            
        avg_volume_20 = hist['Volume'].rolling(window=20).mean().iloc[-2] if len(hist) > 20 else hist['Volume'].mean()  # 使用前一日的20日均量
        volume_multiplier = latest['Volume'] / avg_volume_20 if avg_volume_20 > 0 else float('inf')
        
        # 成交量標準差突破指標
        vol_std = hist['Volume'].rolling(window=20).std().iloc[-2] if len(hist) > 20 else 0
        vol_mean = avg_volume_20
        volume_bollinger_upper = vol_mean + 2 * vol_std if vol_std > 0 else float('inf')
        
        volume_condition = (
            volume_multiplier >= self.volume_multiplier_threshold and
            latest['Volume'] > volume_bollinger_upper
        )
        
        if not volume_condition:
            return StrategyResult(passed=False)
        
        # 二、股價波動有限條件
        # 2.1 價格波動率量化定義
        true_range_pct = (latest['High'] - latest['Low']) / prev['Close'] if prev['Close'] > 0 else float('inf')
        close_change_pct = abs((latest['Close'] - prev['Close']) / prev['Close']) if prev['Close'] > 0 else float('inf')
        price_efficiency = true_range_pct / volume_multiplier if volume_multiplier > 0 else float('inf')
        
        volatility_condition = (
            true_range_pct <= self.volatility_threshold and
            close_change_pct <= self.close_volatility_threshold and
            price_efficiency <= self.price_efficiency_ratio
        )
        
        if not volatility_condition:
            return StrategyResult(passed=False)
        
        # 2.2 K線形態特徵
        kline_body_ratio = abs(latest['Close'] - latest['Open']) / (latest['High'] - latest['Low']) if (latest['High'] - latest['Low']) > 0 else 0
        upper_shadow = (latest['High'] - max(latest['Open'], latest['Close']))
        lower_shadow = (min(latest['Open'], latest['Close']) - latest['Low'])
        shadow_symmetry = min(upper_shadow, lower_shadow) / max(upper_shadow, lower_shadow) if max(upper_shadow, lower_shadow) > 0 else 1
        
        kline_condition = kline_body_ratio <= 0.3  # 實體長度比率不超過30%
        
        if not kline_condition:
            return StrategyResult(passed=False)
        
        # 三、價格處於相對低位或整理區間
        # 3.1 價格相對位置度量
        recent_high = hist['High'].rolling(window=20).max().iloc[-1]
        recent_low = hist['Low'].rolling(window=20).min().iloc[-1]
        price_position = (latest['Close'] - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
        
        ma_20 = indicators['ma_20'].iloc[-1] if 'ma_20' in indicators else hist['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = indicators['ma_50'].iloc[-1] if 'ma_50' in indicators else hist['Close'].rolling(window=50).mean().iloc[-1]
        
        ma_deviation_20 = abs(latest['Close'] - ma_20) / ma_20 if ma_20 > 0 else float('inf')
        ma_deviation_50 = abs(latest['Close'] - ma_50) / ma_50 if ma_50 > 0 else float('inf')
        
        position_condition = (
            price_position <= self.price_position_threshold and
            ma_deviation_20 <= self.ma_deviation_threshold and
            ma_deviation_50 <= self.ma_deviation_threshold
        )
        
        if not position_condition:
            return StrategyResult(passed=False)
        
        # 3.2 整理形態識別標準
        ma_5 = indicators['ma_5'].iloc[-1] if 'ma_5' in indicators else hist['Close'].rolling(window=5).mean().iloc[-1]
        ma_10 = indicators['ma_10'].iloc[-1] if 'ma_10' in indicators else hist['Close'].rolling(window=10).mean().iloc[-1]
        ma_20 = indicators['ma_20'].iloc[-1] if 'ma_20' in indicators else hist['Close'].rolling(window=20).mean().iloc[-1]
        
        # 均線系統收斂
        max_ma_diff = max(abs(ma_5 - ma_10), abs(ma_5 - ma_20), abs(ma_10 - ma_20)) / latest['Close'] if latest['Close'] > 0 else float('inf')
        ma_convergence = max_ma_diff <= 0.03  # 不超過3%
        
        # 布林帶形態特徵
        bb_upper = indicators['bb_upper'].iloc[-1] if 'bb_upper' in indicators else None
        bb_lower = indicators['bb_lower'].iloc[-1] if 'bb_lower' in indicators else None
        bb_middle = indicators['bb_middle'].iloc[-1] if 'bb_middle' in indicators else None
        
        bb_width_ratio = (bb_upper - bb_lower) / bb_middle if bb_middle and bb_middle > 0 else float('inf')
        bb_contracted = bb_width_ratio <= 0.1  # 布林帶寬度不超過中軌的10%
        
        consolidation_condition = ma_convergence and bb_contracted
        
        if not consolidation_condition:
            return StrategyResult(passed=False)
        
        # 四、市場情緒尚未過熱
        # 4.1 RSI指標精細應用
        rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators else self.calculate_rsi(hist['Close']).iloc[-1]
        rsi_condition = self.rsi_lower <= rsi <= self.rsi_upper
        
        if not rsi_condition:
            return StrategyResult(passed=False)
        
        # 4.2 輔助情緒指標
        ma_deviation = abs(latest['Close'] - ma_20) / ma_20 if ma_20 > 0 else float('inf')
        macd_condition = True  # 簡化處理，因為MACD計算較複雜
        
        mfi = indicators['mfi'].iloc[-1] if 'mfi' in indicators else self.calculate_mfi(hist['High'], hist['Low'], 
                                                                                      hist['Close'], hist['Volume']).iloc[-1]
        mfi_condition = mfi <= self.mfi_threshold if pd.notna(mfi) else False
        
        sentiment_condition = ma_deviation <= 0.08 and mfi_condition
        
        if not sentiment_condition:
            return StrategyResult(passed=False)
        
        # 五、綜合篩選與確認體系
        # 5.1 條件優先級設置 - 核心必要條件
        core_conditions = [
            volume_condition,      # 成交量放大倍數≥2.5倍
            volatility_condition,  # 日內振幅≤5%
            position_condition,    # 價格處於20日價格區間下半部分
            rsi_condition          # RSI（14）≤60
        ]
        
        if not all(core_conditions):
            return StrategyResult(passed=False)
        
        # 5.2 輔助確認條件（至少滿足三項）
        bb_width_recent = (hist['High'].rolling(window=20).max() - hist['Low'].rolling(window=20).min()).iloc[-1] / ma_20 if ma_20 > 0 else float('inf')
        bb_recent_narrow = bb_width_recent <= 0.75 * (hist['High'].rolling(window=60).max() - hist['Low'].rolling(window=60).min()).iloc[-1] / ma_20 if ma_20 > 0 else False
        
        volume_not_new_high = latest['Volume'] < hist['Volume'].rolling(window=20).max().iloc[-1]  # 成交量放大但價格未創新高
        
        money_flow_balance = True  # 簡化處理
        if len(hist) > 10:
            up_volume = hist['Volume'].where(hist['Close'] > hist['Close'].shift()).rolling(window=10).sum().iloc[-1]
            down_volume = hist['Volume'].where(hist['Close'] < hist['Close'].shift()).rolling(window=10).sum().iloc[-1]
            if up_volume and down_volume and down_volume > 0:
                volume_ratio = up_volume / down_volume
                money_flow_balance = 0.7 <= volume_ratio <= 1.3
            else:
                money_flow_balance = True  # 如果無法計算，視為符合條件
        
        # 計算滿足的輔助確認條件數量
        auxiliary_conditions = [
            bb_recent_narrow,      # 布林帶寬度收縮至近期最窄水平的75%以下
            ma_convergence,        # 短期均線粘合
            volume_not_new_high,   # 成交量放大但價格未創新高
            money_flow_balance     # 資金流向指標顯示平衡狀態
        ]
        
        satisfied_auxiliary = sum(auxiliary_conditions)
        
        # 需要至少滿足三項輔助確認條件
        if satisfied_auxiliary < 3:
            return StrategyResult(passed=False)
        
        return StrategyResult(passed=True, confidence=0.8)