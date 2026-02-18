"""
主力吸筹加速策略

检测横盘吸筹、量能趋势、突破信号、RSI动态上穿

优化：
- 使用配置类管理参数
- 复用预计算的 RSI 指标
- 添加完整的错误处理和日志
"""
from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import AccumulationAccelerationConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np


class AccumulationAccelerationStrategy(BaseStrategy):
    """
    主力吸筹加速策略：检测横盘吸筹、量能趋势、突破信号、RSI动态上穿逻辑
    """
    
    def __init__(self, config: AccumulationAccelerationConfig = None):
        """
        初始化策略
        
        Args:
            config: 策略配置，如果为 None 则从配置文件加载
        """
        self._config = config or strategy_config_manager.get_config('accumulation_acceleration')
        if not isinstance(self._config, AccumulationAccelerationConfig):
            self._config = AccumulationAccelerationConfig()
        
        # 验证配置
        if not self._config.validate():
            self._config = AccumulationAccelerationConfig()
        
        self._logger = get_analysis_logger()
    
    @property
    def name(self) -> str:
        return "主力吸筹加速策略"
    
    @property
    def category(self) -> str:
        return "吸筹策略"
    
    @property
    def config(self) -> AccumulationAccelerationConfig:
        return self._config
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """
        检查主力吸筹加速条件（包含动态RSI上穿）
        
        Args:
            context: 策略上下文
        
        Returns:
            StrategyResult: 包含是否通过、置信度和详细信息
        """
        hist = context.hist
        
        # 数据验证
        if hist is None or len(hist) < self._config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"数据不足，需要至少 {self._config.min_data_points} 天"}
            )
        
        try:
            # 获取当前数据
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # 检查数据有效性
            if pd.isna(current_price) or pd.isna(current_volume) or current_price <= 0:
                return StrategyResult(
                    passed=False,
                    confidence=0.0,
                    details={"reason": "价格或成交量数据无效"}
                )
            
            # 1. 吸筹期检查：近30日价格波动幅度 < 阈值
            accumulation_period, accumulation_details = self._check_accumulation_period(hist)
            
            # 2. 量能趋势检查：近30日成交量呈上升趋势
            volume_trend, volume_trend_details = self._check_volume_trend(hist)
            
            # 3. 加速信号检查：价格突破 + 量比
            acceleration_signal, acceleration_details = self._check_acceleration_signal(
                hist, current_price, current_volume
            )
            
            # 4. 动态RSI条件检查
            rsi_momentum, rsi_details = self._check_dynamic_rsi_condition(hist)
            
            # 综合判断
            passed = accumulation_period and volume_trend and acceleration_signal and rsi_momentum
            
            # 计算置信度
            confidence = self._calculate_confidence(
                accumulation_period, volume_trend, acceleration_signal, rsi_momentum
            )
            
            # 日志记录
            if passed:
                self._logger.info(
                    f"[{self.name}] {context.info.get('symbol', 'Unknown')} 通过筛选 - "
                    f"吸筹期: {accumulation_details.get('volatility_pct', 'N/A')}, "
                    f"RSI: {rsi_details.get('rsi_current', 'N/A')}"
                )
            
            return StrategyResult(
                passed=passed,
                confidence=confidence,
                details={
                    "accumulation_period": accumulation_period,
                    "accumulation_period_details": accumulation_details,
                    "volume_trend": volume_trend,
                    "volume_trend_details": volume_trend_details,
                    "acceleration_signal": acceleration_signal,
                    "acceleration_signal_details": acceleration_details,
                    "rsi_momentum": rsi_momentum,
                    "rsi_details": rsi_details,
                    "config": {
                        "accumulation_volatility_threshold": self._config.accumulation_volatility_threshold,
                        "rsi_breakout_threshold": self._config.rsi_breakout_threshold,
                    }
                }
            )
            
        except Exception as e:
            self._logger.error(f"[{self.name}] 执行策略时出错: {e}")
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"策略执行错误: {str(e)}"}
            )
    
    def _check_accumulation_period(self, hist: pd.DataFrame) -> tuple[bool, dict]:
        """
        检查吸筹期：近N日价格波动幅度 < 阈值
        
        Returns:
            (是否处于吸筹期, 详细信息字典)
        """
        period = self._config.accumulation_period
        
        if len(hist) < period:
            return False, {"error": "数据不足"}
        
        try:
            recent_hist = hist.iloc[-period:]
            price_range = recent_hist['High'].max() - recent_hist['Low'].min()
            avg_price = recent_hist['Close'].mean()
            
            if pd.isna(avg_price) or avg_price <= 0:
                return False, {"error": "平均价格无效"}
            
            volatility = price_range / avg_price
            is_accumulating = volatility < self._config.accumulation_volatility_threshold
            
            return is_accumulating, {
                "period": period,
                "price_range": round(price_range, 2),
                "avg_price": round(avg_price, 2),
                "volatility_pct": round(volatility * 100, 2),
                "threshold_pct": round(self._config.accumulation_volatility_threshold * 100, 2)
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _check_volume_trend(self, hist: pd.DataFrame) -> tuple[bool, dict]:
        """
        检查量能趋势：近30日成交量呈上升趋势
        
        Returns:
            (是否上升趋势, 详细信息字典)
        """
        period = self._config.accumulation_period
        
        if len(hist) < period:
            return False, {"error": "数据不足"}
        
        try:
            half_period = period // 2
            
            # 前半段成交量均值
            vol_early = hist['Volume'].iloc[-period:-half_period].mean()
            # 后半段成交量均值
            vol_late = hist['Volume'].iloc[-half_period:].mean()
            
            if pd.isna(vol_early) or pd.isna(vol_late):
                return False, {"error": "成交量数据无效"}
            
            # 计算趋势
            vol_ratio = vol_late / vol_early if vol_early > 0 else 0
            is_uptrend = vol_late > vol_early
            
            return is_uptrend, {
                "vol_early_avg": int(vol_early),
                "vol_late_avg": int(vol_late),
                "vol_ratio": round(vol_ratio, 2),
                "trend": "上升" if is_uptrend else "下降"
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _check_acceleration_signal(
        self, 
        hist: pd.DataFrame, 
        current_price: float, 
        current_volume: float
    ) -> tuple[bool, dict]:
        """
        检查加速信号：当前价 > 横盘上沿 × multiplier 且 量比 > threshold
        
        Returns:
            (是否有加速信号, 详细信息字典)
        """
        period = self._config.accumulation_period
        
        if len(hist) < period:
            return False, {"error": "数据不足"}
        
        try:
            # 横盘上沿（期间最高价）
            range_high = hist['High'].iloc[-period:].max()
            
            if pd.isna(range_high) or range_high <= 0:
                return False, {"error": "区间最高价无效"}
            
            # 计算20日均量
            if 'Volume_MA_20' in hist.columns:
                vol_20_avg = hist['Volume_MA_20'].iloc[-1]
            else:
                vol_20_avg = hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            if pd.isna(vol_20_avg) or vol_20_avg <= 0:
                return False, {"error": "成交量均值无效"}
            
            # 价格突破检查
            breakout_price = range_high * self._config.acceleration_price_multiplier
            price_breakout = current_price > breakout_price
            
            # 量比检查
            vol_ratio = current_volume / vol_20_avg
            volume_surge = vol_ratio > self._config.acceleration_volume_ratio
            
            has_signal = price_breakout and volume_surge
            
            return has_signal, {
                "range_high": round(range_high, 2),
                "breakout_price": round(breakout_price, 2),
                "current_price": round(current_price, 2),
                "price_breakout": price_breakout,
                "vol_ratio": round(vol_ratio, 2),
                "vol_threshold": self._config.acceleration_volume_ratio,
                "volume_surge": volume_surge
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _check_dynamic_rsi_condition(self, hist: pd.DataFrame) -> tuple[bool, dict]:
        """
        检查动态RSI条件：前一日RSI在区间内，当日RSI > 阈值
        
        优先使用预计算的 RSI 指标，避免重复计算
        
        Returns:
            (是否满足条件, 详细信息字典)
        """
        if len(hist) < 2:
            return False, {"error": "数据不足"}
        
        try:
            rsi_period = self._config.rsi_period
            
            # 尝试使用预计算的 RSI
            rsi_col = f'RSI_{rsi_period}'
            
            if rsi_col in hist.columns:
                # 使用预计算的 RSI
                rsi_series = hist[rsi_col]
                rsi_current = rsi_series.iloc[-1]
                rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else None
            else:
                # 后备计算
                rsi_series = self._calculate_rsi_series(hist, rsi_period)
                rsi_current = rsi_series.iloc[-1] if len(rsi_series) > 0 else None
                rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else None
            
            if pd.isna(rsi_current) or pd.isna(rsi_prev):
                return False, {"error": "RSI 计算失败"}
            
            # 检查条件
            prev_in_range = (
                self._config.rsi_low_range_min <= rsi_prev <= self._config.rsi_low_range_max
            )
            current_above_threshold = rsi_current > self._config.rsi_breakout_threshold
            
            has_momentum = prev_in_range and current_above_threshold
            
            return has_momentum, {
                "rsi_prev": round(rsi_prev, 2),
                "rsi_current": round(rsi_current, 2),
                "rsi_range": f"{self._config.rsi_low_range_min}-{self._config.rsi_low_range_max}",
                "rsi_threshold": self._config.rsi_breakout_threshold,
                "prev_in_range": prev_in_range,
                "current_above_threshold": current_above_threshold
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _calculate_rsi_series(self, hist: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算 RSI 序列（后备方法）
        
        Args:
            hist: 历史数据
            period: RSI 周期
        
        Returns:
            RSI 序列
        """
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_confidence(
        self,
        accumulation: bool,
        volume_trend: bool,
        acceleration: bool,
        rsi_momentum: bool
    ) -> float:
        """
        计算策略置信度
        """
        if accumulation and volume_trend and acceleration and rsi_momentum:
            return 0.85
        elif accumulation and acceleration and rsi_momentum:
            return 0.7
        elif accumulation and volume_trend and acceleration:
            return 0.6
        elif acceleration and rsi_momentum:
            return 0.5
        elif accumulation and acceleration:
            return 0.4
        else:
            return 0.1
