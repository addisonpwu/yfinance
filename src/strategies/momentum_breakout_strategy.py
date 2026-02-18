"""
动量爆发策略

检测价格突破、量能爆发、动量强度的共振信号

优化：
- 使用配置类管理参数
- 复用预计算技术指标
- 添加完整的错误处理和日志
"""
from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import MomentumBreakoutConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np


class MomentumBreakoutStrategy(BaseStrategy):
    """
    动量爆发策略：检测价格突破、量能爆发、动量强度逻辑
    """
    
    def __init__(self, config: MomentumBreakoutConfig = None):
        """
        初始化策略
        
        Args:
            config: 策略配置，如果为 None 则从配置文件加载
        """
        self._config = config or strategy_config_manager.get_config('momentum_breakout')
        if not isinstance(self._config, MomentumBreakoutConfig):
            self._config = MomentumBreakoutConfig()
        
        # 验证配置
        if not self._config.validate():
            self._config = MomentumBreakoutConfig()  # 使用默认配置
        
        self._logger = get_analysis_logger()
    
    @property
    def name(self) -> str:
        return "动量爆发策略"
    
    @property
    def category(self) -> str:
        return "动量策略"
    
    @property
    def config(self) -> MomentumBreakoutConfig:
        return self._config
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """
        检查动量爆发条件
        
        Args:
            context: 策略上下文，包含 hist, info, market_return 等
        
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
            
            # 1. 价格突破检查
            price_breakout, price_details = self._check_price_breakout(hist, current_price)
            
            # 2. 量能爆发检查
            volume_burst, volume_details = self._check_volume_burst(hist, current_volume)
            
            # 3. 动量强度检查
            momentum_strength, momentum_details = self._check_momentum_strength(hist, current_price)
            
            # 综合判断
            passed = price_breakout and volume_burst and momentum_strength
            
            # 计算置信度
            confidence = self._calculate_confidence(
                price_breakout, volume_burst, momentum_strength
            )
            
            # 日志记录
            if passed:
                self._logger.info(
                    f"[{self.name}] {context.info.get('symbol', 'Unknown')} 通过筛选 - "
                    f"价格突破: {price_details}, 量能爆发: {volume_details}, 动量强度: {momentum_details}"
                )
            
            return StrategyResult(
                passed=passed,
                confidence=confidence,
                details={
                    "price_breakout": price_breakout,
                    "price_breakout_details": price_details,
                    "volume_burst": volume_burst,
                    "volume_burst_details": volume_details,
                    "momentum_strength": momentum_strength,
                    "momentum_strength_details": momentum_details,
                    "current_price": float(current_price),
                    "current_volume": float(current_volume),
                    "config": {
                        "price_breakout_threshold": self._config.price_breakout_threshold,
                        "volume_burst_multiplier": self._config.volume_burst_multiplier,
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
    
    def _check_price_breakout(self, hist: pd.DataFrame, current_price: float) -> tuple[bool, dict]:
        """
        检查价格突破条件
        
        当前收盘价 > 近20日最高价 × price_breakout_threshold
        
        Returns:
            (是否突破, 详细信息字典)
        """
        try:
            # 尝试使用预计算的 MA 指标
            if 'MA_20' in hist.columns:
                # 使用20日均线作为参考
                ma_20 = hist['MA_20'].iloc[-1]
                if pd.notna(ma_20):
                    high_20 = hist['High'].rolling(window=20).max().iloc[-1]
                else:
                    return False, {"error": "MA_20 数据无效"}
            else:
                # 后备计算
                high_20 = hist['High'].rolling(window=20).max().iloc[-1]
            
            if pd.isna(high_20) or high_20 <= 0:
                return False, {"error": "20日最高价数据无效"}
            
            breakout_price = high_20 * self._config.price_breakout_threshold
            is_breakout = current_price > breakout_price
            
            return is_breakout, {
                "current_price": round(current_price, 2),
                "high_20": round(high_20, 2),
                "breakout_price": round(breakout_price, 2),
                "breakout_pct": round((current_price / high_20 - 1) * 100, 2)
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _check_volume_burst(self, hist: pd.DataFrame, current_volume: float) -> tuple[bool, dict]:
        """
        检查量能爆发条件
        
        当日成交量 > 近20日均量 × volume_burst_multiplier
        
        Returns:
            (是否爆发, 详细信息字典)
        """
        try:
            # 尝试使用预计算的成交量均线
            if 'Volume_MA_20' in hist.columns:
                vol_20_avg = hist['Volume_MA_20'].iloc[-1]
            else:
                vol_20_avg = hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            if pd.isna(vol_20_avg) or vol_20_avg <= 0:
                return False, {"error": "20日均量数据无效"}
            
            vol_ratio = current_volume / vol_20_avg
            is_burst = current_volume > vol_20_avg * self._config.volume_burst_multiplier
            
            return is_burst, {
                "current_volume": int(current_volume),
                "vol_20_avg": int(vol_20_avg),
                "vol_ratio": round(vol_ratio, 2),
                "threshold": self._config.volume_burst_multiplier
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _check_momentum_strength(self, hist: pd.DataFrame, current_price: float) -> tuple[bool, dict]:
        """
        检查动量强度条件
        
        5日涨幅 > momentum_5d_threshold 且 20日涨幅 > momentum_20d_threshold
        
        Returns:
            (是否强势, 详细信息字典)
        """
        try:
            # 计算5日涨幅
            if len(hist) >= 6:
                price_5d_ago = hist['Close'].iloc[-6]
                if pd.notna(price_5d_ago) and price_5d_ago > 0:
                    change_5d = (current_price / price_5d_ago - 1)
                else:
                    change_5d = 0
            else:
                change_5d = 0
            
            # 计算20日涨幅
            if len(hist) >= 21:
                price_20d_ago = hist['Close'].iloc[-21]
                if pd.notna(price_20d_ago) and price_20d_ago > 0:
                    change_20d = (current_price / price_20d_ago - 1)
                else:
                    change_20d = 0
            else:
                change_20d = 0
            
            is_strong = (
                change_5d > self._config.momentum_5d_threshold and
                change_20d > self._config.momentum_20d_threshold
            )
            
            return is_strong, {
                "change_5d_pct": round(change_5d * 100, 2),
                "change_20d_pct": round(change_20d * 100, 2),
                "threshold_5d": self._config.momentum_5d_threshold * 100,
                "threshold_20d": self._config.momentum_20d_threshold * 100
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _calculate_confidence(
        self, 
        price_breakout: bool, 
        volume_burst: bool, 
        momentum_strength: bool
    ) -> float:
        """
        计算策略置信度
        
        根据满足条件的程度计算置信度
        """
        if price_breakout and volume_burst and momentum_strength:
            return 0.85  # 全部满足，高置信度
        elif price_breakout and volume_burst:
            return 0.6   # 价格+量能
        elif price_breakout and momentum_strength:
            return 0.5   # 价格+动量
        elif volume_burst and momentum_strength:
            return 0.4   # 量能+动量
        elif price_breakout:
            return 0.25  # 仅价格
        else:
            return 0.1   # 不满足
