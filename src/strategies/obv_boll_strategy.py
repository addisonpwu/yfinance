"""
OBV 底背离 + BOLL 超卖共振策略

核心逻辑：在长期上升趋势中，捕捉短期超卖且出现底背离的个股

四大核心指标：
1. OBV 底背离 - 股价创新低但 OBV 未创新低，暗示吸筹
2. BOLL 跌破下轨 - 短期超卖，有反弹需求
3. 量比控制 (0.4-1) - 缩量下跌，抛压枯竭
4. 长期趋势过滤 - 价格在 120 日均线之上

最终条件：底背离 AND BOLL 跌破 AND 量比条件 AND 股价线上
"""

from src.core.strategies.strategy import BaseStrategy, StrategyContext, StrategyResult
from src.strategies.strategy_config import OBVBollConfig
from src.config.constants import OBV_BOLL_MIN_DATA_POINTS, OBV_BOLL_CONFIDENCE
from src.utils.logger import get_analysis_logger

logger = get_analysis_logger()


class OBVBollDivergenceStrategy(BaseStrategy):
    """OBV 底背离 + BOLL 超卖共振策略"""
    
    def __init__(self, config: OBVBollConfig = None):
        self.config = config or OBVBollConfig()
    
    @property
    def name(self) -> str:
        return "OBV底背离+BOLL超卖策略"
    
    @property
    def category(self) -> str:
        return "底部反转策略"
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """执行策略"""
        hist = context.hist
        
        # 数据点检查
        if len(hist) < self.config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0,
                details={"reason": f"数据不足，需要至少 {self.config.min_data_points} 天"}
            )
        
        # 获取最新数据
        current_close = hist['Close'].iloc[-1]
        current_low = hist['Low'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        
        # 条件一：OBV 底背离
        obv_divergence, obv_details = self._check_obv_divergence(hist)
        
        # 条件二：BOLL 跌破下轨
        boll_oversold, boll_details = self._check_boll_oversold(hist)
        
        # 条件三：成交量比值控制
        volume_ratio_ok, volume_details = self._check_volume_ratio(hist)
        
        # 条件四：长期趋势过滤
        trend_up, trend_details = self._check_long_term_trend(hist)
        
        # 最终条件
        passed = obv_divergence and boll_oversold and volume_ratio_ok and trend_up
        
        # 计算置信度
        conditions_met = sum([obv_divergence, boll_oversold, volume_ratio_ok, trend_up])
        confidence = self.config.confidence if passed else conditions_met * 0.2
        
        details = {
            "obv_divergence": obv_details,
            "boll_oversold": boll_details,
            "volume_ratio": volume_details,
            "long_term_trend": trend_details,
            "conditions_met": conditions_met,
            "total_conditions": 4
        }
        
        if passed:
            logger.info(f"[{self.name}] OBV底背离 + BOLL超卖共振信号确认")
        
        return StrategyResult(passed=passed, confidence=confidence, details=details)
    
    def _check_obv_divergence(self, hist) -> tuple:
        """检查 OBV 底背离"""
        try:
            current_low = hist['Low'].iloc[-1]
            llv_20 = hist['LLV_20'].iloc[-1]
            
            # 当前最低价是否为 20 日新低
            is_new_low = current_low <= llv_20
            
            # OBV 对比
            current_obv = hist['OBV'].iloc[-1]
            obv_5_days_ago = hist['OBV'].iloc[-(self.config.obv_lookback + 1)]
            
            # OBV 未创新低（高于 5 天前）
            obv_not_lower = current_obv > obv_5_days_ago
            
            # 底背离：股价新低但 OBV 未新低
            divergence = is_new_low and obv_not_lower
            
            details = {
                "passed": divergence,
                "is_new_low": is_new_low,
                "current_low": float(current_low),
                "llv_20": float(llv_20),
                "obv_not_lower": obv_not_lower,
                "current_obv": float(current_obv),
                "obv_5_days_ago": float(obv_5_days_ago),
                "obv_change": float(current_obv - obv_5_days_ago)
            }
            
            return divergence, details
        except Exception as e:
            logger.warning(f"OBV 底背离检查失败: {e}")
            return False, {"passed": False, "error": str(e)}
    
    def _check_boll_oversold(self, hist) -> tuple:
        """检查 BOLL 超卖（跌破下轨）"""
        try:
            current_close = hist['Close'].iloc[-1]
            bb_lower = hist['BB_Lower'].iloc[-1]
            
            # 收盘价跌破布林带下轨
            oversold = current_close < bb_lower
            
            details = {
                "passed": oversold,
                "current_close": float(current_close),
                "bb_lower": float(bb_lower),
                "distance_pct": float((bb_lower - current_close) / bb_lower * 100) if bb_lower > 0 else 0
            }
            
            return oversold, details
        except Exception as e:
            logger.warning(f"BOLL 超卖检查失败: {e}")
            return False, {"passed": False, "error": str(e)}
    
    def _check_volume_ratio(self, hist) -> tuple:
        """检查量比条件"""
        try:
            volume_ratio = hist['Volume_Ratio'].iloc[-1]
            
            # 0.4 < 量比 < 1
            ratio_ok = (self.config.volume_ratio_min < volume_ratio < self.config.volume_ratio_max)
            
            details = {
                "passed": ratio_ok,
                "volume_ratio": float(volume_ratio),
                "min_threshold": self.config.volume_ratio_min,
                "max_threshold": self.config.volume_ratio_max
            }
            
            return ratio_ok, details
        except Exception as e:
            logger.warning(f"量比检查失败: {e}")
            return False, {"passed": False, "error": str(e)}
    
    def _check_long_term_trend(self, hist) -> tuple:
        """检查长期趋势（价格在 120 日均线之上）"""
        try:
            current_close = hist['Close'].iloc[-1]
            ma_120 = hist['MA_120'].iloc[-1]
            
            # 价格在 120 日均线之上
            trend_up = current_close > ma_120
            
            details = {
                "passed": trend_up,
                "current_close": float(current_close),
                "ma_120": float(ma_120),
                "distance_pct": float((current_close - ma_120) / ma_120 * 100) if ma_120 > 0 else 0
            }
            
            return trend_up, details
        except Exception as e:
            logger.warning(f"长期趋势检查失败: {e}")
            return False, {"passed": False, "error": str(e)}
