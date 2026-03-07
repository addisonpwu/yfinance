"""
OBV 底背离 + BOLL 超卖共振策略 v2.0 (评分制)

核心逻辑：在长期上升趋势中，捕捉短期超卖且出现底背离的个股

五大评分维度：
1. OBV 底背离 (30%) - 股价创新低但 OBV 未创新低，暗示吸筹
2. 布林带超卖 (25%) - 跌破下轨程度，细分深度/中度/轻度
3. 量比条件 (15%) - 地量最佳，缩量次之，正常再次
4. 长期趋势 (15%) - 价格在 120 日均线之上，距离越远越强
5. CMF 资金流向 (15%) - 资金流入确认

评分制优势：
- 避免硬性 AND 条件过于严格
- 支持多维度综合评估
- 输出止盈止损建议
"""

import pandas as pd

from src.core.strategies.strategy import BaseStrategy, StrategyContext, StrategyResult
from src.strategies.strategy_config import OBVBollConfig
from src.config.constants import OBV_BOLL_MIN_DATA_POINTS, OBV_BOLL_CONFIDENCE
from src.utils.logger import get_analysis_logger

logger = get_analysis_logger()


class OBVBollDivergenceStrategy(BaseStrategy):
    """OBV 底背离 + BOLL 超卖共振策略 v2.0 (评分制)"""
    
    def __init__(self, config: OBVBollConfig = None):
        self.config = config or OBVBollConfig()
    
    @property
    def name(self) -> str:
        return "OBV底背离+BOLL超卖策略"
    
    @property
    def category(self) -> str:
        return "底部反转策略"
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """执行策略 - 评分制 v2.0"""
        hist = context.hist
        
        # 数据点检查
        if len(hist) < self.config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0,
                details={"reason": f"数据不足，需要至少 {self.config.min_data_points} 天"}
            )
        
        # 1. 计算各维度评分
        obv_score, obv_type, obv_details = self._check_obv_divergence_enhanced(hist)
        boll_score, boll_type, boll_details = self._check_boll_oversold_enhanced(hist)
        vol_score, vol_type, vol_details = self._check_volume_ratio_standard(hist)
        cmf_score, cmf_type, cmf_details = self._check_money_flow(hist)
        trend_score, trend_type, trend_details = self._check_long_term_trend_enhanced(hist)
        
        # 2. 计算总分
        total_score = obv_score + boll_score + vol_score + cmf_score + trend_score
        
        # 3. 市场环境调整
        effective_threshold = self.config.min_pass_score
        if not context.is_market_healthy:
            effective_threshold += self.config.unhealthy_threshold_add
        
        # 4. 判断是否通过
        passed = total_score >= effective_threshold
        
        # 5. 动态置信度
        base_confidence = total_score / 100
        if not context.is_market_healthy:
            base_confidence *= self.config.unhealthy_confidence_penalty
        confidence = max(0.3, min(0.95, base_confidence))
        
        # 6. 强信号判断
        is_strong_signal = total_score >= self.config.strong_signal_score
        
        # 7. 止盈止损计算（仅通过时计算）
        risk_management = None
        if passed:
            risk_management = self._calculate_risk_management(hist)
        
        # 构建结果
        details = {
            "total_score": round(total_score, 1),
            "effective_threshold": effective_threshold,
            "is_strong_signal": is_strong_signal,
            "divergence_type": obv_type,
            "oversold_type": boll_type,
            "volume_type": vol_type,
            "money_flow_type": cmf_type,
            "trend_type": trend_type,
            "score_breakdown": {
                "obv_divergence": round(obv_score, 1),
                "boll_oversold": round(boll_score, 1),
                "volume_ratio": round(vol_score, 1),
                "money_flow": round(cmf_score, 1),
                "trend": round(trend_score, 1)
            },
            "obv_details": obv_details,
            "boll_details": boll_details,
            "volume_details": vol_details,
            "cmf_details": cmf_details,
            "trend_details": trend_details,
            "market_healthy": context.is_market_healthy,
            "risk_management": risk_management
        }
        
        if passed:
            logger.info(f"[{self.name}] 评分制通过: 总分={total_score:.1f}, 阈值={effective_threshold}")
        
        return StrategyResult(passed=passed, confidence=round(confidence, 2), details=details)
    
    def _check_obv_divergence_enhanced(self, hist) -> tuple:
        """
        OBV底背离增强版：
        1. 股价创新低（20日）
        2. 双区间对比：当前OBV vs 5日前 vs 10日前
        3. OBV斜率验证
        
        返回: (score, divergence_type, details)
        """
        try:
            current_low = hist['Low'].iloc[-1]
            llv_20 = hist['LLV_20'].iloc[-1]
            is_new_low = current_low <= llv_20
            
            obv_current = hist['OBV'].iloc[-1]
            obv_5d_ago = hist['OBV'].iloc[-6] if len(hist) >= 6 else obv_current
            obv_10d_ago = hist['OBV'].iloc[-11] if len(hist) >= 11 else obv_current
            
            # 双区间背离
            obv_vs_5d = obv_current > obv_5d_ago
            obv_vs_10d = obv_current > obv_10d_ago
            
            # OBV斜率 (5日)
            obv_slope = (obv_current - obv_5d_ago) / 5 / abs(obv_5d_ago) if obv_5d_ago != 0 else 0
            
            # 评分
            score = 0
            divergence_type = "无"
            
            if is_new_low:
                if obv_vs_5d and obv_vs_10d and obv_slope > self.config.obv_slope_min:
                    score = self.config.obv_weight  # 30分，强背离
                    divergence_type = "强"
                elif obv_vs_5d:
                    score = self.config.obv_weight * 0.67  # 20分，中等背离
                    divergence_type = "中等"
                elif obv_current > obv_10d_ago:
                    score = self.config.obv_weight * 0.33  # 10分，弱背离
                    divergence_type = "弱"
            
            details = {
                "is_new_low": is_new_low,
                "current_low": round(float(current_low), 2),
                "llv_20": round(float(llv_20), 2),
                "obv_vs_5d": obv_vs_5d,
                "obv_vs_10d": obv_vs_10d,
                "obv_slope": round(obv_slope, 4),
                "obv_current": round(float(obv_current), 0),
                "divergence_type": divergence_type
            }
            
            return score, divergence_type, details
            
        except Exception as e:
            logger.warning(f"OBV底背离检查失败: {e}")
            return 0, "错误", {"error": str(e)}
    
    def _check_boll_oversold_enhanced(self, hist) -> tuple:
        """
        布林带超卖细化：
        1. 跌破幅度
        2. 带宽压缩状态
        
        返回: (score, oversold_type, details)
        """
        try:
            current_close = hist['Close'].iloc[-1]
            bb_lower = hist['BB_Lower'].iloc[-1]
            bb_upper = hist['BB_Upper'].iloc[-1]
            bb_middle = hist['BB_Middle'].iloc[-1]
            
            # 跌破幅度
            breach_pct = (bb_lower - current_close) / bb_lower if bb_lower > 0 else 0
            
            # 带宽
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            
            # 带宽百分位（近100日）
            lookback = min(self.config.bb_width_lookback, len(hist))
            bb_width_series = (hist['BB_Upper'].iloc[-lookback:] - hist['BB_Lower'].iloc[-lookback:]) / hist['BB_Middle'].iloc[-lookback:]
            # 过滤无效值
            bb_width_series = bb_width_series.dropna()
            if len(bb_width_series) > 0:
                width_percentile = (bb_width_series < bb_width).sum() / len(bb_width_series)
            else:
                width_percentile = 0.5
            
            # 评分
            score = 0
            oversold_type = "无"
            
            if breach_pct > self.config.bb_deep_oversold:
                score = self.config.boll_weight  # 25分，深度超卖
                oversold_type = "深度"
            elif breach_pct > self.config.bb_mid_oversold:
                score = self.config.boll_weight * 0.72  # 18分，中度超卖
                oversold_type = "中度"
            elif breach_pct > 0:
                score = self.config.boll_weight * 0.4  # 10分，轻度超卖
                oversold_type = "轻度"
            
            # 带宽压缩加分
            is_squeeze = width_percentile < self.config.bb_squeeze_percentile
            if is_squeeze and score > 0:
                score = min(score + 3, self.config.boll_weight)  # 加3分，但不超过满分
            
            details = {
                "breach_pct": round(breach_pct * 100, 2),
                "bb_width": round(bb_width, 4),
                "width_percentile": round(width_percentile, 2),
                "is_squeeze": is_squeeze,
                "current_close": round(float(current_close), 2),
                "bb_lower": round(float(bb_lower), 2),
                "oversold_type": oversold_type
            }
            
            return score, oversold_type, details
            
        except Exception as e:
            logger.warning(f"布林带超卖检查失败: {e}")
            return 0, "错误", {"error": str(e)}
    
    def _check_volume_ratio_standard(self, hist) -> tuple:
        """
        标准量比评估
        
        返回: (score, label, details)
        """
        try:
            vol_ratio_col = 'Volume_Ratio_Standard'
            
            if vol_ratio_col not in hist.columns:
                # 降级使用旧量比
                vol_ratio = hist['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in hist.columns else 1.0
            else:
                vol_ratio = hist[vol_ratio_col].iloc[-1]
            
            score = 0
            label = "异常"
            
            if pd.isna(vol_ratio):
                score = self.config.volume_weight * 0.5
                label = "数据缺失"
            elif vol_ratio < 0.5:
                score = self.config.volume_weight  # 15分，地量
                label = "地量"
            elif vol_ratio < 0.8:
                score = self.config.volume_weight * 0.8  # 12分，缩量
                label = "缩量"
            elif vol_ratio <= 1.2:
                score = self.config.volume_weight * 0.5  # 8分，正常
                label = "正常"
            else:
                score = self.config.volume_weight * 0.2  # 3分，放量
                label = "放量"
            
            details = {
                "volume_ratio": round(float(vol_ratio), 2),
                "label": label
            }
            
            return score, label, details
            
        except Exception as e:
            logger.warning(f"量比检查失败: {e}")
            return 0, "错误", {"error": str(e)}
    
    def _check_money_flow(self, hist) -> tuple:
        """
        CMF资金流向验证
        
        返回: (score, flow_type, details)
        """
        try:
            cmf_col = f'CMF_{self.config.cmf_period}'
            
            if cmf_col not in hist.columns:
                return 0, "无数据", {"error": "CMF not calculated"}
            
            cmf_current = hist[cmf_col].iloc[-1]
            
            score = 0
            flow_type = "流出"
            
            if cmf_current > 0.1:
                score = self.config.cmf_weight  # 15分
                flow_type = "强流入"
            elif cmf_current > 0:
                score = self.config.cmf_weight * 0.67  # 10分
                flow_type = "流入"
            elif cmf_current > -0.1:
                score = self.config.cmf_weight * 0.33  # 5分
                flow_type = "平衡"
            else:
                score = 0
                flow_type = "流出"
            
            details = {
                "cmf_value": round(float(cmf_current), 3),
                "flow_type": flow_type
            }
            
            return score, flow_type, details
            
        except Exception as e:
            logger.warning(f"CMF资金流检查失败: {e}")
            return 0, "错误", {"error": str(e)}
    
    def _check_long_term_trend_enhanced(self, hist) -> tuple:
        """
        长期趋势评分
        
        返回: (score, trend_type, details)
        """
        try:
            current_close = hist['Close'].iloc[-1]
            ma_120 = hist['MA_120'].iloc[-1] if 'MA_120' in hist.columns else None
            
            score = 0
            trend_type = "下跌趋势"
            distance_pct = None
            
            if ma_120 is None or pd.isna(ma_120):
                score = self.config.trend_weight * 0.5  # 数据不足，给一半分
                trend_type = "数据不足"
            elif current_close > ma_120:
                distance_pct = (current_close - ma_120) / ma_120
                if distance_pct > 0.05:
                    score = self.config.trend_weight  # 15分，强势上涨
                    trend_type = "强势上涨"
                else:
                    score = self.config.trend_weight * 0.67  # 10分，上涨趋势
                    trend_type = "上涨趋势"
            else:
                score = 0
                trend_type = "下跌趋势"
                distance_pct = (current_close - ma_120) / ma_120
            
            details = {
                "ma_120": round(float(ma_120), 2) if ma_120 and not pd.isna(ma_120) else None,
                "current_close": round(float(current_close), 2),
                "distance_pct": round(distance_pct * 100, 2) if distance_pct is not None else None,
                "trend_type": trend_type
            }
            
            return score, trend_type, details
            
        except Exception as e:
            logger.warning(f"长期趋势检查失败: {e}")
            return 0, "错误", {"error": str(e)}
    
    def _calculate_risk_management(self, hist) -> dict:
        """
        计算止盈止损
        
        返回: 风险管理字典，包含入场价、止损价、止盈价等
        """
        try:
            atr = self.calculate_atr(hist, period=14)
            current_close = hist['Close'].iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                return None
            
            stop_loss = current_close - (atr * self.config.stop_loss_atr_mult)
            take_profit_1 = current_close + (atr * self.config.take_profit_atr_mult_1)
            take_profit_2 = current_close + (atr * self.config.take_profit_atr_mult_2)
            
            risk = current_close - stop_loss
            reward = take_profit_1 - current_close
            rr_ratio = reward / risk if risk > 0 else 0
            
            return {
                "entry_price": round(float(current_close), 2),
                "stop_loss": round(float(stop_loss), 2),
                "stop_loss_pct": round((current_close - stop_loss) / current_close * 100, 2),
                "take_profit_1": round(float(take_profit_1), 2),
                "take_profit_2": round(float(take_profit_2), 2),
                "risk_reward_ratio": round(rr_ratio, 2),
                "atr_value": round(float(atr), 2)
            }
            
        except Exception as e:
            logger.warning(f"止盈止损计算失败: {e}")
            return None