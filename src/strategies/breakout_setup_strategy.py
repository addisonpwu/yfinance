"""
启动前兆策略 v2.0

最具"埋伏"性质的策略，最大化信号的前瞻性和准确性

核心优化：
1. 蓄势形态强化：三角形整理、旗形整理、楔形识别 + 线性回归价格通道
2. 波动率压缩深化：TTM Squeeze（布林带收窄至肯特纳通道内部）
3. 弱转强信号：相对强弱（RS）分析，抗跌+回升特征
4. 评分体系精细化：乘法模型，均线粘合作为前提条件
"""

from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import BreakoutSetupConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    线性回归（纯 numpy 实现，避免 scipy 依赖）
    
    Returns:
        (slope, intercept, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if len(y) > 0 else 0.0, 0.0
    
    # 计算斜率和截距
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0, y_mean, 0.0
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # 计算 R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    
    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared


class BreakoutSetupStrategy(BaseStrategy):
    """
    启动前兆策略 v2.0
    
    捕捉即将突破启动的股票：
    - 均线粘合（前提条件）
    - 蓄势形态识别（三角形、旗形、楔形）
    - TTM Squeeze 波动率压缩
    - 弱转强信号（相对强弱分析）
    """
    
    def __init__(self, config: BreakoutSetupConfig = None):
        self._config = config or strategy_config_manager.get_config('breakout_setup')
        if not isinstance(self._config, BreakoutSetupConfig):
            self._config = BreakoutSetupConfig()
        
        if not self._config.validate():
            self._config = BreakoutSetupConfig()
        
        self._logger = get_analysis_logger()
    
    @property
    def name(self) -> str:
        return "启动前兆策略"
    
    @property
    def category(self) -> str:
        return "早期信号策略"
    
    @property
    def config(self) -> BreakoutSetupConfig:
        return self._config
    
    def execute(self, context: StrategyContext) -> StrategyResult:
        """执行启动前兆策略检查"""
        hist = context.hist
        benchmark = context.benchmark  # 大盘数据（可选）
        
        if hist is None or len(hist) < self._config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"数据不足，需要至少 {self._config.min_data_points} 天"}
            )
        
        try:
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            if pd.isna(current_price) or pd.isna(current_volume) or current_price <= 0:
                return StrategyResult(
                    passed=False,
                    confidence=0.0,
                    details={"reason": "价格或成交量数据无效"}
                )
            
            # === 1. 均线粘合评分（前提条件）===
            ma_result = self._score_ma_convergence_enhanced(hist, current_price)
            
            # === 2. 蓄势形态识别（强化版）===
            pattern_result = self._score_price_pattern_enhanced(hist, current_price)
            
            # === 3. 成交量模式评分 ===
            vol_result = self._score_volume_setup(hist, current_volume)
            
            # === 4. 波动率压缩评分（TTM Squeeze）===
            volatility_result = self._score_volatility_squeeze_enhanced(hist, current_price)
            
            # === 5. 技术指标评分 ===
            tech_result = self._score_technical_setup(hist, current_price)
            
            # === 6. 弱转强信号（新增）===
            rs_result = self._score_relative_strength(hist, benchmark, current_price)
            
            # === 乘法模型评分 ===
            # 均线粘合是前提，如果粘合程度得分低，总分打折
            ma_multiplier = self._calculate_ma_multiplier(ma_result['score'])
            
            # 计算加权总分
            raw_score = (
                ma_result['score'] * 0.20 +           # 均线粘合 20%
                pattern_result['score'] * 0.25 +      # 蓄势形态 25%
                vol_result['score'] * 0.15 +          # 成交量 15%
                volatility_result['score'] * 0.20 +   # 波动率压缩 20%
                tech_result['score'] * 0.10 +         # 技术指标 10%
                rs_result['score'] * 0.10             # 弱转强 10%
            )
            
            # 应用乘法模型
            total_score = raw_score * ma_multiplier
            
            # 通过条件
            passed = (
                total_score >= self._config.min_score and
                ma_result['passed']  # 均线粘合必须通过
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(
                total_score, ma_result, pattern_result, volatility_result
            )
            
            # 预测突破方向
            breakout_direction = self._predict_breakout_direction(
                hist, ma_result, pattern_result, rs_result
            )
            
            # 信号强度
            signal_strength = self._determine_signal_strength(total_score)
            
            # 预警等级
            alert_level = self._determine_alert_level(
                total_score, volatility_result, pattern_result
            )
            
            if passed:
                self._logger.info(
                    f"[{self.name}] {context.info.get('symbol', 'Unknown')} 通过筛选 - "
                    f"总分: {total_score:.1f} (乘数: {ma_multiplier:.2f}), "
                    f"方向: {breakout_direction}, 形态: {pattern_result.get('pattern', 'N/A')}, "
                    f"Squeeze: {volatility_result.get('squeeze_status', 'N/A')}"
                )
            
            return StrategyResult(
                passed=passed,
                confidence=confidence,
                details={
                    "total_score": round(total_score, 1),
                    "raw_score": round(raw_score, 1),
                    "ma_multiplier": round(ma_multiplier, 2),
                    "signal_strength": signal_strength,
                    "alert_level": alert_level,
                    "breakout_direction": breakout_direction,
                    "scores": {
                        "ma_convergence": ma_result['score'],
                        "price_pattern": pattern_result['score'],
                        "volume": vol_result['score'],
                        "volatility": volatility_result['score'],
                        "technical": tech_result['score'],
                        "relative_strength": rs_result['score']
                    },
                    "ma_details": ma_result,
                    "pattern_details": pattern_result,
                    "volume_details": vol_result,
                    "volatility_details": volatility_result,
                    "technical_details": tech_result,
                    "rs_details": rs_result
                }
            )
            
        except Exception as e:
            self._logger.error(f"[{self.name}] 执行策略时出错: {e}")
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"策略执行错误: {str(e)}"}
            )
    
    def _score_ma_convergence_enhanced(
        self, 
        hist: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        增强的均线粘合评分
        
        作为前提条件，影响最终评分的乘数
        
        Returns:
            包含 score, passed, details 的字典
        """
        try:
            # 获取均线
            ma_5 = hist['MA_5'].iloc[-1] if 'MA_5' in hist.columns else hist['Close'].rolling(5).mean().iloc[-1]
            ma_10 = hist['MA_10'].iloc[-1] if 'MA_10' in hist.columns else hist['Close'].rolling(10).mean().iloc[-1]
            ma_20 = hist['MA_20'].iloc[-1] if 'MA_20' in hist.columns else hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['MA_50'].iloc[-1] if 'MA_50' in hist.columns else (hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None)
            
            if pd.isna(ma_5) or pd.isna(ma_10) or pd.isna(ma_20):
                return {"score": 0, "passed": False, "details": {"error": "均线数据无效"}}
            
            # 计算均线粘合程度
            ma_values = [ma for ma in [ma_5, ma_10, ma_20, ma_50] if ma is not None and not pd.isna(ma)]
            ma_max = max(ma_values)
            ma_min = min(ma_values)
            ma_avg = sum(ma_values) / len(ma_values)
            ma_spread = (ma_max - ma_min) / ma_avg
            
            # 均线粘合评分（0-25分）
            score = 0
            if ma_spread < 0.015:  # 极度粘合
                score = 25
                status = "极度粘合"
            elif ma_spread < 0.025:  # 强粘合
                score = 22
                status = "强粘合"
            elif ma_spread < 0.04:  # 粘合
                score = 18
                status = "均线粘合"
            elif ma_spread < 0.06:  # 收敛
                score = 12
                status = "均线收敛"
            elif ma_spread < 0.10:  # 接近
                score = 6
                status = "均线接近"
            else:
                score = 0
                status = "均线发散"
            
            # 检查均线多头排列
            alignment = "交叉排列"
            alignment_bonus = 0
            
            if ma_5 > ma_10 > ma_20:
                alignment = "多头排列"
                alignment_bonus = 5
                score += alignment_bonus
            elif ma_5 < ma_10 < ma_20:
                alignment = "空头排列"
                alignment_bonus = 2
                score += alignment_bonus
            
            # 检查均线发散趋势（粘合后开始发散 = 突破信号）
            diverging = False
            if len(hist) >= 10:
                ma_5_prev = hist['MA_5'].iloc[-10] if 'MA_5' in hist.columns else None
                ma_20_prev = hist['MA_20'].iloc[-10] if 'MA_20' in hist.columns else None
                
                if ma_5_prev and ma_20_prev and not pd.isna(ma_5_prev) and not pd.isna(ma_20_prev):
                    prev_spread = abs(ma_5_prev - ma_20_prev) / ma_20_prev
                    if ma_spread > prev_spread * 1.3 and ma_5 > ma_20:  # 向上发散
                        diverging = True
                        score += 3
            
            # 价格与均线关系
            price_ma_score = 0
            if current_price > ma_5 > ma_10 > ma_20:
                price_ma_score = 3
                score += 3
            elif current_price > ma_avg:
                price_ma_score = 1
                score += 1
            
            # 通过条件：粘合程度达到阈值
            passed = ma_spread < self._config.ma_convergence_threshold
            
            return {
                "score": min(score, 30),
                "passed": passed,
                "ma_spread_pct": round(ma_spread * 100, 2),
                "ma_status": status,
                "ma_alignment": alignment,
                "alignment_bonus": alignment_bonus,
                "diverging": diverging,
                "price_ma_score": price_ma_score,
                "details": {
                    "ma_5": round(ma_5, 2),
                    "ma_10": round(ma_10, 2),
                    "ma_20": round(ma_20, 2),
                    "ma_50": round(ma_50, 2) if ma_50 else None
                }
            }
            
        except Exception as e:
            return {"score": 0, "passed": False, "details": {"error": str(e)}}
    
    def _score_price_pattern_enhanced(
        self, 
        hist: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        增强的蓄势形态识别
        
        包含：
        1. 三角形整理（上升三角形、下降三角形、对称三角形）
        2. 旗形整理（看涨旗形、看跌旗形）
        3. 楔形（上升楔形、下降楔形）
        4. 线性回归价格通道
        
        Returns:
            包含 score, pattern, details 的字典
        """
        try:
            period = self._config.consolidation_period
            if len(hist) < period:
                return {"score": 0, "pattern": "数据不足"}
            
            recent = hist.iloc[-period:].copy()
            
            # === 1. 线性回归价格通道 ===
            x = np.arange(len(recent))
            y = recent['Close'].values
            
            # 线性回归（纯 numpy 实现）
            slope, intercept, r_squared = _linear_regression(x, y)
            
            # 计算通道上下轨
            regression_line = slope * x + intercept
            residuals = y - regression_line
            std_residual = np.std(residuals)
            
            upper_channel = regression_line + 2 * std_residual
            lower_channel = regression_line - 2 * std_residual
            
            # 当前价格在通道中的位置
            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            channel_position = (current_price - current_lower) / (current_upper - current_lower)
            
            # 通道宽度变化（收敛判断）
            channel_width_start = upper_channel[0] - lower_channel[0]
            channel_width_end = current_upper - current_lower
            channel_convergence = channel_width_end / channel_width_start if channel_width_start > 0 else 1
            
            # === 2. 形态识别 ===
            pattern = "未识别"
            pattern_score = 0
            
            # 计算高点和低点序列
            highs = recent['High'].values
            lows = recent['Low'].values
            
            # 识别趋势
            high_trend = self._calculate_trend(highs)
            low_trend = self._calculate_trend(lows)
            
            # 三角形整理识别
            if channel_convergence < 0.7:  # 通道收敛
                if high_trend < -0.3 and low_trend > 0.3:
                    # 高点下降，低点上升 = 对称三角形
                    pattern = "对称三角形"
                    pattern_score = 18
                elif abs(high_trend) < 0.2 and low_trend > 0.3:
                    # 高点持平，低点上升 = 上升三角形（看涨）
                    pattern = "上升三角形"
                    pattern_score = 22
                elif high_trend < -0.3 and abs(low_trend) < 0.2:
                    # 高点下降，低点持平 = 下降三角形（看跌）
                    pattern = "下降三角形"
                    pattern_score = 15
            
            # 旗形整理识别
            elif 0.3 < abs(slope) < 1.5:  # 有一定斜率
                if slope < 0 and self._is_flag_pattern(recent, direction='bull'):
                    pattern = "看涨旗形"
                    pattern_score = 20
                elif slope > 0 and self._is_flag_pattern(recent, direction='bear'):
                    pattern = "看跌旗形"
                    pattern_score = 12
            
            # 楔形识别
            elif channel_convergence < 0.85:
                if slope > 0 and high_trend > 0 and low_trend > 0:
                    # 上升楔形（看跌）
                    pattern = "上升楔形"
                    pattern_score = 10
                elif slope < 0 and high_trend < 0 and low_trend < 0:
                    # 下降楔形（看涨）
                    pattern = "下降楔形"
                    pattern_score = 18
            
            # 横盘整理（兜底）
            price_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].mean()
            if pattern == "未识别" and price_range < 0.12:
                pattern = "横盘整理"
                pattern_score = 12
            
            # === 3. 价格位置评分 ===
            position_score = 0
            recent_high = recent['High'].max()
            price_to_high = current_price / recent_high
            
            if price_to_high >= 0.95:
                position_score = 10
                position_status = "接近高点"
            elif price_to_high >= 0.88:
                position_score = 7
                position_status = "上部区域"
            elif price_to_high >= 0.75:
                position_score = 4
                position_status = "中部区域"
            else:
                position_score = 1
                position_status = "下部区域"
            
            # === 4. 突破概率评估 ===
            breakout_probability = self._estimate_breakout_probability(
                pattern, channel_position, channel_convergence
            )
            
            total_score = min(pattern_score + position_score, 30)
            
            return {
                "score": total_score,
                "pattern": pattern,
                "pattern_score": pattern_score,
                "position_score": position_score,
                "channel_position": round(channel_position, 3),
                "channel_convergence": round(channel_convergence, 3),
                "price_to_high": round(price_to_high, 3),
                "position_status": position_status,
                "slope": round(slope, 4),
                "r_squared": round(r_squared, 3),
                "breakout_probability": round(breakout_probability, 2),
                "details": {
                    "regression_slope_pct": round(slope * 100, 2),
                    "high_trend": round(high_trend, 3),
                    "low_trend": round(low_trend, 3)
                }
            }
            
        except Exception as e:
            return {"score": 0, "pattern": "分析失败", "details": {"error": str(e)}}
    
    def _calculate_trend(self, series: np.ndarray) -> float:
        """计算序列趋势强度和方向"""
        if len(series) < 5:
            return 0
        
        x = np.arange(len(series))
        slope, _, _ = _linear_regression(x, series)
        
        # 归一化斜率
        mean_val = np.mean(series)
        normalized_slope = slope / mean_val if mean_val != 0 else 0
        
        return normalized_slope * 10  # 放大以便比较
    
    def _is_flag_pattern(self, hist: pd.DataFrame, direction: str) -> bool:
        """判断是否为旗形整理"""
        # 简化判断：前期有明显趋势，近期横盘或轻微回调
        if len(hist) < 20:
            return False
        
        early = hist.iloc[:10]
        late = hist.iloc[-10:]
        
        early_change = (early['Close'].iloc[-1] - early['Close'].iloc[0]) / early['Close'].iloc[0]
        late_range = (late['High'].max() - late['Low'].min()) / late['Close'].mean()
        
        if direction == 'bull':
            # 看涨旗形：前期上涨，近期小幅回调/横盘
            return early_change > 0.08 and late_range < 0.08
        else:
            # 看跌旗形：前期下跌，近期小幅反弹/横盘
            return early_change < -0.08 and late_range < 0.08
    
    def _estimate_breakout_probability(
        self, 
        pattern: str, 
        channel_position: float,
        channel_convergence: float
    ) -> float:
        """估算突破概率"""
        base_prob = 0.5
        
        # 形态加成
        pattern_probs = {
            "上升三角形": 0.25,
            "下降楔形": 0.20,
            "看涨旗形": 0.18,
            "对称三角形": 0.10,
            "横盘整理": 0.08,
            "下降三角形": -0.15,
            "上升楔形": -0.12,
            "看跌旗形": -0.10
        }
        
        base_prob += pattern_probs.get(pattern, 0)
        
        # 通道位置加成
        if channel_position > 0.8:
            base_prob += 0.15
        elif channel_position > 0.6:
            base_prob += 0.08
        
        # 收敛程度加成
        if channel_convergence < 0.5:
            base_prob += 0.10
        
        return min(max(base_prob, 0.1), 0.95)
    
    def _score_volume_setup(
        self, 
        hist: pd.DataFrame, 
        current_volume: float
    ) -> Dict[str, Any]:
        """成交量模式评分"""
        try:
            vol_ma_20 = hist['Volume_MA_20'].iloc[-1] if 'Volume_MA_20' in hist.columns else \
                       hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            if pd.isna(vol_ma_20) or vol_ma_20 <= 0:
                return {"score": 0, "status": "数据无效"}
            
            vol_ratio = current_volume / vol_ma_20
            score = 0
            status = ""
            
            # 温和放量（最佳）
            if self._config.volume_expansion_min <= vol_ratio <= self._config.volume_expansion_max:
                score = 18
                status = "温和放量"
            elif 1.0 <= vol_ratio < self._config.volume_expansion_min:
                score = 14
                status = "略增量"
            elif vol_ratio < self._config.volume_contraction_threshold:
                score = 8
                status = "缩量"
            elif self._config.volume_contraction_threshold <= vol_ratio < 1.0:
                score = 10
                status = "量偏低"
            else:
                score = 5
                status = "量偏高"
            
            # 成交量趋势
            if len(hist) >= 10:
                vol_early = hist['Volume'].iloc[-10:-5].mean()
                vol_recent = hist['Volume'].iloc[-5:].mean()
                
                if vol_recent > vol_early * 1.3:
                    score += 5
                    trend = "放量趋势"
                elif vol_recent > vol_early:
                    score += 2
                    trend = "量能上升"
                else:
                    trend = "量能平稳"
            else:
                trend = "数据不足"
            
            # 缩量后放量模式
            volume_pattern = ""
            if len(hist) >= 15:
                vol_early = hist['Volume'].iloc[-15:-10].mean()
                vol_mid = hist['Volume'].iloc[-10:-5].mean()
                vol_recent = hist['Volume'].iloc[-5:].mean()
                
                if vol_mid < vol_early * 0.7 and vol_recent > vol_mid * 1.4:
                    score += 5
                    volume_pattern = "缩量后放量"
            
            return {
                "score": min(score, 25),
                "vol_ratio": round(vol_ratio, 2),
                "status": status,
                "trend": trend,
                "volume_pattern": volume_pattern
            }
            
        except Exception as e:
            return {"score": 0, "status": "分析失败", "details": {"error": str(e)}}
    
    def _score_volatility_squeeze_enhanced(
        self, 
        hist: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        增强的波动率压缩评分 - TTM Squeeze
        
        TTM Squeeze: 布林带收窄至肯特纳通道内部
        
        Returns:
            包含 score, squeeze_status, details 的字典
        """
        try:
            # === 1. 计算布林带 ===
            period = 20
            std_dev = 2.0
            
            if 'BB_Upper' in hist.columns:
                bb_upper = hist['BB_Upper'].iloc[-1]
                bb_lower = hist['BB_Lower'].iloc[-1]
                bb_middle = hist['BB_Middle'].iloc[-1]
            else:
                bb_middle = hist['Close'].rolling(period).mean().iloc[-1]
                std = hist['Close'].rolling(period).std().iloc[-1]
                bb_upper = bb_middle + std_dev * std
                bb_lower = bb_middle - std_dev * std
            
            if pd.isna(bb_middle):
                return {"score": 0, "squeeze_status": "数据不足"}
            
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # === 2. 计算肯特纳通道 ===
            # Keltner Channel = EMA ± ATR × multiplier
            atr_period = 14
            kc_multiplier = 1.5
            
            if 'ATR_14' in hist.columns:
                atr = hist['ATR_14'].iloc[-1]
            else:
                tr = pd.DataFrame({
                    'hl': hist['High'] - hist['Low'],
                    'hc': abs(hist['High'] - hist['Close'].shift(1)),
                    'lc': abs(hist['Low'] - hist['Close'].shift(1))
                }).max(axis=1)
                atr = tr.rolling(atr_period).mean().iloc[-1]
            
            # EMA
            ema = hist['Close'].ewm(span=period).mean().iloc[-1]
            
            kc_upper = ema + kc_multiplier * atr
            kc_lower = ema - kc_multiplier * atr
            
            if pd.isna(atr) or pd.isna(ema):
                return {"score": 0, "squeeze_status": "ATR计算失败"}
            
            kc_width = (kc_upper - kc_lower) / ema
            
            # === 3. TTM Squeeze 判断 ===
            # 布林带在肯特纳通道内部 = 强挤压
            squeeze_status = "无挤压"
            score = 0
            
            if bb_upper < kc_upper and bb_lower > kc_lower:
                # 强挤压：布林带完全在肯特纳通道内
                squeeze_status = "强挤压"
                score = 25
                
                # 计算挤压程度
                squeeze_ratio = (kc_upper - kc_lower) / (bb_upper - bb_lower)
                if squeeze_ratio > 1.5:
                    score += 5
                    squeeze_status = "极强挤压"
                
            elif bb_upper < kc_upper or bb_lower > kc_lower:
                # 部分挤压
                squeeze_status = "中等挤压"
                score = 18
            else:
                # 布林带宽判断
                if bb_width < 0.10:
                    score = 12
                    squeeze_status = "轻度收窄"
                elif bb_width < 0.15:
                    score = 8
                    squeeze_status = "收窄"
            
            # === 4. 挤压趋势（是否在收窄中）===
            if len(hist) >= 20:
                bb_upper_prev = hist['BB_Upper'].iloc[-20] if 'BB_Upper' in hist.columns else None
                bb_lower_prev = hist['BB_Lower'].iloc[-20] if 'BB_Lower' in hist.columns else None
                
                if bb_upper_prev and bb_lower_prev:
                    bb_width_prev = (bb_upper_prev - bb_lower_prev) / hist['Close'].iloc[-20]
                    
                    if bb_width < bb_width_prev * 0.85:
                        score += 3
                        narrowing = True
                    else:
                        narrowing = False
                else:
                    narrowing = False
            else:
                narrowing = False
            
            # === 5. 波动率释放方向预测 ===
            momentum = 0
            if 'MACD' in hist.columns:
                macd = hist['MACD'].iloc[-1]
                macd_hist = hist['MACD_Hist'].iloc[-1] if 'MACD_Hist' in hist.columns else 0
                if macd_hist > 0:
                    momentum = 1  # 向上
                elif macd_hist < 0:
                    momentum = -1  # 向下
            
            return {
                "score": min(score, 30),
                "squeeze_status": squeeze_status,
                "bb_width": round(bb_width, 4),
                "kc_width": round(kc_width, 4),
                "bb_in_kc": bb_upper < kc_upper and bb_lower > kc_lower,
                "narrowing": narrowing,
                "momentum_direction": "向上" if momentum > 0 else "向下" if momentum < 0 else "中性",
                "details": {
                    "bb_upper": round(bb_upper, 2),
                    "bb_lower": round(bb_lower, 2),
                    "kc_upper": round(kc_upper, 2),
                    "kc_lower": round(kc_lower, 2)
                }
            }
            
        except Exception as e:
            return {"score": 0, "squeeze_status": "分析失败", "details": {"error": str(e)}}
    
    def _score_technical_setup(
        self, 
        hist: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """技术指标评分"""
        try:
            score = 0
            details = {}
            
            # RSI 评分
            rsi = hist['RSI_14'].iloc[-1] if 'RSI_14' in hist.columns else None
            
            if rsi is not None and not pd.isna(rsi):
                details['rsi'] = round(rsi, 2)
                
                if self._config.rsi_neutral_low <= rsi <= self._config.rsi_neutral_high:
                    score += 8
                    details['rsi_status'] = "中性区域"
                elif rsi < 35:
                    score += 5
                    details['rsi_status'] = "超卖区"
                elif rsi < 50:
                    score += 6
                    details['rsi_status'] = "偏弱"
                elif rsi < 70:
                    score += 4
                    details['rsi_status'] = "偏强"
                
                # RSI 趋势
                if len(hist) >= 5:
                    rsi_prev = hist['RSI_14'].iloc[-5]
                    if not pd.isna(rsi_prev) and rsi > rsi_prev:
                        score += 4
                        details['rsi_trend'] = "上升"
            
            # MACD 评分
            macd = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else None
            macd_signal = hist['MACD_Signal'].iloc[-1] if 'MACD_Signal' in hist.columns else None
            macd_hist = hist['MACD_Hist'].iloc[-1] if 'MACD_Hist' in hist.columns else None
            
            if macd is not None and macd_signal is not None:
                details['macd'] = round(macd, 4)
                
                if macd_hist is not None and macd_hist > 0:
                    score += 6
                    details['macd_status'] = "金叉状态"
                elif macd < macd_signal:
                    diff = abs(macd - macd_signal)
                    if diff < self._config.macd_threshold:
                        score += 5
                        details['macd_status'] = "即将金叉"
                    else:
                        score += 2
                        details['macd_status'] = "零轴下方"
                
                if abs(macd) < 0.05:
                    score += 3
                    details['macd_near_zero'] = True
            
            return {
                "score": min(score, 20),
                "details": details
            }
            
        except Exception as e:
            return {"score": 0, "details": {"error": str(e)}}
    
    def _score_relative_strength(
        self, 
        hist: pd.DataFrame, 
        benchmark: Optional[pd.DataFrame],
        current_price: float
    ) -> Dict[str, Any]:
        """
        弱转强信号评分
        
        分析股票相对大盘的表现：
        - 大盘下跌时抗跌
        - 大盘企稳时快速回升
        - 这是典型的"弱转强"特征
        
        Returns:
            包含 score, status, details 的字典
        """
        try:
            if benchmark is None or len(benchmark) < 20:
                # 没有大盘数据，使用内部强度分析
                return self._score_internal_strength(hist, current_price)
            
            # 计算相对强度
            period = 20
            stock_change = (current_price - hist['Close'].iloc[-period]) / hist['Close'].iloc[-period]
            benchmark_change = (benchmark['Close'].iloc[-1] - benchmark['Close'].iloc[-period]) / benchmark['Close'].iloc[-period]
            
            # 相对强度比
            rs_ratio = (1 + stock_change) / (1 + benchmark_change) if benchmark_change != -1 else 1
            
            score = 0
            status = ""
            
            # 相对强度评分
            if rs_ratio > 1.10:  # 跑赢大盘10%+
                score = 20
                status = "强势领涨"
            elif rs_ratio > 1.05:
                score = 16
                status = "相对强势"
            elif rs_ratio > 1.0:
                score = 12
                status = "跑赢大盘"
            elif rs_ratio > 0.95:
                score = 8
                status = "与大盘同步"
            else:
                score = 4
                status = "弱于大盘"
            
            # 弱转强特征检测
            weak_to_strong = False
            
            # 分段分析
            if len(hist) >= 20 and len(benchmark) >= 20:
                # 前10天
                stock_early = (hist['Close'].iloc[-20] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] if len(hist) >= 30 else 0
                bench_early = (benchmark['Close'].iloc[-20] - benchmark['Close'].iloc[-30]) / benchmark['Close'].iloc[-30] if len(benchmark) >= 30 else 0
                
                # 后10天
                stock_late = (current_price - hist['Close'].iloc[-10]) / hist['Close'].iloc[-10]
                bench_late = (benchmark['Close'].iloc[-1] - benchmark['Close'].iloc[-10]) / benchmark['Close'].iloc[-10]
                
                # 弱转强：前期大盘下跌时抗跌，后期大盘企稳时快速回升
                if bench_early < 0:  # 大盘前期下跌
                    rs_early = (1 + stock_early) / (1 + bench_early)
                    if rs_early > 1.02:  # 抗跌
                        if stock_late > 0 and stock_late > bench_late * 1.5:  # 快速回升
                            score += 10
                            weak_to_strong = True
                            status = "弱转强"
            
            return {
                "score": min(score, 25),
                "rs_ratio": round(rs_ratio, 3),
                "status": status,
                "weak_to_strong": weak_to_strong,
                "stock_change_pct": round(stock_change * 100, 2),
                "benchmark_change_pct": round(benchmark_change * 100, 2)
            }
            
        except Exception as e:
            return {"score": 8, "status": "分析失败", "details": {"error": str(e)}}
    
    def _score_internal_strength(
        self, 
        hist: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        内部强度分析（无大盘数据时使用）
        
        分析价格相对于自身均线的表现
        """
        try:
            score = 8  # 基础分
            status = "中等强度"
            
            # 价格相对均线位置
            ma_20 = hist['MA_20'].iloc[-1] if 'MA_20' in hist.columns else hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['MA_50'].iloc[-1] if 'MA_50' in hist.columns else (hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None)
            
            if ma_20 and not pd.isna(ma_20):
                price_to_ma20 = current_price / ma_20
                
                if price_to_ma20 > 1.05:
                    score += 8
                    status = "强势"
                elif price_to_ma20 > 1.0:
                    score += 5
                    status = "偏强"
            
            # 近期走势
            if len(hist) >= 10:
                recent_change = (current_price - hist['Close'].iloc[-10]) / hist['Close'].iloc[-10]
                if recent_change > 0.05:
                    score += 5
                elif recent_change > 0:
                    score += 2
            
            return {
                "score": min(score, 20),
                "status": status,
                "note": "无大盘数据，使用内部强度分析"
            }
            
        except Exception as e:
            return {"score": 8, "status": "分析失败"}
    
    def _calculate_ma_multiplier(self, ma_score: float) -> float:
        """
        计算均线粘合乘数
        
        均线粘合是前提条件，粘合程度影响最终评分
        """
        if ma_score >= 20:  # 强粘合
            return 1.0
        elif ma_score >= 15:
            return 0.9
        elif ma_score >= 10:
            return 0.75
        elif ma_score >= 5:
            return 0.5
        else:
            return 0.3
    
    def _calculate_confidence(
        self,
        total_score: float,
        ma_result: Dict,
        pattern_result: Dict,
        volatility_result: Dict
    ) -> float:
        """计算置信度"""
        base_confidence = total_score / 100.0
        
        # 均线粘合加成
        if ma_result.get('diverging', False):
            base_confidence += 0.05
        
        # 形态识别加成
        pattern = pattern_result.get('pattern', '')
        if pattern in ['上升三角形', '下降楔形', '看涨旗形']:
            base_confidence += 0.08
        
        # TTM Squeeze 加成
        if volatility_result.get('squeeze_status') == '强挤压':
            base_confidence += 0.1
        elif volatility_result.get('squeeze_status') == '极强挤压':
            base_confidence += 0.15
        
        return round(min(base_confidence, 0.95), 2)
    
    def _determine_signal_strength(self, total_score: float) -> str:
        """确定信号强度"""
        if total_score >= 75:
            return "强信号"
        elif total_score >= 60:
            return "中强信号"
        elif total_score >= 50:
            return "中等信号"
        elif total_score >= 40:
            return "弱信号"
        else:
            return "无信号"
    
    def _determine_alert_level(
        self,
        total_score: float,
        volatility_result: Dict,
        pattern_result: Dict
    ) -> str:
        """确定预警等级"""
        squeeze = volatility_result.get('squeeze_status', '')
        pattern = pattern_result.get('pattern', '')
        
        if total_score >= 70 and ('强挤压' in squeeze or '极强挤压' in squeeze):
            return "高预警"
        elif total_score >= 60:
            return "中预警"
        elif total_score >= 50:
            return "低预警"
        else:
            return "无预警"
    
    def _predict_breakout_direction(
        self,
        hist: pd.DataFrame,
        ma_result: Dict,
        pattern_result: Dict,
        rs_result: Dict
    ) -> str:
        """预测突破方向"""
        bullish_score = 0
        bearish_score = 0
        
        # 均线排列
        if ma_result.get('ma_alignment') == '多头排列':
            bullish_score += 2
        elif ma_result.get('ma_alignment') == '空头排列':
            bearish_score += 2
        
        # 形态方向
        pattern = pattern_result.get('pattern', '')
        bullish_patterns = ['上升三角形', '下降楔形', '看涨旗形']
        bearish_patterns = ['下降三角形', '上升楔形', '看跌旗形']
        
        if pattern in bullish_patterns:
            bullish_score += 3
        elif pattern in bearish_patterns:
            bearish_score += 3
        
        # 弱转强
        if rs_result.get('weak_to_strong', False):
            bullish_score += 2
        
        # 相对强度
        if rs_result.get('rs_ratio', 1) > 1.05:
            bullish_score += 1
        
        if bullish_score > bearish_score + 1:
            return "向上突破"
        elif bearish_score > bullish_score + 1:
            return "向下突破"
        else:
            return "方向不明"


def quick_breakout_setup_check(hist: pd.DataFrame, benchmark: pd.DataFrame = None) -> Dict:
    """便捷函数：快速检查启动前兆信号"""
    strategy = BreakoutSetupStrategy()
    context = StrategyContext(hist=hist, info={}, benchmark=benchmark, is_market_healthy=True)
    result = strategy.execute(context)
    
    return {
        "passed": result.passed,
        "confidence": result.confidence,
        "total_score": result.details.get('total_score', 0),
        "signal_strength": result.details.get('signal_strength', '无信号'),
        "alert_level": result.details.get('alert_level', '无预警'),
        "breakout_direction": result.details.get('breakout_direction', '方向不明'),
        "pattern": result.details.get('pattern_details', {}).get('pattern', '未识别'),
        "squeeze_status": result.details.get('volatility_details', {}).get('squeeze_status', 'N/A'),
        "details": result.details
    }