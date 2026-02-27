"""
启动捕捉策略 v1.0 (整合版)

整合"启动前兆策略"和"主力建仓策略"的核心优势，专门捕捉股票即将启动的综合策略。

核心特点：
1. 均线粘合作为乘法前提（技术形态基础）
2. 形态识别结合蓄势形态和挖坑形态
3. TTM Squeeze波动率压缩确认
4. CMF资金流 + 价平量缩识别主力动向
5. Beta系数 + 相对强度分析抗跌性
6. RSI/MACD技术指标辅助确认
7. 成交量模式验证筹码锁定

评分体系（7大维度）：
- 均线粘合 (15%): 多周期均线收敛程度
- 形态识别 (20%): 三角形/旗形/楔形/挖坑
- 波动率压缩 (15%): TTM Squeeze布林带收窄
- 资金信号 (20%): CMF资金流 + 价平量缩 + OBV
- 抗跌特征 (15%): Beta系数 + 相对强度
- 技术指标 (10%): RSI + MACD辅助确认
- 成交量模式 (5%): 缩量蓄势/温和放量
"""

from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import LaunchCaptureConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    线性回归（纯numpy实现，避免scipy依赖）
    Returns: (slope, intercept, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if len(y) > 0 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0, y_mean, 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # 计算R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return slope, intercept, r_squared


class LaunchCaptureStrategy(BaseStrategy):
    """
    启动捕捉策略 v1.0 (整合版)

    综合技术分析和资金流分析，捕捉股票即将启动的最佳时机：
    - 技术面：均线粘合、形态识别、波动率压缩
    - 资金面：CMF资金流、价平量缩、筹码锁定
    - 抗跌性：Beta系数、相对强度
    - 确认指标：RSI、MACD、成交量模式
    """

    def __init__(self, config: LaunchCaptureConfig = None):
        self._config = config or strategy_config_manager.get_config('launch_capture')
        if not isinstance(self._config, LaunchCaptureConfig):
            self._config = LaunchCaptureConfig()

        if not self._config.validate():
            self._config = LaunchCaptureConfig()

        self._logger = get_analysis_logger()

    @property
    def name(self) -> str:
        return "启动捕捉策略"

    @property
    def category(self) -> str:
        return "早期信号策略"

    @property
    def config(self) -> LaunchCaptureConfig:
        return self._config

    def execute(self, context: StrategyContext) -> StrategyResult:
        """执行启动捕捉策略检查"""
        hist = context.hist
        benchmark = context.benchmark

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

            # === 七大维度评分 ===

            # 1. 均线粘合评分（前提条件）
            ma_result = self._score_ma_convergence(hist, current_price)

            # 2. 形态识别评分
            pattern_result = self._score_price_pattern(hist, current_price)

            # 3. 波动率压缩评分（TTM Squeeze）
            volatility_result = self._score_volatility_squeeze(hist, current_price)

            # 4. 资金信号评分
            money_flow_result = self._score_money_flow(hist, current_price, current_volume)

            # 5. 抗跌特征评分
            resilience_result = self._score_resilience(hist, benchmark, current_price)

            # 6. 技术指标评分
            technical_result = self._score_technical_indicators(hist, current_price)

            # 7. 成交量模式评分
            volume_result = self._score_volume_pattern(hist, current_volume)

            # === 综合评分计算 ===

            # 基础加权总分
            base_score = (
                ma_result['score'] * self._config.ma_weight +
                pattern_result['score'] * self._config.pattern_weight +
                volatility_result['score'] * self._config.volatility_weight +
                money_flow_result['score'] * self._config.money_flow_weight +
                resilience_result['score'] * self._config.resilience_weight +
                technical_result['score'] * self._config.technical_weight +
                volume_result['score'] * self._config.volume_weight
            )

            # 均线粘合乘数（前提条件）
            ma_multiplier = self._calculate_ma_multiplier(ma_result)

            # 最终总分
            total_score = base_score * ma_multiplier

            # 通过条件
            passed = (
                total_score >= self._config.min_score and
                ma_result['passed']  # 均线粘合必须通过
            )

            # 计算置信度
            confidence = self._calculate_confidence(
                total_score, ma_result, pattern_result,
                money_flow_result, volatility_result
            )

            # 确定启动阶段
            launch_phase = self._determine_launch_phase(
                total_score, money_flow_result, pattern_result
            )

            # 预警等级
            alert_level = self._determine_alert_level(
                total_score, money_flow_result, pattern_result
            )

            if passed:
                self._logger.info(
                    f"[{self.name}] {context.info.get('symbol', 'Unknown')} 通过筛选 - "
                    f"总分: {total_score:.1f} (乘数: {ma_multiplier:.2f}), "
                    f"阶段: {launch_phase}, "
                    f"形态: {pattern_result.get('pattern', 'N/A')}, "
                    f"CMF: {money_flow_result.get('cmf', 'N/A')}"
                )

            return StrategyResult(
                passed=passed,
                confidence=confidence,
                details={
                    "total_score": round(total_score, 1),
                    "base_score": round(base_score, 1),
                    "ma_multiplier": round(ma_multiplier, 2),
                    "launch_phase": launch_phase,
                    "alert_level": alert_level,
                    "scores": {
                        "ma_convergence": ma_result['score'],
                        "price_pattern": pattern_result['score'],
                        "volatility_squeeze": volatility_result['score'],
                        "money_flow": money_flow_result['score'],
                        "resilience": resilience_result['score'],
                        "technical": technical_result['score'],
                        "volume": volume_result['score']
                    },
                    "ma_details": ma_result,
                    "pattern_details": pattern_result,
                    "volatility_details": volatility_result,
                    "money_flow_details": money_flow_result,
                    "resilience_details": resilience_result,
                    "technical_details": technical_result,
                    "volume_details": volume_result
                }
            )

        except Exception as e:
            self._logger.error(f"[{self.name}] 执行策略时出错: {e}")
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"策略执行错误: {str(e)}"}
            )

    # ==================== 七大维度评分方法 ====================

    def _score_ma_convergence(self, hist: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        均线粘合评分（前提条件）

        评估多周期均线的收敛程度，是启动前的典型技术特征。
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
            if ma_spread < self._config.ma_spread_extreme:  # 极度粘合 < 1.5%
                score = 25
                status = "极度粘合"
            elif ma_spread < self._config.ma_spread_strong:  # 强粘合 < 2.5%
                score = 22
                status = "强粘合"
            elif ma_spread < self._config.ma_spread_moderate:  # 粘合 < 4%
                score = 18
                status = "均线粘合"
            elif ma_spread < self._config.ma_spread_weak:  # 收敛 < 6%
                score = 12
                status = "均线收敛"
            elif ma_spread < 0.10:  # 接近
                score = 6
                status = "均线接近"
            else:
                score = 0
                status = "均线发散"

            # 检查均线多头排列
            alignment_bonus = 0
            if ma_5 > ma_10 > ma_20:
                alignment = "多头排列"
                alignment_bonus = 5
                score += alignment_bonus
            elif ma_5 < ma_10 < ma_20:
                alignment = "空头排列"
                alignment_bonus = 2
                score += alignment_bonus
            else:
                alignment = "交叉排列"

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

    def _score_price_pattern(self, hist: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        形态识别评分

        识别蓄势形态（三角形、旗形、楔形）和挖坑形态。
        """
        try:
            period = self._config.pattern_period
            if len(hist) < period:
                return {"score": 0, "pattern": "数据不足"}

            recent = hist.iloc[-period:].copy()

            # === 1. 线性回归价格通道 ===
            x = np.arange(len(recent))
            y = recent['Close'].values

            slope, intercept, r_squared = _linear_regression(x, y)

            # 计算通道
            regression_line = slope * x + intercept
            residuals = y - regression_line
            std_residual = np.std(residuals)

            upper_channel = regression_line + 2 * std_residual
            lower_channel = regression_line - 2 * std_residual

            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            channel_position = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0.5

            # 通道宽度变化
            channel_width_start = upper_channel[0] - lower_channel[0]
            channel_width_end = current_upper - current_lower
            channel_convergence = channel_width_end / channel_width_start if channel_width_start > 0 else 1

            # === 2. 形态识别 ===
            highs = recent['High'].values
            lows = recent['Low'].values

            high_trend = self._calculate_trend(highs)
            low_trend = self._calculate_trend(lows)

            pattern = "未识别"
            pattern_score = 0

            # 三角形整理识别
            if channel_convergence < self._config.channel_convergence:
                if high_trend < -self._config.triangle_threshold and low_trend > self._config.triangle_threshold:
                    pattern = "对称三角形"
                    pattern_score = 18
                elif abs(high_trend) < 0.2 and low_trend > self._config.triangle_threshold:
                    pattern = "上升三角形"
                    pattern_score = 22
                elif high_trend < -self._config.triangle_threshold and abs(low_trend) < 0.2:
                    pattern = "下降三角形"
                    pattern_score = 15

            # 旗形整理识别
            elif 0.3 < abs(slope) < 1.5:
                if slope < 0 and self._is_flag_pattern(recent, direction='bull'):
                    pattern = "看涨旗形"
                    pattern_score = 20
                elif slope > 0 and self._is_flag_pattern(recent, direction='bear'):
                    pattern = "看跌旗形"
                    pattern_score = 12

            # 楔形识别
            elif channel_convergence < 0.85:
                if slope > 0 and high_trend > 0 and low_trend > 0:
                    pattern = "上升楔形"
                    pattern_score = 10
                elif slope < 0 and high_trend < 0 and low_trend < 0:
                    pattern = "下降楔形"
                    pattern_score = 18

            # 挖坑形态识别
            pit_result = self._detect_pit_pattern(hist, current_price)
            if pit_result['is_pit'] and pattern_score < 20:
                pattern = "挖坑形态"
                pattern_score = max(pattern_score, pit_result['score'])

            # 横盘整理兜底
            if pattern == "未识别":
                price_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].mean()
                if price_range < 0.12:
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
                "pit_details": pit_result,
                "slope": round(slope, 4),
                "r_squared": round(r_squared, 3)
            }

        except Exception as e:
            return {"score": 0, "pattern": "分析失败", "error": str(e)}

    def _score_volatility_squeeze(self, hist: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        波动率压缩评分（TTM Squeeze）

        布林带收窄至肯特纳通道内部，预示大行情即将爆发。
        """
        try:
            period = self._config.ttm_bb_period
            std_dev = self._config.ttm_bb_std_dev

            # 计算布林带
            if 'BB_Upper' in hist.columns and 'BB_Lower' in hist.columns:
                bb_upper = hist['BB_Upper'].iloc[-1]
                bb_lower = hist['BB_Lower'].iloc[-1]
                bb_middle = hist['BB_Middle'].iloc[-1] if 'BB_Middle' in hist.columns else (bb_upper + bb_lower) / 2
            else:
                bb_middle = hist['Close'].rolling(period).mean().iloc[-1]
                std = hist['Close'].rolling(period).std().iloc[-1]
                bb_upper = bb_middle + std_dev * std
                bb_lower = bb_middle - std_dev * std

            if pd.isna(bb_middle):
                return {"score": 0, "squeeze_status": "数据不足"}

            bb_width = (bb_upper - bb_lower) / bb_middle

            # 计算肯特纳通道
            atr_period = self._config.ttm_atr_period
            kc_multiplier = self._config.ttm_kc_multiplier

            if 'ATR_14' in hist.columns:
                atr = hist['ATR_14'].iloc[-1]
            else:
                tr = pd.DataFrame({
                    'hl': hist['High'] - hist['Low'],
                    'hc': abs(hist['High'] - hist['Close'].shift(1)),
                    'lc': abs(hist['Low'] - hist['Close'].shift(1))
                }).max(axis=1)
                atr = tr.rolling(atr_period).mean().iloc[-1]

            ema = hist['Close'].ewm(span=period).mean().iloc[-1]
            kc_upper = ema + kc_multiplier * atr
            kc_lower = ema - kc_multiplier * atr

            if pd.isna(atr) or pd.isna(ema):
                return {"score": 0, "squeeze_status": "ATR计算失败"}

            # TTM Squeeze判断
            squeeze_status = "无挤压"
            score = 0

            if bb_upper < kc_upper and bb_lower > kc_lower:
                # 强挤压
                squeeze_status = "强挤压"
                score = 25

                squeeze_ratio = (kc_upper - kc_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 1
                if squeeze_ratio > 1.5:
                    score += 5
                    squeeze_status = "极强挤压"
            elif bb_upper < kc_upper or bb_lower > kc_lower:
                squeeze_status = "中等挤压"
                score = 18
            else:
                if bb_width < 0.10:
                    score = 12
                    squeeze_status = "轻度收窄"
                elif bb_width < 0.15:
                    score = 8
                    squeeze_status = "收窄"

            # 挤压趋势
            narrowing = False
            if len(hist) >= 20:
                bb_upper_prev = hist['BB_Upper'].iloc[-20] if 'BB_Upper' in hist.columns else None
                bb_lower_prev = hist['BB_Lower'].iloc[-20] if 'BB_Lower' in hist.columns else None

                if bb_upper_prev and bb_lower_prev:
                    bb_width_prev = (bb_upper_prev - bb_lower_prev) / hist['Close'].iloc[-20]
                    if bb_width < bb_width_prev * 0.85:
                        score += 3
                        narrowing = True

            # 动量方向
            momentum = 0
            if 'MACD' in hist.columns:
                macd_hist = hist['MACD_Hist'].iloc[-1] if 'MACD_Hist' in hist.columns else 0
                if not pd.isna(macd_hist):
                    momentum = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0

            return {
                "score": min(score, 30),
                "squeeze_status": squeeze_status,
                "bb_width": round(bb_width, 4),
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
            return {"score": 0, "squeeze_status": "分析失败", "error": str(e)}

    def _score_money_flow(self, hist: pd.DataFrame, current_price: float, current_volume: float) -> Dict[str, Any]:
        """
        资金信号评分

        结合CMF资金流、价平量缩、筹码锁定性分析。
        """
        try:
            period = self._config.cmf_period
            score = 0
            details = {}

            # === 1. CMF计算 ===
            cmf_series = self._calculate_cmf(hist, period)
            current_cmf = cmf_series.iloc[-1]
            prev_cmf = cmf_series.iloc[-2] if len(cmf_series) >= 2 else current_cmf

            if pd.isna(current_cmf):
                current_cmf = 0.0
            if pd.isna(prev_cmf):
                prev_cmf = 0.0

            details['cmf'] = round(float(current_cmf), 3)
            details['cmf_prev'] = round(float(prev_cmf), 3)

            # CMF评分
            if current_cmf > self._config.cmf_strong_threshold:
                score += 10
                details['cmf_status'] = "强势资金流入"
            elif current_cmf > 0:
                score += 6
                details['cmf_status'] = "资金流入"
            elif current_cmf > self._config.cmf_weak_threshold:
                score += 2
                details['cmf_status'] = "资金平衡"
            else:
                details['cmf_status'] = "资金流出"

            # CMF上穿0轴信号
            if prev_cmf < 0 and current_cmf > 0:
                score += 8
                details['cmf_cross_zero'] = True
                details['cmf_signal'] = "CMF上穿0轴，建仓结束"
            elif prev_cmf < 0.05 and current_cmf > 0.05:
                score += 4
                details['cmf_signal'] = "CMF走强"

            # === 2. 价平量缩分析 ===
            vol_30d = hist['Volume'].iloc[-30:].mean()
            vol_10d = hist['Volume'].iloc[-10:].mean()
            vol_ratio = vol_10d / vol_30d if vol_30d > 0 else 1

            period_high = hist['High'].iloc[-30:].max()
            period_low = hist['Low'].iloc[-30:].min()
            price_position = (current_price - period_low) / (period_high - period_low) if period_high != period_low else 0.5

            details['vol_ratio'] = round(vol_ratio, 2)
            details['price_position'] = round(price_position, 3)

            # 价平量缩评分
            if vol_ratio < self._config.vol_ratio_threshold and price_position > self._config.price_position_threshold:
                score += 10
                details['ending_signal'] = "价平量缩，筹码锁定"
            elif vol_ratio < 0.8 and price_position > 0.5:
                score += 6
                details['ending_signal'] = "量缩价稳"

            # 地量确认
            min_vol = hist['Volume'].iloc[-30:].min()
            min_vol_ratio = min_vol / vol_30d if vol_30d > 0 else 1
            if min_vol_ratio < self._config.ground_volume_ratio:
                score += 5
                details['ground_volume'] = True

            # === 3. OBV分析 ===
            obv = self._calculate_obv(hist)
            obv_current = obv.iloc[-1]
            obv_high = obv.iloc[-30:-1].max()

            if obv_current > obv_high:
                score += 5
                details['obv_new_high'] = True
                details['obv_status'] = "OBV同步新高"

            return {
                "score": min(score, 30),
                **details
            }

        except Exception as e:
            return {"score": 0, "error": str(e)}

    def _score_resilience(self, hist: pd.DataFrame, benchmark: Optional[pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """
        抗跌特征评分

        分析Beta系数和相对强度，识别抗跌性强的股票。
        """
        try:
            score = 0
            details = {}

            # === 1. Beta分析 ===
            beta = None
            if benchmark is not None and len(benchmark) >= 20:
                beta, beta_trend = self._calculate_beta(hist, benchmark, 30)

            if beta is not None:
                details['beta'] = round(beta, 3)
                details['beta_trend'] = round(beta_trend, 3) if beta_trend is not None else 0

                if beta < self._config.beta_low_threshold:
                    score += 10
                    details['beta_status'] = "低Beta，强抗跌"
                elif beta < 1.0:
                    score += 7
                    details['beta_status'] = "Beta<1，抗跌"
                elif beta < self._config.beta_high_threshold:
                    score += 4
                    details['beta_status'] = "Beta正常"

                # Beta趋势
                if beta_trend is not None and beta_trend < -0.1:
                    score += 5
                    details['beta_trend_status'] = "抗跌性增强"
            else:
                # 无大盘数据，使用内部稳定性
                recent_close = hist['Close'].iloc[-30:]
                price_volatility = recent_close.std() / recent_close.mean()
                details['price_volatility'] = round(price_volatility * 100, 2)

                if price_volatility < 0.02:
                    score += 8
                    details['volatility_status'] = "极低波动"
                elif price_volatility < 0.03:
                    score += 5
                    details['volatility_status'] = "低波动"

            # === 2. 价格稳定性 ===
            price_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) if len(hist) >= 30 else 0
            details['price_change_pct'] = round(price_change * 100, 2)

            if -0.05 <= price_change <= 0.03:
                score += 6
                details['price_trend'] = "横盘整理"
            elif -0.10 <= price_change < -0.05:
                score += 3
                details['price_trend'] = "小幅回调"

            # === 3. 相对强度分析 ===
            if benchmark is not None and len(benchmark) >= 20:
                stock_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
                market_return = (benchmark['Close'].iloc[-1] / benchmark['Close'].iloc[-20] - 1)
                relative_strength = stock_return - market_return

                details['relative_strength'] = round(relative_strength * 100, 2)

                if relative_strength > 0.05:
                    score += 5
                    details['rs_status'] = "强于大盘"
                elif relative_strength > 0:
                    score += 3
                    details['rs_status'] = "与大盘持平"

            return {
                "score": min(score, 25),
                **details
            }

        except Exception as e:
            return {"score": 0, "error": str(e)}

    def _score_technical_indicators(self, hist: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        技术指标评分

        RSI和MACD辅助确认。
        """
        try:
            score = 0
            details = {}

            # RSI评分
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

                # RSI趋势
                if len(hist) >= 5:
                    rsi_prev = hist['RSI_14'].iloc[-5]
                    if not pd.isna(rsi_prev) and rsi > rsi_prev:
                        score += 4
                        details['rsi_trend'] = "上升"

            # MACD评分
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

                if abs(macd) < 0.05:
                    score += 3
                    details['macd_near_zero'] = True

            return {
                "score": min(score, 20),
                **details
            }

        except Exception as e:
            return {"score": 0, "error": str(e)}

    def _score_volume_pattern(self, hist: pd.DataFrame, current_volume: float) -> Dict[str, Any]:
        """
        成交量模式评分

        识别缩量蓄势、温和放量等模式。
        """
        try:
            vol_ma_20 = hist['Volume_MA_20'].iloc[-1] if 'Volume_MA_20' in hist.columns else hist['Volume'].rolling(20).mean().iloc[-1]

            if pd.isna(vol_ma_20) or vol_ma_20 <= 0:
                return {"score": 0, "status": "数据无效"}

            vol_ratio = current_volume / vol_ma_20
            score = 0
            status = ""

            # 温和放量
            if self._config.volume_expansion_min <= vol_ratio <= self._config.volume_expansion_max:
                score = 15
                status = "温和放量"
            elif 1.0 <= vol_ratio < self._config.volume_expansion_min:
                score = 10
                status = "略增量"
            elif vol_ratio < self._config.volume_contraction:
                score = 8
                status = "缩量"
            elif vol_ratio < 1.0:
                score = 5
                status = "量偏低"
            else:
                score = 3
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

            return {
                "score": min(score, 20),
                "vol_ratio": round(vol_ratio, 2),
                "status": status,
                "trend": trend
            }

        except Exception as e:
            return {"score": 0, "status": "分析失败", "error": str(e)}

    # ==================== 辅助计算方法 ====================

    def _calculate_ma_multiplier(self, ma_result: Dict) -> float:
        """计算均线粘合乘数"""
        score = ma_result.get('score', 0)
        spread_pct = ma_result.get('ma_spread_pct', 100)

        if score >= 25:  # 极度粘合
            return 1.3
        elif score >= 22:  # 强粘合
            return 1.2
        elif score >= 18:  # 粘合
            return 1.1
        elif score >= 12:  # 收敛
            return 1.0
        elif score >= 6:  # 接近
            return 0.85
        else:
            return 0.7

    def _calculate_confidence(self, total_score: float, ma_result: Dict,
                              pattern_result: Dict, money_flow_result: Dict,
                              volatility_result: Dict) -> float:
        """计算置信度"""
        base_confidence = total_score / 100.0

        # 均线粘合加成
        if ma_result.get('ma_spread_pct', 100) < 2.5:
            base_confidence += 0.05

        # 形态确认加成
        if pattern_result.get('score', 0) >= 18:
            base_confidence += 0.05

        # CMF信号加成
        if money_flow_result.get('cmf_cross_zero', False):
            base_confidence += 0.08

        # TTM Squeeze加成
        if "强挤压" in volatility_result.get('squeeze_status', ''):
            base_confidence += 0.05

        return round(min(base_confidence, 0.95), 2)

    def _determine_launch_phase(self, total_score: float, money_flow_result: Dict,
                                 pattern_result: Dict) -> str:
        """确定启动阶段"""
        if total_score >= 75:
            if money_flow_result.get('cmf_cross_zero', False):
                return "即将启动"
            elif pattern_result.get('score', 0) >= 20:
                return "启动前夜"
            return "蓄势待发"
        elif total_score >= 60:
            return "启动准备期"
        else:
            return "观察期"

    def _determine_alert_level(self, total_score: float, money_flow_result: Dict,
                                pattern_result: Dict) -> str:
        """确定预警等级"""
        if total_score >= 75 and money_flow_result.get('cmf_cross_zero', False):
            return "高预警"
        elif total_score >= 65 and (money_flow_result.get('score', 0) >= 15 or
                                     pattern_result.get('score', 0) >= 18):
            return "中预警"
        elif total_score >= 60:
            return "低预警"
        else:
            return "无预警"

    def _calculate_trend(self, series: np.ndarray) -> float:
        """计算序列趋势强度"""
        if len(series) < 5:
            return 0

        x = np.arange(len(series))
        slope, _, _ = _linear_regression(x, series)
        mean_val = np.mean(series)

        return (slope / mean_val * 10) if mean_val != 0 else 0

    def _is_flag_pattern(self, hist: pd.DataFrame, direction: str) -> bool:
        """判断是否为旗形整理"""
        if len(hist) < 20:
            return False

        early = hist.iloc[:10]
        late = hist.iloc[-10:]

        early_change = (early['Close'].iloc[-1] - early['Close'].iloc[0]) / early['Close'].iloc[0]
        late_range = (late['High'].max() - late['Low'].min()) / late['Close'].mean()

        if direction == 'bull':
            return early_change > 0.08 and late_range < 0.08
        else:
            return early_change < -0.08 and late_range < 0.08

    def _detect_pit_pattern(self, hist: pd.DataFrame, current_price: float) -> Dict:
        """检测挖坑形态"""
        try:
            if len(hist) < 40:
                return {"is_pit": False, "score": 0}

            # 前期高点（坑沿）
            prev_period = hist.iloc[-40:-30]
            pit_period = hist.iloc[-30:]

            prev_high = prev_period['High'].max()
            pit_low = pit_period['Low'].min()

            # 坑深度
            pit_depth = (prev_high - pit_low) / prev_high if prev_high > 0 else 0

            # 是否形成挖坑
            is_pit = (self._config.pit_depth_min <= pit_depth <= self._config.pit_depth_max and
                      current_price > pit_low * 1.03)

            score = 0
            details = {
                "is_pit": is_pit,
                "pit_depth_pct": round(pit_depth * 100, 2),
                "prev_high": round(prev_high, 2),
                "pit_low": round(pit_low, 2)
            }

            if is_pit:
                score = 5

                # 站上坑沿
                if current_price >= prev_high:
                    score += 8
                    details["above_rim"] = True

                # 成交量确认
                vol_ma = hist['Volume'].iloc[-30:].mean()
                recent_vol = hist['Volume'].iloc[-5:].mean()
                vol_ratio = recent_vol / vol_ma if vol_ma > 0 else 1

                if 1.0 <= vol_ratio <= 1.5:
                    score += 5
                    details["vol_confirmation"] = "温和放量"

            return {"is_pit": is_pit, "score": score, **details}

        except Exception:
            return {"is_pit": False, "score": 0}

    def _calculate_cmf(self, hist: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算Chaikin Money Flow"""
        try:
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            volume = hist['Volume']

            hl_range = high - low
            hl_range = hl_range.replace(0, np.nan)

            clv = ((close - low) - (high - close)) / hl_range
            clv = clv.fillna(0)

            mfv = clv * volume
            cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

            return cmf

        except Exception:
            return pd.Series([0] * len(hist), index=hist.index)

    def _calculate_obv(self, hist: pd.DataFrame) -> pd.Series:
        """计算OBV指标"""
        # 使用 float 类型避免整数溢出
        obv = [0.0]
        for i in range(1, len(hist)):
            volume = float(hist['Volume'].iloc[i])
            if hist['Close'].iloc[i] > hist['Close'].iloc[i-1]:
                obv.append(obv[-1] + volume)
            elif hist['Close'].iloc[i] < hist['Close'].iloc[i-1]:
                obv.append(obv[-1] - volume)
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=hist.index)

    def _calculate_beta(self, hist: pd.DataFrame, benchmark: pd.DataFrame, period: int) -> Tuple[Optional[float], Optional[float]]:
        """计算Beta系数及其趋势"""
        try:
            stock_returns = hist['Close'].iloc[-period:].pct_change().dropna()
            market_returns = benchmark['Close'].iloc[-period:].pct_change().dropna()

            min_len = min(len(stock_returns), len(market_returns))
            if min_len < 10:
                return None, None

            stock_returns = stock_returns.iloc[-min_len:]
            market_returns = market_returns.iloc[-min_len:]

            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)

            current_beta = covariance / market_variance if market_variance > 0 else 1.0

            # Beta趋势
            half = min_len // 2
            if half >= 5:
                early_beta = np.cov(stock_returns.iloc[:half], market_returns.iloc[:half])[0, 1] / np.var(market_returns.iloc[:half])
                late_beta = np.cov(stock_returns.iloc[half:], market_returns.iloc[half:])[0, 1] / np.var(market_returns.iloc[half:])
                beta_trend = late_beta - early_beta
            else:
                beta_trend = 0

            return current_beta, beta_trend

        except Exception:
            return None, None


def quick_launch_check(hist: pd.DataFrame, benchmark: pd.DataFrame = None) -> Dict:
    """便捷函数：快速检查启动信号"""
    strategy = LaunchCaptureStrategy()
    context = StrategyContext(
        hist=hist,
        info={},
        benchmark=benchmark,
        is_market_healthy=True,
        market_return=0
    )
    result = strategy.execute(context)

    return {
        "passed": result.passed,
        "confidence": result.confidence,
        "total_score": result.details.get('total_score', 0),
        "launch_phase": result.details.get('launch_phase', '未知'),
        "alert_level": result.details.get('alert_level', '无预警'),
        "scores": result.details.get('scores', {}),
        "details": result.details
    }
