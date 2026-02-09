"""
信号评分器

多维度信号综合评分:
- 趋势信号评分
- 动量信号评分
- 量能确认评分
- 市场环境适配度

综合得分 = Σ(信号 × 权重)
"""

from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
import pandas as pd
import numpy as np
from typing import Dict, Optional


class SignalScorer(BaseStrategy):
    """
    信号评分器

    综合多个维度的信号计算最终得分
    """

    # 默认权重配置
    DEFAULT_WEIGHTS = {
        "trend_following": 0.25,      # 趋势信号
        "momentum_breakout": 0.20,    # 动量信号
        "volume_confirmation": 0.15,  # 量能确认
        "market_correction": 0.20,    # 市场回调用
        "sector_strength": 0.20       # 行业强度
    }

    def __init__(self, weights: Optional[Dict] = None):
        """
        初始化信号评分器

        Args:
            weights: 各信号权重，如果不提供则使用默认权重
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    @property
    def name(self) -> str:
        return "信号评分器"

    @property
    def category(self) -> str:
        return "信号分析"

    def execute(self, context: StrategyContext) -> StrategyResult:
        """
        执行信号评分

        Args:
            context: 策略上下文

        Returns:
            StrategyResult: 包含各维度得分和综合得分的策略结果
        """
        hist = context.hist
        info = context.info
        is_market_healthy = context.is_market_healthy

        if hist is None or len(hist) < 50:
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": "数据不足"}
            )

        # 计算各维度得分
        scores = self._calculate_all_scores(hist, info, is_market_healthy)

        # 计算加权综合得分
        final_score = self._calculate_weighted_score(scores)

        # 判断是否通过 (综合得分 >= 0.7 才通过)
        passed = final_score >= 0.7

        return StrategyResult(
            passed=passed,
            confidence=min(final_score, 0.95),
            details={
                "final_score": round(final_score, 3),
                "passed": passed,
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "weights": {k: round(v, 2) for k, v in self.weights.items()},
                "breakdown": self._get_score_breakdown(scores, final_score)
            }
        )

    def _calculate_all_scores(self, hist: pd.DataFrame, info: Dict,
                               is_market_healthy: bool) -> Dict[str, float]:
        """
        计算所有维度得分

        Returns:
            {
                "trend_following": 0.0 ~ 1.0,
                "momentum_breakout": 0.0 ~ 1.0,
                "volume_confirmation": 0.0 ~ 1.0,
                "market_correction": 0.0 ~ 1.0,
                "sector_strength": 0.0 ~ 1.0
            }
        """
        return {
            "trend_following": self._score_trend_following(hist),
            "momentum_breakout": self._score_momentum_breakout(hist),
            "volume_confirmation": self._score_volume_confirmation(hist),
            "market_correction": self._score_market_correction(hist, is_market_healthy),
            "sector_strength": self._score_sector_strength(hist, info)
        }

    def _score_trend_following(self, hist: pd.DataFrame) -> float:
        """
        趋势信号评分

        评分依据:
        - 价格位于多条均线上方
        - 均线呈多头排列
        - 趋势持续时间
        """
        close = hist['Close']
        periods = [5, 10, 20, 50, 200]

        score = 0.0

        # 1. 价格与均线位置 (最高40分)
        ma_scores = []
        for period in periods:
            if len(close) >= period:
                ma = close.rolling(window=period).mean().iloc[-1]
                if pd.notna(ma):
                    # 价格在均线上方得1分，均线在上方得0.5分
                    if close.iloc[-1] > ma:
                        ma_scores.append(1.0)
                    elif close.iloc[-1] > ma * 0.98:  # 接近均线
                        ma_scores.append(0.5)

        if ma_scores:
            score += min(len(ma_scores) / len(periods) * 0.4, 0.4)

        # 2. 均线排列 (最高30分)
        if len(ma_scores) >= 3:
            # 检查短期均线是否在长期均线上方
            valid_mas = [p for p in periods if len(close) >= p and pd.notna(close.rolling(window=p).mean().iloc[-1])]
            if len(valid_mas) >= 3:
                ma_values = [(p, close.rolling(window=p).mean().iloc[-1]) for p in valid_mas]
                ma_values.sort(key=lambda x: x[0])

                correctly_ordered = sum(
                    1 for i in range(len(ma_values) - 1)
                    if ma_values[i][1] < ma_values[i + 1][1]
                )
                score += min(correctly_ordered / (len(ma_values) - 1) * 0.3, 0.3)

        # 3. 趋势持续性 (最高30分)
        if len(close) >= 20:
            # 检查近20日趋势
            recent_trend = (close.iloc[-1] / close.iloc[-20]) - 1
            if recent_trend > 0.1:  # >10% 上涨
                score += 0.3
            elif recent_trend > 0.05:  # >5% 上涨
                score += 0.2
            elif recent_trend > 0:  # 正收益
                score += 0.1

        return min(score, 1.0)

    def _score_momentum_breakout(self, hist: pd.DataFrame) -> float:
        """
        动量信号评分

        评分依据:
        - 价格突破近期高点
        - RSI 位置
        - MACD 状态
        """
        close = hist['Close']
        score = 0.0

        # 1. 价格突破 (最高40分)
        if len(close) >= 20:
            high_20 = hist['High'].rolling(window=20).max().iloc[-1]
            current_price = close.iloc[-1]

            if current_price > high_20 * 1.02:  # 突破2%以上
                score += 0.4
            elif current_price > high_20 * 1.01:  # 突破1%以上
                score += 0.3
            elif current_price > high_20:  # 突破
                score += 0.2

        # 2. RSI 评分 (最高30分)
        if 'RSI_14' in hist.columns:
            rsi = hist['RSI_14'].iloc[-1]
            if pd.notna(rsi):
                if 50 <= rsi <= 70:  # 强势区域
                    score += 0.3
                elif 40 <= rsi <= 80:  # 中性偏强
                    score += 0.2
                elif rsi > 50:  # 50以上
                    score += 0.1

        # 3. MACD 评分 (最高30分)
        if 'MACD' in hist.columns and 'MACD_Signal' in hist.columns:
            macd = hist['MACD'].iloc[-1]
            signal = hist['MACD_Signal'].iloc[-1]
            macd_prev = hist['MACD'].iloc[-2] if len(hist) >= 2 else 0

            if pd.notna(macd) and pd.notna(signal):
                if macd > signal and macd_prev <= signal:  # 金叉
                    score += 0.3
                elif macd > signal:  # 多头
                    score += 0.2
                elif macd > 0:  # 正值
                    score += 0.1

        return min(score, 1.0)

    def _score_volume_confirmation(self, hist: pd.DataFrame) -> float:
        """
        量能确认评分

        评分依据:
        - 成交量突破
        - 量价配合
        - 成交量趋势
        """
        volume = hist['Volume']
        close = hist['Close']
        score = 0.0

        if len(volume) < 20:
            return 0.0

        # 1. 成交量突破 (最高40分)
        vol_current = volume.iloc[-1]
        vol_ma_20 = volume.rolling(window=20).mean().iloc[-1]

        if vol_current > vol_ma_20 * 2.5:
            score += 0.4
        elif vol_current > vol_ma_20 * 2.0:
            score += 0.3
        elif vol_current > vol_ma_20 * 1.5:
            score += 0.2
        elif vol_current > vol_ma_20:
            score += 0.1

        # 2. 量价配合 (最高30分)
        if len(close) >= 2:
            price_change = (close.iloc[-1] / close.iloc[-2]) - 1
            prev_volume = volume.iloc[-2] if volume.iloc[-2] > 0 else 1
            vol_change = (vol_current / prev_volume) - 1

            # 价涨量增 或 价跌量缩 (健康)
            if price_change > 0 and vol_change > 0:
                score += 0.3
            elif price_change > 0 and vol_change > -0.3:  # 价涨量微跌
                score += 0.2
            elif price_change < 0 and vol_change < 0:  # 价跌量缩
                score += 0.15

        # 3. 成交量趋势 (最高30分)
        vol_first_half = volume.iloc[-30:-15].mean() if len(volume) >= 30 else volume.mean()
        vol_second_half = volume.iloc[-15:].mean()

        if vol_second_half > vol_first_half * 1.2:
            score += 0.3
        elif vol_second_half > vol_first_half:
            score += 0.2
        elif vol_second_half > vol_first_half * 0.8:
            score += 0.1

        return min(score, 1.0)

    def _score_market_correction(self, hist: pd.DataFrame,
                                  is_market_healthy: bool) -> float:
        """
        市场回调用评分

        评分依据:
        - 大盘健康状态
        - 相对大盘表现
        - 市场环境适配
        """
        score = 0.0

        # 1. 大盘健康状态 (最高30分)
        if is_market_healthy:
            score += 0.3
        else:
            score += 0.0  # 弱势市场不加分

        # 2. 相对大盘表现 (最高50分)
        if len(hist) >= 20:
            stock_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1

            # 假设市场同期回报为 5% (简化处理)
            market_return = 0.05

            if stock_return > market_return * 2:  # 大幅跑赢市场
                score += 0.5
            elif stock_return > market_return * 1.5:
                score += 0.4
            elif stock_return > market_return:
                score += 0.3
            elif stock_return > 0:
                score += 0.1
            else:
                score += 0.0  # 逆势下跌扣分
                score -= 0.1

        # 3. 市场环境适配 (最高20分)
        # 强势市场中，动量策略评分更高
        if is_market_healthy:
            score += 0.2
        elif 'BBP' in hist.columns:
            # 震荡市场中，关注布林带挤压突破
            bbp = hist['BBP'].iloc[-1]
            if pd.notna(bbp):
                if 0.8 <= bbp <= 1.0:  # 布林带上轨
                    score += 0.15
                elif 0.5 <= bbp <= 0.8:
                    score += 0.1

        return max(0, min(score, 1.0))

    def _score_sector_strength(self, hist: pd.DataFrame, info: Dict) -> float:
        """
        行业强度评分

        评分依据:
        - 行业相对表现
        - 行业动量
        """
        score = 0.0

        # 1. 行业信息 (如果有)
        sector = info.get('sector', '')
        industry = info.get('industry', '')

        if sector or industry:
            # 假设获取行业数据 - 这里简化为基础分
            score += 0.3  # 有行业信息给基础分

        # 2. 个股技术形态 (最高70分)
        if len(hist) >= 20:
            # 检查关键技术形态

            # 形态A: 杯柄形态 (简化检测)
            high_20 = hist['High'].rolling(window=20).max().iloc[-1]
            high_10 = hist['High'].rolling(window=10).max().iloc[-1]

            if high_20 > high_10 * 1.05:  # 20日高点明显高于10日
                score += 0.3

            # 形态B: 整理后突破
            if 'BBP' in hist.columns:
                bbp = hist['BBP'].iloc[-1]
                bbp_prev = hist['BBP'].iloc[-5] if len(hist) >= 5 else bbp

                if pd.notna(bbp) and pd.notna(bbp_prev):
                    if bbp > 0.8 and bbp_prev < 0.5:  # 突破
                        score += 0.4

            # 形态C: 均线支撑
            if len(hist) >= 50:
                ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                if close_price := hist['Close'].iloc[-1]:
                    if close_price > ma_50 * 0.98:  # 价格在50日均线上方
                        score += 0.2

        return min(score, 1.0)

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        计算加权综合得分

        Args:
            scores: 各维度得分

        Returns:
            float: 加权综合得分 (0 ~ 1)
        """
        total = 0.0
        weight_sum = 0.0

        for signal_name, weight in self.weights.items():
            if signal_name in scores:
                total += scores[signal_name] * weight
                weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return total / weight_sum

    def _get_score_breakdown(self, scores: Dict[str, float], final_score: float) -> Dict:
        """
        获取得分分解 (用于详细分析)

        Returns:
            {
                "strengths": [...],      # 优势
                "weaknesses": [...],     # 劣势
                "recommendation": str    # 建议
            }
        """
        strengths = []
        weaknesses = []
        recommendation = ""

        # 找出优势和劣势
        for signal_name, score in scores.items():
            signal_label = {
                "trend_following": "趋势",
                "momentum_breakout": "动量",
                "volume_confirmation": "量能",
                "market_correction": "大盘",
                "sector_strength": "形态"
            }.get(signal_name, signal_name)

            if score >= 0.7:
                strengths.append(f"{signal_label}: {score:.0%}")
            elif score < 0.3:
                weaknesses.append(f"{signal_label}: {score:.0%}")

        # 生成建议
        if final_score >= 0.7:
            recommendation = "强烈看多 - 多维度信号共振"
        elif final_score >= 0.5:
            recommendation = "温和看多 - 主要信号积极"
        elif final_score >= 0.3:
            recommendation = "中性观望 - 信号不够明确"
        else:
            recommendation = "看空观望 - 多个指标偏弱"

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": recommendation
        }

    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        更新权重配置

        Args:
            weights: 新的权重配置
        """
        self.weights.update(weights)
        # 确保权重归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}


# 便捷函数：快速评分
def quick_score(hist: pd.DataFrame, info: Dict = None,
                market_healthy: bool = True) -> Dict:
    """
    便捷函数：快速计算信号得分

    Args:
        hist: OHLCV 数据
        info: 股票信息
        market_healthy: 大盘是否健康

    Returns:
        Dict: 评分结果
    """
    scorer = SignalScorer()
    context = StrategyContext(
        hist=hist,
        info=info or {},
        is_market_healthy=market_healthy
    )
    result = scorer.execute(context)

    return {
        "passed": result.passed,
        "score": result.details.get("final_score", 0) if result.details else 0,
        "scores": result.details.get("scores", {}) if result.details else {},
        "breakdown": result.details.get("breakdown", {}) if result.details else {}
    }
