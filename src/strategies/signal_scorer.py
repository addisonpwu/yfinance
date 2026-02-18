"""
信号评分器

多维度信号综合评分:
- 趋势信号评分
- 动量信号评分
- 量能确认评分
- 市场环境适配度

优化：
- 使用配置类管理参数
- 复用预计算技术指标
- 权重归一化
- 添加完整的错误处理和日志
"""
from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import SignalScorerConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np
from typing import Dict, Optional


class SignalScorer(BaseStrategy):
    """
    信号评分器

    综合多个维度的信号计算最终得分
    """
    
    def __init__(self, config: SignalScorerConfig = None):
        """
        初始化信号评分器

        Args:
            config: 策略配置，如果为 None 则从配置文件加载
        """
        self._config = config or strategy_config_manager.get_config('signal_scorer')
        if not isinstance(self._config, SignalScorerConfig):
            self._config = SignalScorerConfig()
        
        # 验证并归一化权重
        if not self._config.validate():
            self._config = SignalScorerConfig()
        self._config.normalize_weights()
        
        self._logger = get_analysis_logger()
    
    @property
    def name(self) -> str:
        return "信号评分器"
    
    @property
    def category(self) -> str:
        return "信号分析"
    
    @property
    def config(self) -> SignalScorerConfig:
        return self._config
    
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

        # 数据验证
        if hist is None or len(hist) < self._config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"数据不足，需要至少 {self._config.min_data_points} 天"}
            )

        try:
            # 计算各维度得分
            scores = self._calculate_all_scores(hist, info, is_market_healthy)

            # 计算加权综合得分
            final_score = self._calculate_weighted_score(scores)

            # 判断是否通过
            passed = final_score >= self._config.pass_threshold

            # 获取得分分解
            breakdown = self._get_score_breakdown(scores, final_score)

            # 日志记录
            if passed:
                self._logger.info(
                    f"[{self.name}] {info.get('symbol', 'Unknown')} 通过筛选 - "
                    f"综合得分: {final_score:.2f}, 优势: {breakdown.get('strengths', [])}"
                )

            return StrategyResult(
                passed=passed,
                confidence=min(final_score, 0.95),
                details={
                    "final_score": round(final_score, 3),
                    "passed": passed,
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "weights": {k: round(v, 2) for k, v in self._config.weights.items()},
                    "breakdown": breakdown
                }
            )
            
        except Exception as e:
            self._logger.error(f"[{self.name}] 执行策略时出错: {e}")
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={"reason": f"策略执行错误: {str(e)}"}
            )

    def _calculate_all_scores(
        self, 
        hist: pd.DataFrame, 
        info: Dict,
        is_market_healthy: bool
    ) -> Dict[str, float]:
        """
        计算所有维度得分

        Returns:
            各维度得分字典 (0.0 ~ 1.0)
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
        
        使用预计算的均线指标
        """
        close = hist['Close']
        score = 0.0
        
        # 1. 价格与均线位置 (40%)
        ma_periods = [5, 10, 20, 50, 200]
        ma_above_count = 0
        
        for period in ma_periods:
            ma_col = f'MA_{period}'
            if ma_col in hist.columns:
                ma = hist[ma_col].iloc[-1]
            else:
                if len(close) >= period:
                    ma = close.rolling(window=period).mean().iloc[-1]
                else:
                    continue
            
            if pd.notna(ma) and close.iloc[-1] > ma:
                ma_above_count += 1
        
        if len(ma_periods) > 0:
            score += (ma_above_count / len(ma_periods)) * 0.4

        # 2. 均线排列 (30%)
        ma_values = []
        for period in [5, 10, 20, 50]:
            ma_col = f'MA_{period}'
            if ma_col in hist.columns:
                ma = hist[ma_col].iloc[-1]
            elif len(close) >= period:
                ma = close.rolling(window=period).mean().iloc[-1]
            else:
                continue
            
            if pd.notna(ma):
                ma_values.append(ma)
        
        if len(ma_values) >= 3:
            correctly_ordered = sum(
                1 for i in range(len(ma_values) - 1)
                if ma_values[i] < ma_values[i + 1]
            )
            score += (correctly_ordered / (len(ma_values) - 1)) * 0.3

        # 3. 趋势持续性 (30%)
        if len(close) >= 20:
            try:
                recent_return = (close.iloc[-1] / close.iloc[-20]) - 1
                if recent_return > 0.1:
                    score += 0.3
                elif recent_return > 0.05:
                    score += 0.2
                elif recent_return > 0:
                    score += 0.1
            except:
                pass

        return min(score, 1.0)

    def _score_momentum_breakout(self, hist: pd.DataFrame) -> float:
        """
        动量信号评分
        
        使用预计算的 RSI 和 MACD 指标
        """
        close = hist['Close']
        score = 0.0

        # 1. 价格突破 (40%)
        if len(close) >= 20:
            high_20 = hist['High'].rolling(window=20).max().iloc[-1]
            current_price = close.iloc[-1]

            if pd.notna(high_20) and high_20 > 0:
                if current_price > high_20 * 1.02:
                    score += 0.4
                elif current_price > high_20 * 1.01:
                    score += 0.3
                elif current_price > high_20:
                    score += 0.2

        # 2. RSI 评分 (30%) - 使用预计算值
        if 'RSI_14' in hist.columns:
            rsi = hist['RSI_14'].iloc[-1]
            if pd.notna(rsi):
                if 50 <= rsi <= 70:
                    score += 0.3
                elif 40 <= rsi <= 80:
                    score += 0.2
                elif rsi > 50:
                    score += 0.1

        # 3. MACD 评分 (30%) - 使用预计算值
        if 'MACD' in hist.columns and 'MACD_Signal' in hist.columns:
            macd = hist['MACD'].iloc[-1]
            signal = hist['MACD_Signal'].iloc[-1]
            macd_prev = hist['MACD'].iloc[-2] if len(hist) >= 2 else 0

            if pd.notna(macd) and pd.notna(signal):
                if macd > signal and macd_prev <= signal:  # 金叉
                    score += 0.3
                elif macd > signal:
                    score += 0.2
                elif macd > 0:
                    score += 0.1

        return min(score, 1.0)

    def _score_volume_confirmation(self, hist: pd.DataFrame) -> float:
        """
        量能确认评分
        
        使用预计算的成交量均线
        """
        volume = hist['Volume']
        close = hist['Close']
        score = 0.0

        if len(volume) < 20:
            return 0.0

        # 1. 成交量突破 (40%)
        vol_current = volume.iloc[-1]
        
        if 'Volume_MA_20' in hist.columns:
            vol_ma_20 = hist['Volume_MA_20'].iloc[-1]
        else:
            vol_ma_20 = volume.rolling(window=20).mean().iloc[-1]

        if pd.notna(vol_ma_20) and vol_ma_20 > 0:
            vol_ratio = vol_current / vol_ma_20
            if vol_ratio > 2.5:
                score += 0.4
            elif vol_ratio > 2.0:
                score += 0.3
            elif vol_ratio > 1.5:
                score += 0.2
            elif vol_ratio > 1.0:
                score += 0.1

        # 2. 量价配合 (30%)
        if len(close) >= 2:
            price_change = (close.iloc[-1] / close.iloc[-2]) - 1
            prev_volume = volume.iloc[-2] if volume.iloc[-2] > 0 else 1
            vol_change = (vol_current / prev_volume) - 1

            if price_change > 0 and vol_change > 0:
                score += 0.3
            elif price_change > 0 and vol_change > -0.3:
                score += 0.2
            elif price_change < 0 and vol_change < 0:
                score += 0.15

        # 3. 成交量趋势 (30%)
        if len(volume) >= 30:
            vol_first_half = volume.iloc[-30:-15].mean()
            vol_second_half = volume.iloc[-15:].mean()

            if pd.notna(vol_first_half) and pd.notna(vol_second_half) and vol_first_half > 0:
                if vol_second_half > vol_first_half * 1.2:
                    score += 0.3
                elif vol_second_half > vol_first_half:
                    score += 0.2
                elif vol_second_half > vol_first_half * 0.8:
                    score += 0.1

        return min(score, 1.0)

    def _score_market_correction(
        self, 
        hist: pd.DataFrame,
        is_market_healthy: bool
    ) -> float:
        """
        市场回调用评分
        """
        score = 0.0

        # 1. 大盘健康状态 (30%)
        if is_market_healthy:
            score += 0.3

        # 2. 相对大盘表现 (50%)
        if len(hist) >= 20:
            try:
                stock_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1
                market_return = 0.05  # 简化处理

                if stock_return > market_return * 2:
                    score += 0.5
                elif stock_return > market_return * 1.5:
                    score += 0.4
                elif stock_return > market_return:
                    score += 0.3
                elif stock_return > 0:
                    score += 0.1
                else:
                    score = max(0, score - 0.1)
            except:
                pass

        # 3. 市场环境适配 (20%)
        if is_market_healthy:
            score += 0.2
        elif 'BBP' in hist.columns:
            bbp = hist['BBP'].iloc[-1]
            if pd.notna(bbp):
                if 0.8 <= bbp <= 1.0:
                    score += 0.15
                elif 0.5 <= bbp <= 0.8:
                    score += 0.1

        return max(0, min(score, 1.0))

    def _score_sector_strength(self, hist: pd.DataFrame, info: Dict) -> float:
        """
        行业强度评分
        """
        score = 0.0

        # 1. 行业信息 (30%)
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        if sector or industry:
            score += 0.3

        # 2. 技术形态 (70%)
        if len(hist) >= 20:
            # 形态A: 杯柄形态简化检测
            try:
                high_20 = hist['High'].rolling(window=20).max().iloc[-1]
                high_10 = hist['High'].rolling(window=10).max().iloc[-1]

                if pd.notna(high_20) and pd.notna(high_10):
                    if high_20 > high_10 * 1.05:
                        score += 0.3
            except:
                pass

            # 形态B: 布林带突破
            if 'BBP' in hist.columns:
                bbp = hist['BBP'].iloc[-1]
                bbp_prev = hist['BBP'].iloc[-5] if len(hist) >= 5 else bbp

                if pd.notna(bbp) and pd.notna(bbp_prev):
                    if bbp > 0.8 and bbp_prev < 0.5:
                        score += 0.4

            # 形态C: 均线支撑
            if len(hist) >= 50:
                if 'MA_50' in hist.columns:
                    ma_50 = hist['MA_50'].iloc[-1]
                else:
                    ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                
                if pd.notna(ma_50) and hist['Close'].iloc[-1] > ma_50 * 0.98:
                    score += 0.2

        return min(score, 1.0)

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        计算加权综合得分

        Args:
            scores: 各维度得分

        Returns:
            加权综合得分 (0 ~ 1)
        """
        total = 0.0
        weight_sum = 0.0

        for signal_name, weight in self._config.weights.items():
            if signal_name in scores:
                total += scores[signal_name] * weight
                weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return total / weight_sum

    def _get_score_breakdown(self, scores: Dict[str, float], final_score: float) -> Dict:
        """
        获取得分分解
        """
        strengths = []
        weaknesses = []
        
        signal_labels = {
            "trend_following": "趋势",
            "momentum_breakout": "动量",
            "volume_confirmation": "量能",
            "market_correction": "大盘",
            "sector_strength": "形态"
        }

        for signal_name, score in scores.items():
            label = signal_labels.get(signal_name, signal_name)

            if score >= 0.7:
                strengths.append(f"{label}: {score:.0%}")
            elif score < 0.3:
                weaknesses.append(f"{label}: {score:.0%}")

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
        self._config.weights.update(weights)
        self._config.normalize_weights()


def quick_score(
    hist: pd.DataFrame, 
    info: Dict = None,
    market_healthy: bool = True
) -> Dict:
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