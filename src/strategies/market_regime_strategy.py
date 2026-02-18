"""
市场环境识别策略

识别市场趋势环境:
- trending: 趋势市场 (适合动量策略)
- mean_reverting: 震荡市场 (适合均值回归策略)
- volatile: 高波动市场 (谨慎操作)

增强功能：
- 整合宏观指标（VIX、美债收益率、收益率曲线、美元指数）
- 提高市场环境识别准确性
- 提供综合风险评估

优化：
- 使用配置类管理参数
- 添加完整的错误处理和日志
- 优化 ADX 计算
"""
from src.core.strategies.strategy import BaseStrategy, StrategyContext
from src.core.models.entities import StrategyResult
from src.strategies.strategy_config import MarketRegimeConfig, strategy_config_manager
from src.utils.logger import get_analysis_logger
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class MarketRegimeStrategy(BaseStrategy):
    """
    市场环境识别策略
    
    整合：
    1. 技术指标分析（趋势强度、波动率、ADX）
    2. 宏观指标分析（VIX、美债收益率、收益率曲线、美元指数）
    """

    def __init__(self, config: MarketRegimeConfig = None, use_macro: bool = True):
        """
        初始化策略
        
        Args:
            config: 策略配置，如果为 None 则从配置文件加载
            use_macro: 是否启用宏观指标分析
        """
        self._config = config or strategy_config_manager.get_config('market_regime')
        if not isinstance(self._config, MarketRegimeConfig):
            self._config = MarketRegimeConfig()
        
        # 验证配置
        if not self._config.validate():
            self._config = MarketRegimeConfig()
        
        self._use_macro = use_macro
        self._logger = get_analysis_logger()
        
        # 宏观指标服务（延迟加载）
        self._macro_service = None
    
    @property
    def name(self) -> str:
        return "市场环境识别"

    @property
    def category(self) -> str:
        return "市场分析"
    
    @property
    def config(self) -> MarketRegimeConfig:
        return self._config

    def execute(self, context: StrategyContext) -> StrategyResult:
        """
        执行市场环境分析

        Args:
            context: 策略上下文 (包含 hist, info 等)

        Returns:
            StrategyResult: 包含市场环境信息的策略结果
            注意：此策略用于大盘环境分析，对个股总是返回 passed=False
        """
        hist = context.hist

        # 数据验证
        if hist is None or len(hist) < self._config.min_data_points:
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={
                    "regime": "unknown",
                    "reason": "数据不足",
                    "note": "市场环境识别策略不对个股进行筛选"
                }
            )

        try:
            # 计算技术指标市场环境
            regime_info = self._analyze_market_regime(hist)
            
            # 整合宏观指标
            if self._use_macro:
                macro_info = self._get_macro_analysis()
                regime_info = self._integrate_macro_indicators(regime_info, macro_info)

            return StrategyResult(
                passed=False,  # 此策略不对个股进行筛选
                confidence=regime_info["confidence"],
                details={
                    **regime_info,
                    "note": "市场环境识别策略不对个股进行筛选，仅用于获取市场环境信息"
                }
            )
            
        except Exception as e:
            self._logger.error(f"[{self.name}] 执行策略时出错: {e}")
            return StrategyResult(
                passed=False,
                confidence=0.0,
                details={
                    "regime": "unknown",
                    "reason": f"策略执行错误: {str(e)}",
                    "note": "市场环境识别策略不对个股进行筛选"
                }
            )

    def _get_macro_service(self):
        """获取宏观指标服务（延迟加载）"""
        if self._macro_service is None:
            try:
                from src.data.external.macro_indicators import get_macro_service
                self._macro_service = get_macro_service()
            except ImportError:
                self._logger.warning("宏观指标服务未安装，跳过宏观分析")
                return None
        return self._macro_service

    def _get_macro_analysis(self) -> Dict:
        """获取宏观指标分析结果"""
        service = self._get_macro_service()
        if service is None:
            return {}
        
        try:
            result = service.analyze_all()
            return result.to_dict()
        except Exception as e:
            self._logger.error(f"获取宏观指标失败: {e}")
            return {}

    def _integrate_macro_indicators(self, regime_info: Dict, macro_info: Dict) -> Dict:
        """
        整合宏观指标到市场环境分析
        
        Args:
            regime_info: 技术指标分析结果
            macro_info: 宏观指标分析结果
            
        Returns:
            整合后的市场环境信息
        """
        if not macro_info:
            return regime_info
        
        # 获取宏观风险评分
        macro_risk_score = macro_info.get("risk_score", 50)
        macro_sentiment = macro_info.get("sentiment", "neutral")
        
        # 调整健康得分
        original_health = regime_info.get("health_score", 0.5)
        
        # 宏观风险调整因子
        if macro_risk_score >= 70:
            # 高风险环境，降低健康得分
            health_adjustment = -0.2
        elif macro_risk_score >= 55:
            health_adjustment = -0.1
        elif macro_risk_score <= 30:
            # 低风险环境，提高健康得分
            health_adjustment = 0.1
        else:
            health_adjustment = 0
        
        adjusted_health = np.clip(original_health + health_adjustment, 0, 1)
        
        # 整合宏观指标详情
        macro_indicators = macro_info.get("indicators", {})
        
        # 根据 VIX 调整波动率判断
        vix_data = macro_indicators.get("vix", {})
        if vix_data:
            vix_value = vix_data.get("current_value", 20)
            if vix_value > 30:
                # VIX 高时强制判断为高波动
                regime_info["volatility_level"] = "high"
            elif vix_value < 15 and regime_info["volatility_level"] == "high":
                # VIX 低但技术指标显示高波动时，需要重新评估
                pass
        
        # 根据收益率曲线调整趋势判断
        curve_data = macro_indicators.get("yield_curve", {})
        if curve_data:
            curve_trend = curve_data.get("trend", "flat")
            if curve_trend == "inverted":
                # 收益率曲线倒挂，可能预示趋势反转
                regime_info["yield_curve_warning"] = True
        
        # 更新市场环境
        original_regime = regime_info.get("regime", "unknown")
        
        # 如果宏观指标显示高风险，调整市场类型
        if macro_risk_score >= 70 and original_regime != "volatile":
            regime_info["regime"] = "volatile"
            regime_info["regime_adjustment_reason"] = "宏观风险指标显示高风险环境"
        
        # 更新结果
        regime_info.update({
            "health_score": round(adjusted_health, 2),
            "is_healthy": adjusted_health >= self._config.health_score_threshold,
            "macro_risk_score": macro_risk_score,
            "macro_sentiment": macro_sentiment,
            "macro_indicators": macro_indicators,
            "macro_summary": macro_info.get("analysis_summary", ""),
            "recommended_strategy": macro_info.get("recommended_strategy", "balanced"),
        })
        
        # 提高置信度（因为整合了更多数据）
        confidence_boost = 0.1 if macro_info.get("confidence", 0) > 0.5 else 0
        regime_info["confidence"] = min(regime_info.get("confidence", 0.5) + confidence_boost, 0.98)
        
        return regime_info

    def _analyze_market_regime(self, hist: pd.DataFrame) -> Dict:
        """
        分析市场环境

        Returns:
            市场环境信息字典
        """
        close = hist['Close']
        returns = close.pct_change(fill_method=None).dropna()

        # 1. 计算趋势强度
        trend_strength = self._calculate_trend_strength(hist)

        # 2. 计算波动率水平
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        volatility_level = self._classify_volatility(volatility)

        # 3. 判断市场类型
        regime = self._classify_regime(trend_strength, volatility, returns)

        # 4. 计算趋势方向
        trend_direction = self._get_trend_direction(hist)

        # 5. 计算市场健康得分
        health_score = self._calculate_health_score(
            trend_strength, volatility_level, trend_direction, returns
        )
        is_healthy = health_score >= self._config.health_score_threshold

        return {
            "regime": regime,
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "volatility_level": volatility_level,
            "volatility_pct": round(volatility * 100, 2),
            "is_healthy": is_healthy,
            "health_score": round(health_score, 2),
            "confidence": min(abs(trend_strength) + 0.3, 0.95),
            "regime_metrics": {
                "adx_approx": round(self._calculate_adx(hist), 2),
                "annualized_volatility": round(volatility * 100, 2),
                "sharpe_approx": round(self._calculate_sharpe_approx(returns), 2),
                "drawdown": round(self._calculate_max_drawdown(close) * 100, 2)
            }
        }

    def _calculate_trend_strength(self, hist: pd.DataFrame, period: int = 20) -> float:
        """
        计算趋势强度 (近似 ADX)

        Returns:
            float: -1 (强下跌趋势) ~ 0 (无趋势) ~ 1 (强上涨趋势)
        """
        close = hist['Close']
        
        # 尝试使用预计算的均线
        if 'MA_20' in hist.columns:
            sma_short = hist['MA_20']
        else:
            sma_short = close.rolling(window=period).mean()
        
        if 'MA_50' in hist.columns:
            sma_long = hist['MA_50']
        else:
            sma_long = close.rolling(window=period * 2).mean()

        # 价格与均线的位置
        price_position = (close - sma_long) / (sma_long + 1e-10)

        # 均线的斜率
        sma_short_prev = sma_short.shift(5)
        sma_slope = (sma_short - sma_short_prev) / (sma_short_prev + 1e-10)

        # 综合趋势强度
        trend_strength = np.clip(
            price_position.iloc[-1] * 2 + sma_slope.iloc[-1] * 50, 
            -1, 
            1
        )

        return round(trend_strength, 3)

    def _calculate_adx(self, hist: pd.DataFrame, period: int = 14) -> float:
        """
        计算 ADX (平均趋向指数)
        """
        try:
            high = hist['High']
            low = hist['Low']
            close = hist['Close']

            # +DM 和 -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # +DI 和 -DI
            tr_avg = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / (tr_avg + 1e-10))
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / (tr_avg + 1e-10))

            # DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

            # ADX
            adx = dx.rolling(window=period * 2).mean()

            return adx.iloc[-1] if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0
            
        except Exception:
            return 0

    def _classify_volatility(self, volatility: float) -> str:
        """
        分类波动率水平
        """
        if volatility < self._config.low_volatility_threshold:
            return "low"
        elif volatility < self._config.high_volatility_threshold:
            return "medium"
        else:
            return "high"

    def _classify_regime(
        self, 
        trend_strength: float, 
        volatility: float, 
        returns: pd.Series
    ) -> str:
        """
        分类市场类型
        """
        # 高波动
        if volatility > self._config.extreme_volatility_threshold:
            return "volatile"

        # 强趋势
        if abs(trend_strength) > self._config.trend_strength_threshold:
            return "trending"

        # 低波动+无明显趋势 = 震荡
        if volatility < self._config.high_volatility_threshold:
            return "mean_reverting"

        return "mean_reverting"

    def _get_trend_direction(self, hist: pd.DataFrame, periods: list = None) -> str:
        """
        判断趋势方向
        """
        periods = periods or [5, 10, 20]
        close = hist['Close']
        current_price = close.iloc[-1]

        up_count = 0
        for period in periods:
            if len(close) >= period:
                # 使用预计算的均线
                ma_col = f'MA_{period}'
                if ma_col in hist.columns:
                    ma = hist[ma_col].iloc[-1]
                else:
                    ma = close.rolling(window=period).mean().iloc[-1]
                
                if pd.notna(ma):
                    if current_price > ma:
                        up_count += 1
                    elif current_price < ma:
                        up_count -= 1

        if up_count > 0:
            return "up"
        elif up_count < 0:
            return "down"
        else:
            return "neutral"

    def _calculate_health_score(
        self, 
        trend_strength: float, 
        volatility_level: str,
        trend_direction: str, 
        returns: pd.Series
    ) -> float:
        """
        计算市场健康得分
        """
        score = 0.5  # 基础分

        # 趋势强度贡献
        score += trend_strength * 0.2

        # 波动率惩罚
        if volatility_level == "high":
            score -= 0.2
        elif volatility_level == "low":
            score += 0.1

        # 方向奖励
        if trend_direction == "up":
            score += 0.1
        elif trend_direction == "down":
            score -= 0.1

        # 近期收益贡献
        if len(returns) >= 20:
            try:
                recent_return = returns.tail(20).sum()
                score += np.clip(recent_return * 0.5, -0.15, 0.15)
            except:
                pass

        return np.clip(score, 0, 1)

    def _calculate_sharpe_approx(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = None
    ) -> float:
        """
        近似计算夏普比率 (年化)
        """
        risk_free_rate = risk_free_rate or self._config.risk_free_rate
        
        if len(returns) < 10:
            return 0
        
        returns_std = returns.std()
        if returns_std == 0 or pd.isna(returns_std):
            return 0

        # 年化
        mean_return = returns.mean() * 252
        std_return = returns_std * np.sqrt(252)

        return (mean_return - risk_free_rate) / (std_return + 1e-10)

    def _calculate_max_drawdown(self, close: pd.Series) -> float:
        """
        计算最大回撤
        """
        try:
            cummax = close.cummax()
            drawdown = (close - cummax) / cummax
            return drawdown.min()
        except Exception:
            return 0


def get_market_regime(hist: pd.DataFrame) -> Dict:
    """
    便捷函数：快速获取市场环境

    Args:
        hist: OHLC 数据

    Returns:
        Dict: 市场环境信息
    """
    strategy = MarketRegimeStrategy()
    context = StrategyContext(hist=hist, info={}, is_market_healthy=True)
    result = strategy.execute(context)
    return result.details if result.details else {}