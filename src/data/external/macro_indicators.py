"""
宏观指标服务

获取和分析宏观经济指标，用于增强市场环境识别：
- VIX 指数（恐慌指数）
- 美债收益率（10年期、2年期）
- 美元指数
- 市场宽度指标

依赖：yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

from src.utils.logger import get_analysis_logger
from src.data.cache.cache_service import OptimizedCache


@dataclass
class MacroIndicator:
    """宏观指标数据结构"""
    name: str
    symbol: str
    current_value: float
    previous_value: Optional[float] = None
    change_pct: Optional[float] = None
    trend: str = "neutral"  # "up", "down", "neutral"
    signal: str = "neutral"  # "bullish", "bearish", "neutral"
    description: str = ""
    last_updated: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "change_pct": self.change_pct,
            "trend": self.trend,
            "signal": self.signal,
            "description": self.description,
            "last_updated": self.last_updated
        }


@dataclass
class MacroAnalysisResult:
    """宏观分析结果"""
    # 综合风险评分 (0-100，越高越风险)
    risk_score: float = 50.0
    
    # 市场情绪 ("fear", "greed", "neutral")
    sentiment: str = "neutral"
    
    # 建议策略类型
    recommended_strategy: str = "balanced"  # "aggressive", "defensive", "balanced"
    
    # 各指标详情
    indicators: Dict[str, MacroIndicator] = field(default_factory=dict)
    
    # 综合分析说明
    analysis_summary: str = ""
    
    # 置信度
    confidence: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "risk_score": self.risk_score,
            "sentiment": self.sentiment,
            "recommended_strategy": self.recommended_strategy,
            "indicators": {k: v.to_dict() for k, v in self.indicators.items()},
            "analysis_summary": self.analysis_summary,
            "confidence": self.confidence
        }


class MacroIndicatorsService:
    """
    宏观指标服务
    
    获取和分析关键宏观指标：
    1. VIX 指数 - 市场恐慌程度
    2. 美债收益率 - 利率环境
    3. 收益率曲线 - 经济周期信号
    4. 美元指数 - 全球资金流向
    """
    
    # 指标代码映射
    INDICATOR_SYMBOLS = {
        "vix": "^VIX",           # VIX 恐慌指数
        "tnx": "^TNX",           # 10年期美债收益率
        "fvx": "^FVX",           # 5年期美债收益率  
        "irx": "^IRX",           # 13周国债收益率
        "dxy": "DX-Y.NYB",       # 美元指数
        "spx": "^GSPC",          # S&P 500
        "ndq": "^NDX",           # 纳斯达克100
    }
    
    # VIX 区间定义
    VIX_LEVELS = {
        "complacency": (0, 12),     # 自满/低波动
        "normal": (12, 20),         # 正常
        "elevated": (20, 30),       # 升高
        "fear": (30, 40),           # 恐惧
        "panic": (40, 100),         # 恐慌
    }
    
    # 10年期美债收益率区间
    TNX_LEVELS = {
        "very_low": (0, 1.5),       # 极低（宽松环境）
        "low": (1.5, 2.5),          # 低
        "normal": (2.5, 4.0),       # 正常
        "high": (4.0, 5.0),         # 高
        "very_high": (5.0, 10),     # 极高（紧缩环境）
    }
    
    def __init__(self, cache_enabled: bool = True, cache_hours: int = 4):
        """
        初始化宏观指标服务
        
        Args:
            cache_enabled: 是否启用缓存
            cache_hours: 缓存有效期（小时）
        """
        self.logger = get_analysis_logger()
        self.cache = OptimizedCache(enabled=cache_enabled)
        self.cache_hours = cache_hours
        self._indicator_data: Dict[str, pd.DataFrame] = {}
    
    def fetch_indicator(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        获取单个指标数据
        
        Args:
            symbol: 指标代码
            period: 数据周期
            
        Returns:
            DataFrame 或 None
        """
        cache_key = f"macro_{symbol}_{period}"
        
        # 尝试从缓存获取
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, auto_adjust=True)
            
            if hist.empty:
                self.logger.warning(f"宏观指标 {symbol} 数据为空")
                return None
            
            # 缓存数据
            self.cache.set(cache_key, hist, ttl=self.cache_hours * 3600)
            self._indicator_data[symbol] = hist
            
            return hist
            
        except Exception as e:
            self.logger.error(f"获取宏观指标 {symbol} 失败: {e}")
            return None
    
    def get_vix_indicator(self) -> Optional[MacroIndicator]:
        """
        获取 VIX 指标分析
        
        VIX 解读：
        - < 12: 市场过度自满，可能预示回调风险
        - 12-20: 正常波动区间
        - 20-30: 市场担忧增加
        - > 30: 恐慌情绪，可能是买入机会
        """
        symbol = self.INDICATOR_SYMBOLS["vix"]
        hist = self.fetch_indicator(symbol, period="1mo")
        
        if hist is None or hist.empty:
            return None
        
        current = hist['Close'].iloc[-1]
        previous = hist['Close'].iloc[-2] if len(hist) > 1 else None
        change_pct = ((current - previous) / previous * 100) if previous else None
        
        # 确定信号
        if current < 12:
            signal = "cautious"  # 过度自满，谨慎
            description = "VIX 极低，市场过度乐观，可能存在回调风险"
        elif current < 20:
            signal = "neutral"
            description = "VIX 处于正常区间，市场情绪稳定"
        elif current < 30:
            signal = "opportunity"
            description = "VIX 升高，市场担忧增加，关注买入机会"
        else:
            signal = "contrarian_buy"  # 反向买入信号
            description = "VIX 极高，市场恐慌，可能是逆向买入时机"
        
        # 确定趋势
        ma_5 = hist['Close'].tail(5).mean()
        ma_10 = hist['Close'].tail(10).mean()
        if ma_5 > ma_10 * 1.05:
            trend = "up"  # 波动率上升
        elif ma_5 < ma_10 * 0.95:
            trend = "down"  # 波动率下降
        else:
            trend = "neutral"
        
        return MacroIndicator(
            name="VIX 恐慌指数",
            symbol=symbol,
            current_value=round(current, 2),
            previous_value=round(previous, 2) if previous else None,
            change_pct=round(change_pct, 2) if change_pct else None,
            trend=trend,
            signal=signal,
            description=description,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    def get_treasury_yield_indicator(self) -> Optional[MacroIndicator]:
        """
        获取 10 年期美债收益率分析
        
        收益率解读：
        - 上升: 紧缩预期，成长股承压
        - 下降: 宽松预期，利好股市
        - 倒挂（2年>10年）: 衰退信号
        """
        symbol = self.INDICATOR_SYMBOLS["tnx"]
        hist = self.fetch_indicator(symbol, period="3mo")
        
        if hist is None or hist.empty:
            return None
        
        current = hist['Close'].iloc[-1]
        previous = hist['Close'].iloc[-2] if len(hist) > 1 else None
        change_pct = ((current - previous) / previous * 100) if previous else None
        
        # 计算 20 日和 60 日均线
        ma_20 = hist['Close'].tail(20).mean()
        ma_60 = hist['Close'].tail(60).mean() if len(hist) >= 60 else ma_20
        
        # 确定趋势
        if current > ma_20 > ma_60:
            trend = "up"
            signal = "bearish"  # 收益率上升对股市偏空
            description = f"收益率上升趋势 (当前: {current:.2f}%)，紧缩预期增强，成长股可能承压"
        elif current < ma_20 < ma_60:
            trend = "down"
            signal = "bullish"  # 收益率下降对股市偏多
            description = f"收益率下降趋势 (当前: {current:.2f}%)，宽松预期利好股市"
        else:
            trend = "neutral"
            signal = "neutral"
            description = f"收益率横盘整理 (当前: {current:.2f}%)"
        
        return MacroIndicator(
            name="10年期美债收益率",
            symbol=symbol,
            current_value=round(current, 2),
            previous_value=round(previous, 2) if previous else None,
            change_pct=round(change_pct, 2) if change_pct else None,
            trend=trend,
            signal=signal,
            description=description,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    def get_yield_curve_indicator(self) -> Optional[MacroIndicator]:
        """
        获取收益率曲线分析（2年期 vs 10年期）
        
        收益率曲线解读：
        - 正常：10年 > 2年，经济健康
        - 平坦：利差缩小，增长放缓
        - 倒挂：2年 > 10年，衰退预警
        """
        # 获取 2 年期和 10 年期收益率
        tnx_hist = self.fetch_indicator(self.INDICATOR_SYMBOLS["tnx"], period="3mo")
        
        # 使用 5 年期作为短期代理（2年期数据可能不稳定）
        fvx_hist = self.fetch_indicator(self.INDICATOR_SYMBOLS["fvx"], period="3mo")
        
        if tnx_hist is None or fvx_hist is None:
            return None
        
        tnx_current = tnx_hist['Close'].iloc[-1]
        fvx_current = fvx_hist['Close'].iloc[-1]
        
        # 计算利差
        spread = tnx_current - fvx_current
        
        # 确定信号
        if spread > 0.5:
            signal = "bullish"
            trend = "steepening"
            description = f"收益率曲线正常 (10Y-5Y利差: {spread:.2f}%)，经济预期良好"
        elif spread > 0:
            signal = "neutral"
            trend = "flat"
            description = f"收益率曲线平坦 (10Y-5Y利差: {spread:.2f}%)，增长可能放缓"
        else:
            signal = "bearish"
            trend = "inverted"
            description = f"收益率曲线倒挂 (10Y-5Y利差: {spread:.2f}%)，衰退风险增加"
        
        return MacroIndicator(
            name="收益率曲线利差",
            symbol="10Y-5Y",
            current_value=round(spread, 3),
            previous_value=None,
            change_pct=None,
            trend=trend,
            signal=signal,
            description=description,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    def get_dxy_indicator(self) -> Optional[MacroIndicator]:
        """
        获取美元指数分析
        
        DXY 解读：
        - 强美元: 资金回流美国，新兴市场承压
        - 弱美元: 风险偏好上升，利好大宗商品和新兴市场
        """
        symbol = self.INDICATOR_SYMBOLS["dxy"]
        hist = self.fetch_indicator(symbol, period="3mo")
        
        if hist is None or hist.empty:
            return None
        
        current = hist['Close'].iloc[-1]
        previous = hist['Close'].iloc[-2] if len(hist) > 1 else None
        change_pct = ((current - previous) / previous * 100) if previous else None
        
        # 计算趋势
        ma_20 = hist['Close'].tail(20).mean()
        ma_60 = hist['Close'].tail(60).mean() if len(hist) >= 60 else ma_20
        
        if current > ma_20 > ma_60:
            trend = "up"
            signal = "cautious"  # 强美元可能压制风险资产
            description = f"美元走强 (DXY: {current:.2f})，资金回流美国，新兴市场可能承压"
        elif current < ma_20 < ma_60:
            trend = "down"
            signal = "opportunity"  # 弱美元利好风险资产
            description = f"美元走弱 (DXY: {current:.2f})，风险偏好上升，利好大宗商品"
        else:
            trend = "neutral"
            signal = "neutral"
            description = f"美元横盘 (DXY: {current:.2f})"
        
        return MacroIndicator(
            name="美元指数",
            symbol=symbol,
            current_value=round(current, 2),
            previous_value=round(previous, 2) if previous else None,
            change_pct=round(change_pct, 2) if change_pct else None,
            trend=trend,
            signal=signal,
            description=description,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    def analyze_all(self) -> MacroAnalysisResult:
        """
        综合分析所有宏观指标
        
        Returns:
            MacroAnalysisResult: 综合分析结果
        """
        indicators = {}
        
        # 获取各指标
        vix = self.get_vix_indicator()
        if vix:
            indicators["vix"] = vix
        
        tnx = self.get_treasury_yield_indicator()
        if tnx:
            indicators["treasury_yield"] = tnx
        
        curve = self.get_yield_curve_indicator()
        if curve:
            indicators["yield_curve"] = curve
        
        dxy = self.get_dxy_indicator()
        if dxy:
            indicators["dxy"] = dxy
        
        # 计算综合风险评分
        risk_score = self._calculate_risk_score(indicators)
        
        # 确定市场情绪
        sentiment = self._determine_sentiment(risk_score, indicators)
        
        # 推荐策略
        recommended = self._recommend_strategy(risk_score, indicators)
        
        # 生成分析摘要
        summary = self._generate_summary(indicators, risk_score, sentiment)
        
        # 计算置信度
        confidence = len(indicators) / 4.0  # 基于数据完整性
        
        return MacroAnalysisResult(
            risk_score=risk_score,
            sentiment=sentiment,
            recommended_strategy=recommended,
            indicators=indicators,
            analysis_summary=summary,
            confidence=min(confidence, 1.0)
        )
    
    def _calculate_risk_score(self, indicators: Dict[str, MacroIndicator]) -> float:
        """
        计算综合风险评分 (0-100)
        
        评分逻辑：
        - VIX 越高风险越高
        - 收益率上升风险增加
        - 收益率曲线倒挂风险增加
        - 美元过强风险增加
        """
        score = 50.0  # 基准分
        
        # VIX 贡献 (最大 ±25 分)
        if "vix" in indicators:
            vix = indicators["vix"].current_value
            if vix > 30:
                score += min((vix - 20) * 1.5, 25)
            elif vix < 12:
                score -= (12 - vix) * 2  # 过低也有风险
        
        # 收益率曲线贡献 (最大 ±15 分)
        if "yield_curve" in indicators:
            spread = indicators["yield_curve"].current_value
            if spread < 0:
                score += abs(spread) * 20  # 倒挂增加风险
            elif spread < 0.3:
                score += 5  # 接近倒挂
        
        # 美债收益率贡献 (最大 ±10 分)
        if "treasury_yield" in indicators:
            tnx = indicators["treasury_yield"]
            if tnx.trend == "up" and tnx.current_value > 4:
                score += 10  # 高利率上升风险
            elif tnx.trend == "down":
                score -= 5  # 利率下降降低风险
        
        # 美元指数贡献 (最大 ±10 分)
        if "dxy" in indicators:
            dxy = indicators["dxy"]
            if dxy.trend == "up" and dxy.current_value > 105:
                score += 10  # 强美元风险
        
        return max(0, min(100, round(score, 1)))
    
    def _determine_sentiment(self, risk_score: float, indicators: Dict) -> str:
        """确定市场情绪"""
        if risk_score >= 70:
            return "fear"
        elif risk_score >= 55:
            return "cautious"
        elif risk_score >= 45:
            return "neutral"
        elif risk_score >= 30:
            return "optimistic"
        else:
            return "greed"
    
    def _recommend_strategy(self, risk_score: float, indicators: Dict) -> str:
        """推荐策略类型"""
        if risk_score >= 70:
            return "defensive"  # 防守型：减仓、避险
        elif risk_score >= 55:
            return "cautious"   # 谨慎型：控制仓位
        elif risk_score >= 45:
            return "balanced"   # 平衡型：正常配置
        elif risk_score >= 30:
            return "selective"  # 选择性：精选标的
        else:
            return "aggressive" # 进取型：可加仓
    
    def _generate_summary(self, indicators: Dict, risk_score: float, sentiment: str) -> str:
        """生成分析摘要"""
        parts = []
        
        parts.append(f"宏观风险评分: {risk_score}/100 ({sentiment})")
        
        if "vix" in indicators:
            vix = indicators["vix"]
            parts.append(f"VIX: {vix.current_value} ({vix.trend})")
        
        if "treasury_yield" in indicators:
            tnx = indicators["treasury_yield"]
            parts.append(f"10Y收益率: {tnx.current_value:.2f}% ({tnx.trend})")
        
        if "yield_curve" in indicators:
            curve = indicators["yield_curve"]
            parts.append(f"收益率曲线: {curve.trend}")
        
        if "dxy" in indicators:
            dxy = indicators["dxy"]
            parts.append(f"美元: {dxy.current_value:.1f} ({dxy.trend})")
        
        return " | ".join(parts)


# 便捷函数
_macro_service: Optional[MacroIndicatorsService] = None


def get_macro_service(cache_enabled: bool = True) -> MacroIndicatorsService:
    """获取宏观指标服务实例"""
    global _macro_service
    if _macro_service is None:
        _macro_service = MacroIndicatorsService(cache_enabled=cache_enabled)
    return _macro_service


def get_macro_analysis() -> Dict:
    """
    快速获取宏观分析结果
    
    Returns:
        Dict: 宏观分析结果
    """
    service = get_macro_service()
    result = service.analyze_all()
    return result.to_dict()


def get_vix_level() -> Optional[float]:
    """快速获取 VIX 水平"""
    service = get_macro_service()
    vix = service.get_vix_indicator()
    return vix.current_value if vix else None


def is_high_risk_environment() -> bool:
    """判断是否高风险环境"""
    analysis = get_macro_analysis()
    return analysis.get("risk_score", 50) >= 70


def should_be_defensive() -> bool:
    """判断是否应采取防守策略"""
    analysis = get_macro_analysis()
    return analysis.get("recommended_strategy") in ("defensive", "cautious")
