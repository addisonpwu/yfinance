from dataclasses import dataclass, field
from typing import Dict, Optional, List
import pandas as pd

@dataclass
class StockData:
    symbol: str
    hist: pd.DataFrame
    info: Dict
    news: Optional[List] = None
    finviz_data: Optional[Dict] = field(default_factory=dict)  # Finviz 额外数据

@dataclass
class StrategyResult:
    passed: bool
    confidence: float = 1.0
    details: Optional[Dict] = None

@dataclass
class AIAnalysisResult:
    summary: str
    confidence: float
    model_used: str
    detailed_analysis: Optional[Dict] = None


@dataclass
class StockAnalysisResult:
    """股票分析结果"""
    symbol: str
    exchange: str
    strategies: list
    info: Dict
    news: list
    ai_analysis: Optional[Dict]
    success: bool  # 是否成功分析（通过基础筛选）
    technical_indicators: Optional[Dict] = None  # 技术指标
    strategy_details: list = field(default_factory=list)  # 策略详情列表
    error: Optional[str] = None