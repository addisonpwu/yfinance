from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import pandas as pd


class NewsType(Enum):
    """新闻类型"""
    PROFIT = "profit"      # 盈利预告
    RATING = "rating"      # 机构评级
    OTHER = ""             # 其他


@dataclass
class NewsItem:
    """新闻条目"""
    title: str
    publish_time: str
    url: str
    type: str = ""         # profit, rating, 或空
    agency: str = ""       # 机构名称
    rating: str = ""       # 评级内容
    profit: str = ""       # 盈利信息
    
    def get_type_display(self) -> str:
        """获取类型显示文本"""
        if self.type == "profit":
            return "✅ 盈利预告"
        elif self.type == "rating":
            return "✅ 机构评级"
        return ""


@dataclass
class StockData:
    symbol: str
    hist: pd.DataFrame
    info: Dict
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
    ai_analysis: Optional[Dict]
    success: bool  # 是否成功分析（通过基础筛选）
    technical_indicators: Optional[Dict] = None  # 技术指标
    strategy_details: list = field(default_factory=list)  # 策略详情列表
    news: List[Dict] = field(default_factory=list)  # 新闻列表（从 stock.json 导入）
    error: Optional[str] = None