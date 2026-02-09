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