"""
AI 分析器模块

提供多种 AI 分析器实现：
- IFlowAIAnalyzer: 心流 AI 分析器
- NvidiaAIAnalyzer: NVIDIA API 分析器
- AIAnalysisService: 统一服务封装
"""

from src.ai.analyzer.iflow_analyzer import (
    AIAnalyzer,
    IFlowAIAnalyzer,
    PredictionDirection,
    PredictionRecord,
    MultiModelConsensus,
    PredictionTracker,
)
from src.ai.analyzer.nvidia_analyzer import NvidiaAIAnalyzer
from src.ai.analyzer.service import AIAnalysisService

__all__ = [
    'AIAnalyzer',
    'IFlowAIAnalyzer',
    'NvidiaAIAnalyzer',
    'AIAnalysisService',
    'PredictionDirection',
    'PredictionRecord',
    'MultiModelConsensus',
    'PredictionTracker',
]
