"""
AI 分析服务

提供统一的 AI 分析接口，支持多种 AI 提供商：
- iflow: 心流 AI
- nvidia: NVIDIA NIM API
"""

from typing import Dict, List, Optional, Literal
import pandas as pd
from src.core.models.entities import AIAnalysisResult
from src.ai.analyzer.iflow_analyzer import IFlowAIAnalyzer
from src.ai.analyzer.nvidia_analyzer import NvidiaAIAnalyzer
from src.config.settings import config_manager
from src.utils.logger import get_ai_logger


# 支持的 AI 提供商类型
ProviderType = Literal['iflow', 'nvidia']


class AIAnalysisService:
    """
    AI 分析服务
    
    统一的 AI 分析接口，支持多种提供商
    
    使用示例:
        # 使用 iFlow
        service = AIAnalysisService(provider='iflow')
        result = service.analyze_stock(stock_data, hist)
        
        # 使用 NVIDIA
        service = AIAnalysisService(provider='nvidia')
        result = service.analyze_stock(stock_data, hist, model='z-ai/glm5')
    """
    
    def __init__(self, provider: ProviderType = 'iflow', **kwargs):
        """
        初始化 AI 分析服务
        
        Args:
            provider: AI 提供商 ('iflow' 或 'nvidia')
            **kwargs: 传递给分析器的额外参数
                - enable_cache: 是否启用缓存 (NVIDIA)
                - enable_streaming: 是否启用流式响应 (NVIDIA)
        """
        self.provider = provider
        self.logger = get_ai_logger()
        self.config = config_manager.get_config()
        
        # 根据提供商初始化分析器
        if provider == 'nvidia':
            self.analyzer = NvidiaAIAnalyzer(
                enable_cache=kwargs.get('enable_cache', True),
                enable_streaming=kwargs.get('enable_streaming', False)
            )
            self.logger.info("使用 NVIDIA AI 分析器")
        else:
            self.analyzer = IFlowAIAnalyzer()
            self.logger.info("使用 iFlow AI 分析器")
    
    def analyze_stock(
        self, 
        stock_data: Dict, 
        hist: pd.DataFrame = None, 
        **kwargs
    ) -> Optional[AIAnalysisResult]:
        """
        对单个股票进行 AI 分析
        
        Args:
            stock_data: 股票数据字典
            hist: 历史数据 DataFrame
            **kwargs: 其他参数
                - model: 模型名称
                - interval: 数据间隔
                - use_multi_timeframe: 是否使用多时间框架
        
        Returns:
            AIAnalysisResult 或 None
        """
        try:
            result = self.analyzer.analyze(stock_data, hist, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"AI分析股票 {stock_data.get('symbol', 'Unknown')} 时出错: {e}")
            return None
    
    def analyze_multiple_stocks(
        self, 
        stocks_data: List[Dict], 
        **kwargs
    ) -> List[Dict]:
        """
        批量分析多支股票
        
        Args:
            stocks_data: 股票数据列表
            **kwargs: 其他参数，如 model 等
        
        Returns:
            添加了 AI 分析结果的股票数据列表
        """
        self.logger.info(f"开始 AI 综合分析 ({len(stocks_data)} 支股票) [provider: {self.provider}]")
        
        for i, stock in enumerate(stocks_data):
            self.logger.info(f"AI分析进度: [{i+1}/{len(stocks_data)}] {stock.get('symbol', 'Unknown')}...")
            
            ai_result = self.analyze_stock(stock, **kwargs)
            
            if ai_result:
                stock['ai_analysis'] = {
                    'summary': ai_result.summary,
                    'model_used': ai_result.model_used,
                    'confidence': ai_result.confidence,
                }
            else:
                stock['ai_analysis'] = None
        
        self.logger.info("AI 分析完成")
        return stocks_data
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """获取可用的 AI 提供商列表"""
        return ['iflow', 'nvidia']
    
    def get_available_models(self, provider: ProviderType = None) -> List[str]:
        """
        获取指定提供商的可用模型列表
        
        Args:
            provider: AI 提供商 (默认使用当前实例的 provider)
        
        Returns:
            可用模型列表
        """
        target_provider = provider or self.provider
        
        # 从配置中读取
        providers_config = self.config.ai.providers
        if target_provider == 'nvidia' and providers_config.nvidia:
            return providers_config.nvidia.available_models
        elif target_provider == 'iflow' and providers_config.iflow:
            return providers_config.iflow.available_models
        
        # 默认返回空列表
        return []
