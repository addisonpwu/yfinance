from typing import Dict, List, Optional
import pandas as pd
from src.core.models.entities import AIAnalysisResult
from src.ai.analyzer.iflow_analyzer import IFlowAIAnalyzer
from src.utils.logger import get_ai_logger

class AIAnalysisService:
    def __init__(self):
        self.analyzer = IFlowAIAnalyzer()
        self.logger = get_ai_logger()
    
    def analyze_stock(self, stock_data: Dict, hist: pd.DataFrame = None, **kwargs) -> Optional[AIAnalysisResult]:
        """
        对单个股票进行AI分析
        """
        try:
            result = self.analyzer.analyze(stock_data, hist, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"AI分析股票 {stock_data.get('symbol', 'Unknown')} 时出错: {e}")
            return None
    
    def analyze_multiple_stocks(self, stocks_data: List[Dict], **kwargs) -> List[Dict]:
        """
        批量分析多支股票
        
        Args:
            stocks_data: 股票数据列表
            **kwargs: 其他参数，如model等
        
        Returns:
            添加了AI分析结果的股票数据列表
        """
        self.logger.info(f"开始AI综合分析 ({len(stocks_data)} 支股票)")
        
        for i, stock in enumerate(stocks_data):
            self.logger.info(f"AI分析进度: [{i+1}/{len(stocks_data)}] {stock.get('symbol', 'Unknown')}...")
            
            ai_result = self.analyze_stock(stock, **kwargs)
            
            if ai_result:
                stock['ai_analysis'] = {
                    'summary': ai_result.summary,
                    'model_used': ai_result.model_used
                }
            else:
                stock['ai_analysis'] = None
        
        self.logger.info("AI分析完成")
        return stocks_data