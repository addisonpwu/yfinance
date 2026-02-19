"""
股票分析器

负责单只股票的分析逻辑，包括数据获取、策略执行、AI分析

优化：
- 添加API调用延迟机制，避免被限制
- 动态调整延迟时间
"""
from typing import Dict, Optional, Any
import pandas as pd
from dataclasses import dataclass
import time
import random
import threading

from src.data.loaders.yahoo_loader import YahooFinanceRepository, calculate_technical_indicators, optimize_dataframe_memory
from src.core.strategies.strategy import StrategyContext, StrategyEngine
from src.core.strategies.loader import get_strategies
from src.ai.analyzer.service import AIAnalysisService
from src.config.settings import config_manager
from src.utils.logger import get_analysis_logger


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
    error: Optional[str] = None


class StockAnalyzer:
    """股票分析器"""
    
    # 类级别的锁和计数器，用于控制并发速度
    _rate_limit_lock = threading.Lock()
    _last_request_time = 0.0
    _consecutive_errors = 0
    
    def __init__(self):
        self.data_repo = YahooFinanceRepository()
        self.strategy_engine = StrategyEngine(get_strategies())
        self.ai_service = AIAnalysisService()
        self.config = config_manager.get_config()
        self.logger = get_analysis_logger()
        
        # 从配置读取延迟参数
        self.base_delay = getattr(self.config.api, 'base_delay', 0.5)
        self.max_delay = getattr(self.config.api, 'max_delay', 2.0)
        self.min_delay = getattr(self.config.api, 'min_delay', 0.1)
    
    def _apply_rate_limit(self):
        """应用速率限制，确保API调用之间有适当延迟"""
        with StockAnalyzer._rate_limit_lock:
            now = time.time()
            time_since_last = now - StockAnalyzer._last_request_time
            
            # 根据连续错误数动态调整延迟
            if StockAnalyzer._consecutive_errors > 0:
                delay = min(self.base_delay * (2 ** StockAnalyzer._consecutive_errors), self.max_delay)
            else:
                # 正常情况下使用基础延迟 + 随机抖动
                delay = self.base_delay + random.uniform(0, 0.3)
            
            delay = max(delay, self.min_delay)
            
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                time.sleep(sleep_time)
            
            StockAnalyzer._last_request_time = time.time()
    
    def _on_success(self):
        """成功时重置错误计数"""
        with StockAnalyzer._rate_limit_lock:
            StockAnalyzer._consecutive_errors = max(0, StockAnalyzer._consecutive_errors - 1)
    
    def _on_error(self):
        """错误时增加错误计数"""
        with StockAnalyzer._rate_limit_lock:
            StockAnalyzer._consecutive_errors = min(StockAnalyzer._consecutive_errors + 1, 5)
    
    def analyze(
        self,
        symbol: str,
        market: str,
        market_return: float,
        is_market_healthy: bool,
        skip_strategies: bool = False,
        interval: str = '1d',
        model: str = 'deepseek-v3.2'
    ) -> StockAnalysisResult:
        """
        分析单只股票
        
        Args:
            symbol: 股票代码
            market: 市场代码
            market_return: 大盘回报率
            is_market_healthy: 市场是否健康
            skip_strategies: 是否跳过策略筛选
            interval: 数据时段类型
            model: AI 模型
        
        Returns:
            StockAnalysisResult 分析结果
        """
        # 应用速率限制
        self._apply_rate_limit()
        
        try:
            # 获取股票数据
            hist = self.data_repo.get_historical_data(symbol, market, interval=interval)
            info = self.data_repo.get_financial_info(symbol)
            news = self.data_repo.get_news(symbol, market)  # 传递 market 参数
            
            # 数据质量检查
            if hist.empty or len(hist) < 2 or info is None or (isinstance(info, dict) and len(info) == 0):
                return StockAnalysisResult(
                    symbol=symbol,
                    exchange='',
                    strategies=[],
                    info={},
                    news=[],
                    ai_analysis=None,
                    success=False,
                    error="数据不足"
                )
            
            # 数据预处理优化：基础筛选
            if self.config.analysis.enable_data_preprocessing:
                # 基础数据质量检查
                if 'Volume' in hist.columns and not hist['Volume'].empty:
                    recent_volume = hist['Volume'].tail(5).mean()
                    if recent_volume < self.config.analysis.min_volume_threshold:
                        return StockAnalysisResult(
                            symbol=symbol,
                            exchange='',
                            strategies=[],
                            info={},
                            news=[],
                            ai_analysis=None,
                            success=False,
                            error="成交量过低"
                        )
                
                # 检查价格数据是否有效
                if 'Close' in hist.columns:
                    recent_prices = hist['Close'].tail(10)
                    if recent_prices.isna().all() or (recent_prices <= 0).any():
                        return StockAnalysisResult(
                            symbol=symbol,
                            exchange='',
                            strategies=[],
                            info={},
                            news=[],
                            ai_analysis=None,
                            success=False,
                            error="价格数据无效"
                        )
                
                # 检查是否有足够的有效数据点
                min_data_points = self.config.analysis.min_data_points_threshold
                if len(hist.dropna()) < min_data_points:
                    return StockAnalysisResult(
                        symbol=symbol,
                        exchange='',
                        strategies=[],
                        info={},
                        news=[],
                        ai_analysis=None,
                        success=False,
                        error="数据点不足"
                    )
            
            # 预计算技术指标
            hist = calculate_technical_indicators(hist, self.config)
            
            # 优化内存使用
            hist = optimize_dataframe_memory(hist)
            
            # 执行策略或跳过
            passed_strategies = []
            if skip_strategies:
                passed_strategies = ["跳过策略"]
            else:
                strategy_context = StrategyContext(
                    hist=hist,
                    info=info,
                    market_return=market_return,
                    is_market_healthy=is_market_healthy
                )
                
                strategies_to_run = get_strategies()
                strategy_results = self.strategy_engine.execute_all(strategy_context)
                
                for i, result in enumerate(strategy_results):
                    if result.passed:
                        passed_strategies.append(strategies_to_run[i].name)
            
            # 如果未通过任何策略且未跳过策略，返回失败
            if not passed_strategies and not skip_strategies:
                return StockAnalysisResult(
                    symbol=symbol,
                    exchange=info.get('exchange', 'UNKNOWN'),
                    strategies=[],
                    info=info,
                    news=news,
                    ai_analysis=None,
                    success=False,
                    error="未通过任何策略"
                )
            
            # AI 分析
            ai_analysis = None
            try:
                stock_data = {
                    'symbol': symbol,
                    'strategies': passed_strategies,
                    'info': info,
                    'market': market,
                    'news': news  # 添加新闻数据
                }
                ai_result = self.ai_service.analyze_stock(stock_data, hist, interval=interval, model=model)
                if ai_result:
                    ai_analysis = {
                        'summary': ai_result.summary,
                        'model_used': ai_result.model_used
                    }
            except Exception as ai_e:
                self.logger.error(f"AI 分析出错: {ai_e}")
            
            return StockAnalysisResult(
                symbol=symbol,
                exchange=info.get('exchange', 'UNKNOWN'),
                strategies=passed_strategies,
                info=info,
                news=news,
                ai_analysis=ai_analysis,
                success=True
            )
            
        except Exception as e:
            self._on_error()  # 错误时增加延迟
            self.logger.error(f"分析 {symbol} 时发生错误: {e}")
            return StockAnalysisResult(
                symbol=symbol,
                exchange='',
                strategies=[],
                info={},
                news=[],
                ai_analysis=None,
                success=False,
                error=str(e)
            )
    
    def to_dict(self, result: StockAnalysisResult) -> Dict[str, Any]:
        """将分析结果转换为字典"""
        return {
            'symbol': result.symbol,
            'exchange': result.exchange,
            'strategies': result.strategies,
            'info': result.info,
            'news': result.news,
            'ai_analysis': result.ai_analysis
        }
