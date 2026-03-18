"""
股票分析器

负责单只股票的分析逻辑，包括数据获取、策略执行、AI分析

优化：
- 添加API调用延迟机制，避免被限制
- 动态调整延迟时间
"""
from typing import Dict, Optional, Any
import pandas as pd
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError, TimeoutError

from src.data.loaders.yahoo_loader import YahooFinanceRepository, calculate_technical_indicators, optimize_dataframe_memory
from src.core.strategies.strategy import StrategyContext, StrategyEngine
from src.core.strategies.loader import get_strategies
from src.core.models.entities import StockAnalysisResult
from src.ai.analyzer.service import AIAnalysisService
from src.config.settings import config_manager
from src.utils.logger import get_analysis_logger


class StockAnalyzer:
    """股票分析器"""
    
    # 类级别的锁和计数器，用于控制并发速度
    _rate_limit_lock = threading.Lock()
    _last_request_time = 0.0
    _consecutive_errors = 0
    
    @staticmethod
    def _extract_technical_indicators(hist: pd.DataFrame) -> Dict[str, Any]:
        """从历史数据中提取最新的技术指标"""
        if hist is None or hist.empty:
            return {}
        
        try:
            latest = hist.iloc[-1]
            indicators = {}
            
            # RSI
            if 'RSI_14' in hist.columns:
                rsi_val = latest.get('RSI_14')
                if pd.notna(rsi_val):
                    indicators['rsi'] = round(float(rsi_val), 2)
                    # RSI 状态判断
                    if rsi_val > 70:
                        indicators['rsi_status'] = 'overbought'  # 超买
                    elif rsi_val < 30:
                        indicators['rsi_status'] = 'oversold'  # 超卖
                    else:
                        indicators['rsi_status'] = 'normal'
            
            # MACD
            if 'MACD' in hist.columns and 'MACD_Signal' in hist.columns:
                macd_val = latest.get('MACD')
                signal_val = latest.get('MACD_Signal')
                hist_val = latest.get('MACD_Histogram')
                
                if pd.notna(macd_val) and pd.notna(signal_val):
                    indicators['macd'] = round(float(macd_val), 4)
                    indicators['macd_signal'] = round(float(signal_val), 4)
                    if pd.notna(hist_val):
                        indicators['macd_hist'] = round(float(hist_val), 4)
                    
                    # MACD 状态判断
                    if macd_val > signal_val:
                        indicators['macd_status'] = 'golden_cross'  # 金叉
                    else:
                        indicators['macd_status'] = 'death_cross'  # 死叉
            
            # 布林带
            if all(col in hist.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle', 'BBP']):
                bb_upper = latest.get('BB_Upper')
                bb_lower = latest.get('BB_Lower')
                bb_middle = latest.get('BB_Middle')
                bbp = latest.get('BBP')
                
                if all(pd.notna(v) for v in [bb_upper, bb_lower, bb_middle]):
                    indicators['bb_upper'] = round(float(bb_upper), 2)
                    indicators['bb_lower'] = round(float(bb_lower), 2)
                    indicators['bb_middle'] = round(float(bb_middle), 2)
                    if pd.notna(bbp):
                        indicators['bb_position'] = round(float(bbp), 2)  # 0-1之间，0下轨，1上轨
                    
                    close = latest.get('Close')
                    if pd.notna(close):
                        # 布林带位置状态
                        if close >= bb_upper:
                            indicators['bb_status'] = 'above_upper'  # 突破上轨
                        elif close <= bb_lower:
                            indicators['bb_status'] = 'below_lower'  # 跌破下轨
                        else:
                            indicators['bb_status'] = 'in_band'  # 在带内
            
            # 移动平均线位置
            close = latest.get('Close')
            if pd.notna(close):
                ma_positions = {}
                ma_periods = [5, 10, 20, 50, 200]
                for period in ma_periods:
                    ma_col = f'MA_{period}'
                    if ma_col in hist.columns:
                        ma_val = latest.get(ma_col)
                        if pd.notna(ma_val):
                            ma_positions[f'ma_{period}'] = round(float(ma_val), 2)
                            ma_positions[f'above_ma_{period}'] = close > ma_val
                
                if ma_positions:
                    indicators['ma'] = ma_positions
                    # 计算价格在多少条均线之上
                    above_count = sum(1 for k, v in ma_positions.items() if k.startswith('above_ma_') and v)
                    total_ma = sum(1 for k in ma_positions if k.startswith('above_ma_'))
                    indicators['ma_score'] = f"{above_count}/{total_ma}"
            
            # ATR
            if 'ATR_14' in hist.columns:
                atr_val = latest.get('ATR_14')
                atr_pct = latest.get('ATR_Pct')
                if pd.notna(atr_val):
                    indicators['atr'] = round(float(atr_val), 2)
                if pd.notna(atr_pct):
                    indicators['atr_pct'] = round(float(atr_pct), 2)
            
            return indicators
        except Exception as e:
            return {}
    
    def __init__(self, provider: str = 'iflow', providers: list = None):
        self.data_repo = YahooFinanceRepository()
        self.strategy_engine = StrategyEngine(get_strategies())
        
        # 支持多提供商：providers 参数优先，否则使用 provider 参数
        if providers:
            self.providers = providers if isinstance(providers, list) else [providers]
        else:
            self.providers = [provider]
        
        # 初始化所有提供商的服务
        self.ai_services = {}
        for p in self.providers:
            self.ai_services[p] = AIAnalysisService(provider=p)
        
        # 主服务（用于兼容）
        self.ai_service = self.ai_services.get(self.providers[0])
        
        self.config = config_manager.get_config()
        self.logger = get_analysis_logger()
        self.provider = self.providers[0] if self.providers else provider
        
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
    
    def _analyze_with_provider(
        self,
        provider: str,
        stock_data: Dict[str, Any],
        hist: pd.DataFrame,
        interval: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """
        使用单个提供商进行 AI 分析
        
        Args:
            provider: 提供商名称
            stock_data: 股票数据字典
            hist: 历史价格数据
            interval: 数据时间间隔
            model: AI 模型名称
        
        Returns:
            分析结果字典或 None（失败时）
        """
        # P0: 应用速率限制，确保并行执行时不会绕过限制
        self._apply_rate_limit()
        try:
            ai_service = self.ai_services.get(provider)
            if not ai_service:
                self.logger.warning(f"提供商 {provider} 服务未初始化")
                return None
            
            result = ai_service.analyze_stock(stock_data, hist, interval=interval, model=model)
            if result:
                return {
                    'provider': provider,
                    'summary': result.summary,
                    'model_used': result.model_used,
                    'confidence': result.confidence,
                    'detailed_analysis': result.detailed_analysis if hasattr(result, 'detailed_analysis') else None
                }
        except Exception as e:
            self.logger.warning(f"提供商 {provider} 分析失败: {e}")
        
        return None
    
    def analyze(
        self,
        symbol: str,
        market: str,
        market_return: float,
        is_market_healthy: bool,
        skip_strategies: bool = False,
        interval: str = '1d',
        model: str = 'deepseek-v3.2',
        providers: list = None,
        force_refresh: bool = False
    ) -> StockAnalysisResult:
        """
        分析单只股票
        
        Args:
            symbol: 股票代码
            market: 市场代码
            market_return: 大盘回报率
            is_market_healthy: 市场是否健康
            skip_strategies: 是否跳过策略筛选
            interval: 数据时间间隔
            model: AI 模型名称
            providers: AI 提供商列表
            force_refresh: 是否强制刷新缓存获取最新数据
        
        Returns:
            StockAnalysisResult 分析结果
        """
        # 确定要使用的提供商列表
        providers_to_use = providers if providers else self.providers
        # 应用速率限制
        self._apply_rate_limit()
        
        try:
            # 获取股票数据
            hist = self.data_repo.get_historical_data(symbol, market, interval=interval, force_refresh=force_refresh)
            info = self.data_repo.get_financial_info(symbol)
            
            # 数据质量检查
            if hist.empty or len(hist) < 2 or info is None or (isinstance(info, dict) and len(info) == 0):
                return StockAnalysisResult(
                    symbol=symbol,
                    exchange='',
                    strategies=[],
                    info={},
                    ai_analysis=None,
                    success=False,
                    technical_indicators=None,
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
                            ai_analysis=None,
                            success=False,
                            technical_indicators=None,
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
                            ai_analysis=None,
                            success=False,
                            technical_indicators=None,
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
                        ai_analysis=None,
                        success=False,
                        technical_indicators=None,
                        error="数据点不足"
                    )
            
            # 预计算技术指标
            hist = calculate_technical_indicators(hist, self.config)
            
            # 优化内存使用
            hist = optimize_dataframe_memory(hist)
            
            # 执行策略或跳过
            passed_strategies = []
            strategy_details = []  # 存储策略详情
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
                        # 保存完整的策略详情
                        strategy_details.append({
                            'strategy_name': strategies_to_run[i].name,
                            'details': result.details
                        })
            
            # 如果未通过任何策略且未跳过策略，返回失败
            if not passed_strategies and not skip_strategies:
                return StockAnalysisResult(
                    symbol=symbol,
                    exchange=info.get('exchange', 'UNKNOWN'),
                    strategies=[],
                    strategy_details=[],
                    info=info,
                    ai_analysis=None,
                    success=False,
                    technical_indicators=None,
                    error="未通过任何策略"
                )
            
            # AI 分析（支持多提供商）
            ai_analysis = None
            try:
                stock_data = {
                    'symbol': symbol,
                    'strategies': passed_strategies,
                    'info': info,
                    'market': market
                }
                
                # 如果有多个提供商，并行进行多提供商分析
                if len(providers_to_use) > 1:
                    ai_results = []
                    # 使用 ThreadPoolExecutor 并行执行（P2: 添加上限避免过多并发）
                    max_workers = min(len(providers_to_use), 4)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # 提交所有提供商的分析任务
                        future_to_provider = {
                            executor.submit(
                                self._analyze_with_provider,
                                p, stock_data, hist, interval, model
                            ): p for p in providers_to_use
                        }
                        
                        # 收集结果（P1: 添加并发异常处理）
                        for future in as_completed(future_to_provider):
                            provider = future_to_provider[future]
                            try:
                                result = future.result()
                                if result:
                                    ai_results.append(result)
                            except CancelledError:
                                self.logger.warning(f"提供商 {provider} 任务被取消")
                            except TimeoutError:
                                self.logger.warning(f"提供商 {provider} 任务超时")
                            except Exception as e:
                                self.logger.error(f"提供商 {provider} 任务异常: {e}")
                    
                    if ai_results:
                        # 合并多提供商结果，使用正确格式 --- PROVIDER 分析 ---
                        combined_summary = ""
                        total_confidence = 0
                        for r in ai_results:
                            provider_name = r['provider'].upper()
                            combined_summary += f"--- {provider_name} 分析 ---\n"
                            combined_summary += f"模型: {r['model_used']}\n"
                            combined_summary += f"置信度: {r['confidence']:.0%}\n"
                            combined_summary += f"{r['summary']}\n\n"
                            total_confidence += r['confidence']
                        
                        # 计算平均置信度
                        avg_confidence = total_confidence / len(ai_results) if ai_results else 0.5
                        
                        ai_analysis = {
                            'summary': combined_summary,
                            'model_used': f"multi_provider({', '.join([r['provider'] for r in ai_results])})",
                            'confidence': avg_confidence
                        }
                else:
                    # 单提供商分析
                    ai_result = self.ai_service.analyze_stock(stock_data, hist, interval=interval, model=model)
                    if ai_result:
                        ai_analysis = {
                            'summary': ai_result.summary,
                            'model_used': ai_result.model_used,
                            'confidence': ai_result.confidence if hasattr(ai_result, 'confidence') else 0.5
                        }
            except Exception as ai_e:
                self.logger.error(f"AI 分析出错: {ai_e}")
            
            # 提取技术指标
            technical_indicators = self._extract_technical_indicators(hist)
            
            return StockAnalysisResult(
                symbol=symbol,
                exchange=info.get('exchange', 'UNKNOWN'),
                strategies=passed_strategies,
                strategy_details=strategy_details,
                info=info,
                ai_analysis=ai_analysis,
                success=True,
                technical_indicators=technical_indicators
            )
            
        except Exception as e:
            self._on_error()  # 错误时增加延迟
            self.logger.error(f"分析 {symbol} 时发生错误: {e}")
            return StockAnalysisResult(
                symbol=symbol,
                exchange='',
                strategies=[],
                info={},
                ai_analysis=None,
                success=False,
                technical_indicators=None,
                error=str(e)
            )
    
    def to_dict(self, result: StockAnalysisResult) -> Dict[str, Any]:
        """将分析结果转换为字典"""
        return {
            'symbol': result.symbol,
            'exchange': result.exchange,
            'strategies': result.strategies,
            'info': result.info,
            'ai_analysis': result.ai_analysis,
            'technical_indicators': result.technical_indicators,
            'strategy_details': result.strategy_details,
            'news': result.news
        }
