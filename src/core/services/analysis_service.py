"""
分析服务主入口

协调各服务组件完成股票分析流程
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from src.core.services.market_data_service import MarketDataService
from src.core.services.cache_version_manager import CacheVersionManager
from src.core.services.stock_analyzer import StockAnalyzer, StockAnalysisResult
from src.core.services.progress_tracker import ProgressTracker
from src.core.services.report_writer import ReportWriter
from src.core.strategies.loader import get_strategies
from src.config.settings import config_manager


def run_analysis(
    market: str,
    force_fast_mode: bool = False,
    skip_strategies: bool = False,
    symbol_filter: str = None,
    interval: str = '1d',
    max_workers: int = None,
    model: str = 'deepseek-v3.2',
    providers: List[str] = None,
    output_filename: str = None
) -> List[Dict]:
    """
    对指定市场执行所有选股策略分析

    Args:
        market: 市场代码 ('US' 或 'HK')
        force_fast_mode: 是否强制跳过缓存更新，直接使用快速模式
        skip_strategies: 是否跳过策略筛选，所有股票都进行AI分析
        symbol_filter: 指定分析单一股票代码
        interval: 数据时段类型 ('1d' 日线, '1h' 小时线, '1m' 分钟线)
        max_workers: 最大并行工作线程数
        model: 要使用的AI模型
        providers: AI 提供商列表 (如 ['iflow', 'nvidia', 'gemini'])
        output_filename: 实时报告文件名

    Returns:
        符合条件的股票列表
    """
    # 处理 providers 参数（支持多提供商）
    if providers is None:
        providers = ['iflow']
    elif isinstance(providers, str):
        providers = [p.strip() for p in providers.split(',')]
    
    # 加载配置
    config = config_manager.get_config()
    
    if max_workers is None:
        max_workers = config.api.max_workers
    
    # 初始化服务（使用首个提供商作为默认）
    market_service = MarketDataService()
    cache_manager = CacheVersionManager()
    primary_provider = providers[0] if providers else 'iflow'
    stock_analyzer = StockAnalyzer(provider=primary_provider)
    
    # --- 缓存版本检查 ---
    is_sync_needed, status_msg = cache_manager.check_version(market, force_fast_mode, interval)
    print(status_msg)
    
    # --- 获取股票列表 ---
    if symbol_filter:
        tickers = [symbol_filter]
        print(f"--- 使用指定股票: {symbol_filter} ---")
    else:
        tickers = market_service.get_stock_list(market)
    
    total_stocks = len(tickers)
    
    # --- 获取大盘数据 ---
    market_data = market_service.get_market_data(market)
    market_return = market_data.latest_return if market_data else 0.0
    is_market_healthy = market_data.is_healthy if market_data else False
    
    # --- 加载策略 ---
    strategies_to_run = get_strategies()
    if not strategies_to_run:
        print("警告: 在 'strategies' 文件夹中没有找到任何策略。")
        return []
    print(f"已加载 {len(strategies_to_run)} 个策略: {[s.name for s in strategies_to_run]}")
    
    # --- 初始化报告和进度追踪 ---
    report_writer = ReportWriter(output_filename, market, output_format='both')
    report_writer.initialize()
    print(f"--- 报告将保存至: {report_writer.get_filename()} ---")
    print(f"--- HTML 报告: {report_writer.get_html_filename()} ---")
    
    progress_tracker = ProgressTracker(total_stocks)
    
    # --- 分析股票 ---
    print(f"\n--- 开始逐个股票进行分析和预测 ---")
    qualified_stocks = []
    
    def analyze_single_stock(symbol: str) -> StockAnalysisResult:
        """分析单只股票的包装函数"""
        result = stock_analyzer.analyze(
            symbol=symbol,
            market=market,
            market_return=market_return,
            is_market_healthy=is_market_healthy,
            skip_strategies=skip_strategies,
            interval=interval,
            model=model,
            providers=providers
        )
        
        # 如果成功且有结果，写入报告
        if result.success and result.strategies:
            report_writer.write_stock_result(stock_analyzer.to_dict(result))
            
            if skip_strategies:
                print(f"\r{' ' * 80}\r✅ {symbol} 跳过策略筛选，已进行AI分析")
            else:
                print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {result.strategies}")
            
            if result.ai_analysis:
                print(f"   🤖 AI 分析: 已完成 (模型: {result.ai_analysis.get('model_used', 'N/A')})")
            else:
                print(f"   🤖 AI 分析: 未能完成")
        elif not result.success and result.error:
            print(f"\r{' ' * 80}\r❌ {symbol}: {result.error}")
        
        return result
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(analyze_single_stock, symbol): symbol 
            for symbol in tickers
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                
                # 更新进度
                progress_tracker.update(success=result.success)
                
                # 收集符合条件的股票
                if result.success and result.strategies:
                    qualified_stocks.append(stock_analyzer.to_dict(result))
                
            except Exception as e:
                print(f"\r{' ' * 80}\r❌ 处理 {symbol} 的结果时发生错误: {e}")
                progress_tracker.update(success=False)
            
            # 显示进度
            print(progress_tracker.format_status(), end='')
    
    # --- 更新缓存版本 ---
    if is_sync_needed:
        cache_manager.update_version(market)
    
    # --- 输出摘要 ---
    print(progress_tracker.get_summary())
    
    # --- 写入摘要列表 ---
    if qualified_stocks:
        report_writer.write_summary(qualified_stocks, market)
        print(f"\n--- 完整报告已储存至 {report_writer.get_filename()} ---")
    
    return qualified_stocks
