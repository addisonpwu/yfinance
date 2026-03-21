"""
分析服务主入口

协调各服务组件完成股票分析流程
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from src.core.services.market_data_service import MarketDataService
from src.core.services.cache_version_manager import CacheVersionManager
from src.core.services.stock_analyzer import StockAnalyzer
from src.core.models.entities import StockAnalysisResult
from src.core.services.progress_tracker import ProgressTracker
from src.core.services.report_writer import ReportWriter
from src.core.strategies.loader import get_strategies
from src.config.settings import config_manager
from src.data.loaders.stock_list_loader_enhanced import EnhancedStockListLoader


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
    stock_analyzer = StockAnalyzer(provider=primary_provider, providers=providers)
    
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
    report_writer = ReportWriter(output_filename, market)
    report_writer.initialize()
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
            providers=providers,
            force_refresh=is_sync_needed
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
        print(f"\n--- 完整报告已储存至 {report_writer.get_html_filename()} ---")
    
    return qualified_stocks


def run_analysis_from_json(
    json_path: str,
    market: str = 'HK',
    interval: str = '1d',
    max_workers: int = None,
    model: str = 'deepseek-v3.2',
    providers: List[str] = None,
    output_filename: str = None
) -> List[Dict]:
    """
    从 JSON 文件读取股票列表并执行 AI 分析

    Args:
        json_path: JSON 文件路径
        market: 市场代码 ('US' 或 'HK')，默认 'HK'
        interval: 数据时段类型 ('1d' 日线, '1h' 小时线, '1m' 分钟线)
        max_workers: 最大并行工作线程数
        model: 要使用的AI模型
        providers: AI 提供商列表 (如 ['iflow', 'nvidia', 'gemini'])
        output_filename: 报告文件名

    Returns:
        分析结果列表
    """
    # 处理 providers 参数
    if providers is None:
        providers = ['iflow']
    elif isinstance(providers, str):
        providers = [p.strip() for p in providers.split(',')]
    
    # 加载配置
    config = config_manager.get_config()
    
    if max_workers is None:
        max_workers = config.api.max_workers
    
    # 初始化服务
    market_service = MarketDataService()
    stock_analyzer = StockAnalyzer(provider=providers[0], providers=providers)
    data_repo = stock_analyzer.data_repo  # 获取数据仓库实例
    
    # --- 加载 JSON 文件 ---
    print(f"--- 从 JSON 文件加载股票列表: {json_path} ---")
    loader = EnhancedStockListLoader(json_path)
    stock_items = loader.load()
    stock_with_news = loader.get_stock_with_news()
    
    tickers = [item.stock_code for item in stock_items]
    print(f"--- 已加载 {len(tickers)} 只股票 ---")
    
    # --- 预下载股票历史数据 ---
    print(f"\n--- 开始下载/更新股票历史数据 ---")
    downloaded_count = 0
    failed_downloads = []
    
    for i, symbol in enumerate(tickers, 1):
        try:
            # 强制刷新数据，获取最新数据
            hist = data_repo.get_historical_data(symbol, market, interval=interval, force_refresh=True)
            if hist is not None and not hist.empty:
                downloaded_count += 1
                print(f"\r{' ' * 80}\r📥 [{i}/{len(tickers)}] {symbol} 数据已更新 ({len(hist)} 条记录)")
            else:
                failed_downloads.append(symbol)
                print(f"\r{' ' * 80}\r⚠️ [{i}/{len(tickers)}] {symbol} 数据为空")
        except Exception as e:
            failed_downloads.append(symbol)
            print(f"\r{' ' * 80}\r❌ [{i}/{len(tickers)}] {symbol} 下载失败: {str(e)[:50]}")
    
    print(f"\n--- 数据下载完成: 成功 {downloaded_count}/{len(tickers)} ---")
    if failed_downloads:
        print(f"--- 下载失败的股票: {', '.join(failed_downloads)} ---")
    
    total_stocks = len(tickers)
    
    # --- 获取大盘数据 ---
    market_data = market_service.get_market_data(market)
    market_return = market_data.latest_return if market_data else 0.0
    is_market_healthy = market_data.is_healthy if market_data else False
    
    # --- 初始化报告和进度追踪 ---
    report_writer = ReportWriter(output_filename, market)
    report_writer.initialize()
    print(f"--- HTML 报告: {report_writer.get_html_filename()} ---")
    
    progress_tracker = ProgressTracker(total_stocks)
    
    # --- 分析股票 ---
    print(f"\n--- 开始逐个股票进行 AI 分析 ---")
    qualified_stocks = []
    
    def analyze_single_stock(symbol: str) -> StockAnalysisResult:
        """分析单只股票的包装函数"""
        # 获取 JSON 中的新闻数据
        stock_info = stock_with_news.get(symbol)
        json_news = []
        if stock_info and stock_info.news:
            json_news = loader.convert_news_to_standard_format(stock_info.news)
        
        result = stock_analyzer.analyze(
            symbol=symbol,
            market=market,
            market_return=market_return,
            is_market_healthy=is_market_healthy,
            skip_strategies=True,  # 跳过策略筛选，直接 AI 分析
            skip_volume_check=True,  # 跳过成交量检查
            interval=interval,
            model=model,
            providers=providers,
            force_refresh=False
        )
        
        # 合并 JSON 中的新闻到结果中
        if result.success and json_news:
            # 将 JSON 新闻添加到现有新闻前面
            existing_news = result.news or []
            result.news = json_news + existing_news
        
        # 如果成功，写入报告
        if result.success:
            report_writer.write_stock_result(stock_analyzer.to_dict(result))
            print(f"\r{' ' * 80}\r✅ {symbol} 已完成 AI 分析")
            
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
                
                # 收集成功的结果
                if result.success:
                    qualified_stocks.append(stock_analyzer.to_dict(result))
                
            except Exception as e:
                print(f"\r{' ' * 80}\r❌ 处理 {symbol} 的结果时发生错误: {e}")
                progress_tracker.update(success=False)
            
            # 显示进度
            print(progress_tracker.format_status(), end='')
    
    # --- 输出摘要 ---
    print(progress_tracker.get_summary())
    
    # --- 写入摘要列表 ---
    if qualified_stocks:
        report_writer.write_summary(qualified_stocks, market)
        print(f"\n--- 完整报告已储存至 {report_writer.get_html_filename()} ---")
    
    return qualified_stocks
