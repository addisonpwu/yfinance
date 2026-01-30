import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import re
import subprocess
from typing import List, Dict, Optional
from src.data.loaders.yahoo_loader import YahooFinanceRepository, calculate_technical_indicators, optimize_dataframe_memory
from src.core.strategies.strategy import StrategyContext, StrategyEngine
from src.core.strategies.loader import get_strategies
from src.ai.analyzer.service import AIAnalysisService
from src.config.settings import config_manager
from src.utils.logger import get_analysis_logger
from src.utils.exceptions import AnalysisException

def run_analysis(market: str, force_fast_mode: bool = False, skip_strategies: bool = False, symbol_filter: str = None, interval: str = '1d', max_workers: int = None, model: str = 'iflow-rome-30ba3b'):
    """
    對指定市場執行所有選股策略分析

    Args:
        market: 市場代碼 ('US' 或 'HK')
        force_fast_mode: 是否強制跳過緩存更新，直接使用快速模式
        symbol_filter: 指定分析單一股票代碼（例如：0017.HK）
        interval: 數據時段類型 ('1d' 日線, '1h' 小時線, '1m' 分鐘線)
        max_workers: 最大并行工作线程数，默认为None（从配置文件读取）
        model: 要使用的AI模型
    """
    # 加载配置
    config = config_manager.get_config()
    
    # 如果未指定max_workers，从配置中获取
    if max_workers is None:
        max_workers = config.api.max_workers
    
    # 初始化服务
    data_repo = YahooFinanceRepository()
    strategy_engine = StrategyEngine(get_strategies())
    ai_service = AIAnalysisService()
    logger = get_analysis_logger()
    
    # --- 全局緩存版本檢查 ---
    version_file = f"data_cache/{market.upper()}/version.txt"
    today_str = datetime.now().date().isoformat()
    is_sync_needed = True

    if force_fast_mode:
        is_sync_needed = False
        print(f"--- 強制快速模式：跳過緩存更新檢查 ---")
    else:
        try:
            with open(version_file, 'r') as f:
                last_sync_date = f.read().strip()
            if last_sync_date == today_str:
                is_sync_needed = False
                print(f"--- 數據緩存已是最新 ({today_str})，將以快速模式運行 ---")
            else:
                print(f"--- 數據緩存不是最新 (版本: {last_sync_date})，將執行增量同步 ---")
        except FileNotFoundError:
            print(f"--- 未找到緩存版本文件，將執行首次同步 ---")

    # --- 獲取股票列表 ---
    # 先定義 market_ticker
    if market.upper() == 'US':
        market_ticker = '^GSPC'
    elif market.upper() == 'HK':
        market_ticker = '^HSI'
    else:
        print(f"錯誤: 不支援的市場 '{market}'。請使用 'US' 或 'HK'。")
        return []

    if symbol_filter:
        # 如果指定了單一股票，直接使用該股票
        tickers = [symbol_filter]
        print(f"--- 使用指定股票: {symbol_filter} ---")
    else:
        # 否則獲取整個市場的股票列表
        if market.upper() == 'US':
            from src.data_loader import us_loader
            tickers = us_loader.get_us_tickers()
        elif market.upper() == 'HK':
            from src.data_loader import hk_loader
            tickers = hk_loader.get_hk_tickers()

    is_market_healthy = False
    market_latest_return = 0.0
    try:
        market_hist = yf.Ticker(market_ticker).history(period='1y', auto_adjust=True)
        if not market_hist.empty and len(market_hist) >= 200:
            market_latest_return = market_hist['Close'].pct_change().iloc[-1] * 100
            market_hist['MA200'] = market_hist['Close'].rolling(window=200).mean()
            latest_market_data = market_hist.iloc[-1]
            is_market_healthy = latest_market_data['Close'] > latest_market_data['MA200']
            market_status_str = "多頭" if is_market_healthy else "空頭"
            print(f"已成功獲取大盤({market_ticker})數據。今日漲跌: {market_latest_return:.2f}%。市場趨勢: {market_status_str}")
        else:
            print(f"大盤歷史數據不足以計算200MA")
    except Exception as e:
        print(f"無法下載或分析大盤數據 ({market_ticker})，策略中的大盤濾網將不會啟用。錯誤: {e}")

    strategies_to_run = get_strategies()
    if not strategies_to_run:
        print("警告: 在 'strategies' 文件夾中沒有找到任何策略。")
        return []
    print(f"已加載 {len(strategies_to_run)} 個策略: {[s.name for s in strategies_to_run]}")

    # --- 逐個股票進行分析和預測 ---
    print(f"\n--- 開始逐個股票進行分析和預測 ---")
    qualified_stocks = []
    total_stocks = len(tickers)
    
    # 实时输出符合条件的股票到文件
    if config.analysis.enable_realtime_output:
        output_file = f"{datetime.now().strftime('%Y-%m-%d')}_{market.lower()}_qualified_stocks.txt"
    
    # 使用线程池并行处理股票
    def analyze_single_stock(symbol: str, skip_strategies: bool = False, model: str = 'iflow-rome-30ba3b'):
        """分析单个股票的函数
        
        Args:
            symbol: 股票代码
            skip_strategies: 是否跳过策略筛选，所有股票都进行AI分析
            model: 要使用的AI模型名称
        """
        try:
            # 获取股票數據（會自動處理緩存）
            hist = data_repo.get_historical_data(symbol, market, interval=interval)
            info = data_repo.get_financial_info(symbol)
            news = data_repo.get_news(symbol)
            
            # 数据质量检查
            if hist.empty or len(hist) < 2 or info is None or (isinstance(info, dict) and len(info) == 0):
                return None, 0  # 返回None表示该股票未通过筛选，0表示未分析成功
            
            # 数据预处理优化：基础筛选
            if config.analysis.enable_data_preprocessing:
                # 基础数据质量检查
                if 'Volume' in hist.columns and not hist['Volume'].empty:
                    recent_volume = hist['Volume'].tail(5).mean()  # 最近5天平均成交量
                    if recent_volume < config.analysis.min_volume_threshold:
                        return None, 1  # 成交量过低，跳过分析，但计入已分析计数
                
                # 检查价格数据是否有效
                if 'Close' in hist.columns:
                    recent_prices = hist['Close'].tail(10)  # 最近10天价格
                    if recent_prices.isna().all() or (recent_prices <= 0).any():
                        return None, 1  # 价格数据无效，跳过分析
                
                # 检查是否有足够的有效数据点
                if len(hist.dropna()) < 20:  # 至少需要20个有效数据点
                    return None, 1  # 数据点不足，跳过分析
            
            # 預計算技術指標
            hist = calculate_technical_indicators(hist)
            
            # 優化內存使用
            hist = optimize_dataframe_memory(hist)
            
            # 執行所有策略或跳過策略
            passed_strategies = []
            if skip_strategies:
                # 如果跳過策略，則所有股票都標記為通過空策略列表
                passed_strategies = ["跳過策略"]
                print(f"\r{' ' * 80}\r🔍 {symbol} 已跳過策略篩選，直接進行AI分析")
            else:
                # 執行所有策略
                strategy_context = StrategyContext(
                    hist=hist,
                    info=info,
                    market_return=market_latest_return,
                    is_market_healthy=is_market_healthy
                )
                
                strategy_results = strategy_engine.execute_all(strategy_context)
                
                for i, result in enumerate(strategy_results):
                    if result.passed:
                        passed_strategies.append(strategies_to_run[i].name)
            
            # 无论是否跳过策略，只要通过了基础筛选，都需要进行AI分析
            if passed_strategies or skip_strategies:
            
                # 步骤 1: AI 分析
                ai_analysis = None
                
                try:
                    stock_data = {
                        'symbol': symbol,
                        'strategies': passed_strategies,
                        'info': info,
                        'market': market
                    }
                    ai_analysis = ai_service.analyze_stock(stock_data, hist, interval=interval, model=model)
                except Exception as ai_e:
                    print(f" - AI 分析出错: {ai_e}", end='')

                # 将股票添加到结果中（当启用 skip_strategies 时，所有股票都添加）
                # 如果启用了 skip_strategies，所有通过基础筛选的股票都添加到结果中
                exchange = info.get('exchange', 'UNKNOWN')
                stock_result = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'strategies': passed_strategies,
                    'info': info,
                    'news': news,
                    'ai_analysis': {
                        'summary': ai_analysis.summary if ai_analysis else 'N/A',
                        'model_used': ai_analysis.model_used if ai_analysis else 'N/A'
                    } if ai_analysis else None
                }
                
                # 实时输出符合条件的股票
                if config.analysis.enable_realtime_output:
                    with threading.Lock():
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(f"{symbol} 符合策略: {passed_strategies}\n")
                            if ai_analysis:
                                f.write(f"AI 分析: {ai_analysis.summary}\n")
                            f.write("-" * 50 + "\n")
                
                if skip_strategies:
                    print(f"\r{' ' * 80}\r✅ {symbol} 跳過策略篩選，已進行AI分析")
                else:
                    print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {passed_strategies}")
                # 仅输出简要AI分析信息，详细内容在最终报告中显示
                if ai_analysis:
                    print(f"   🤖 AI 分析: 已完成 (模型: {ai_analysis.model_used})")
                else:
                    print(f"   🤖 AI 分析: 未能完成")
                return stock_result, 1
            else:
                return None, 1  # 返回None表示该股票未通过策略，但已分析成功
        except Exception as e:
            print(f"\r{' ' * 80}\r❌ 分析 {symbol} 時發生錯誤: {e}")
            return None, 0  # 返回0表示分析失败

    # 使用线程池并行处理所有股票
    analyzed_count = 0
    qualified_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_symbol = {executor.submit(analyze_single_stock, symbol, skip_strategies, model): symbol for symbol in tickers}
        
        # 处理完成的任务
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result, count = future.result()
                if result is not None:
                    qualified_stocks.append(result)
                    qualified_count += 1
                if count > 0:
                    analyzed_count += count
            except Exception as e:
                print(f"\r{' ' * 80}\r❌ 處理 {symbol} 的結果時發生錯誤: {e}")
            
            # 计算预估完成时间
            elapsed_time = time.time() - start_time
            if analyzed_count > 0:
                avg_time_per_stock = elapsed_time / analyzed_count
                estimated_total_time = avg_time_per_stock * total_stocks
                remaining_time = estimated_total_time - elapsed_time
                remaining_minutes = max(0, int(remaining_time / 60))
            else:
                remaining_minutes = -1  # 未开始计算
            
            # 更新进度
            progress = analyzed_count / total_stocks
            if remaining_minutes >= 0:
                print(f"\r分析進度: [{int(progress * 20) * '#'}{int((1 - progress) * 20) * '-'}] {analyzed_count}/{total_stocks} 已分析, {qualified_count} 符合條件, 預估剩餘: {remaining_minutes} 分鐘", end='')
            else:
                print(f"\r分析進度: [{int(progress * 20) * '#'}{int((1 - progress) * 20) * '-'}] {analyzed_count}/{total_stocks} 已分析, {qualified_count} 符合條件", end='')

    # --- 更新緩存版本文件 ---
    if is_sync_needed:
        print(f"\n--- 更新緩存版本至 {today_str} ---")
        with open(version_file, 'w') as f:
            f.write(today_str)
    
    print(f"\n--- 分析完成！成功分析 {analyzed_count}/{total_stocks} 支股票，找到 {len(qualified_stocks)} 支符合條件的股票 ---")
    print(f"--- 總耗時: {int((time.time() - start_time) / 60)} 分鐘 {int((time.time() - start_time) % 60)} 秒 ---")
    return qualified_stocks