
import yfinance as yf
import pandas as pd
import pkgutil
import importlib
import inspect
from strategies.base_strategy import BaseStrategy
from data_loader import us_loader, hk_loader

def get_strategies():
    """
    動態從 strategies 模組加載所有策略類別的實例。
    """
    strategies = []
    # 我們需要一個路徑來給 pkgutil.iter_modules
    # __path__ 是一個列表，所以我們取第一個元素
    import strategies as strategies_module
    strategy_path = strategies_module.__path__

    for _, name, _ in pkgutil.iter_modules(strategy_path):
        if name != 'base_strategy':
            module = importlib.import_module(f"strategies.{name}")
            for item_name, item in inspect.getmembers(module, inspect.isclass):
                # 確保它是一個具體的策略類別，而不是基礎類別或導入的其他類別
                if issubclass(item, BaseStrategy) and item is not BaseStrategy:
                    strategies.append(item()) # 創建策略的實例
    return strategies

def run_analysis(market: str):
    """
    對指定市場執行所有選股策略分析

    :param market: 要分析的市場 ('US' 或 'HK')
    :return: 一個包含符合條件股票及其通過策略的列表
    """
    if market.upper() == 'US':
        tickers = us_loader.get_us_tickers()
        market_ticker = '^GSPC' # S&P 500 作為美股基準
    elif market.upper() == 'HK':
        tickers = hk_loader.get_hk_tickers()
        market_ticker = '^HSI'  # 恆生指數作為港股基準
    else:
        print(f"錯誤: 不支援的市場 '{market}'。請使用 'US' 或 'HK'。")
        return []

    # 獲取大盤基準數據
    is_market_healthy = False # 預設為不健康
    market_latest_return = 0.0
    try:
        # 需要更長的歷史數據來計算200MA
        market_hist = yf.Ticker(market_ticker).history(period='max', auto_adjust=True)
        if market_hist.empty or len(market_hist) < 200:
            raise ValueError("大盤歷史數據不足以計算200MA")
        
        market_latest_return = market_hist['Close'].pct_change().iloc[-1] * 100
        
        # 計算大盤200日均線
        market_hist['MA200'] = market_hist['Close'].rolling(window=200).mean()
        latest_market_data = market_hist.iloc[-1]
        
        is_market_healthy = latest_market_data['Close'] > latest_market_data['MA200']
        
        market_status_str = "多頭" if is_market_healthy else "空頭"
        print(f"已成功獲取大盤({market_ticker})數據。今日漲跌: {market_latest_return:.2f}%。市場趨勢: {market_status_str}")

    except Exception as e:
        print(f"無法下載或分析大盤數據 ({market_ticker})，策略中的大盤濾網將不會啟用。錯誤: {e}")

    # 動態加載所有策略
    strategies_to_run = get_strategies()
    if not strategies_to_run:
        print("警告: 在 'strategies' 文件夾中沒有找到任何策略。")
        return []
    print(f"已加載 {len(strategies_to_run)} 個策略: {[s.name for s in strategies_to_run]}")

    qualified_stocks = []
    total_stocks = len(tickers)

    for i, symbol in enumerate(tickers):
        # 簡單的進度條
        progress = (i + 1) / total_stocks
        print(f"\r進度: [{int(progress * 20) * '#'}{int((1 - progress) * 20) * '-'}] {i+1}/{total_stocks} - 正在分析 {symbol}...", end='')

        try:
            hist = yf.Ticker(symbol).history(period="max", auto_adjust=True)
            print(f" - {len(hist)} 條數據", end='')
            if hist.empty or len(hist) < 2:
                continue

            passed_strategies = []
            for strategy in strategies_to_run:
                # 傳遞 hist 的副本以避免副作用
                if strategy.run(hist.copy(), market_return=market_latest_return, is_market_healthy=is_market_healthy):
                    passed_strategies.append(strategy.name)
            
            if passed_strategies:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                news = ticker.news
                exchange = info.get('exchange', 'UNKNOWN')
                qualified_stocks.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'strategies': passed_strategies,
                    'info': info,
                    'news': news
                })
                # 清除進度條並打印結果
                print(f"\r{'' * 80}\r✅ {symbol} 符合策略: {passed_strategies}")

        except Exception:
            # 在分析單支股票時出錯，可以選擇忽略
            pass
            
    print("\n分析完成！")
    return qualified_stocks
