






import yfinance as yf



import pandas as pd



import pkgutil



import importlib



import inspect



import os



from datetime import datetime, timedelta



from strategies.base_strategy import BaseStrategy



from data_loader import us_loader, hk_loader







def get_strategies():



    """



    動態從 strategies 模組加載所有策略類別的實例。



    """



    strategies = []



    import strategies as strategies_module



    strategy_path = strategies_module.__path__







    for _, name, _ in pkgutil.iter_modules(strategy_path):



        if name != 'base_strategy':



            module = importlib.import_module(f"strategies.{name}")



            for item_name, item in inspect.getmembers(module, inspect.isclass):



                if issubclass(item, BaseStrategy) and item is not BaseStrategy:



                    strategies.append(item())



    return strategies







def get_data_with_cache(symbol: str, market: str) -> (pd.DataFrame, dict, dict):



    """



    獲取股票數據（歷史、基本面、新聞），優先使用本地緩存，必要時下載並更新緩存。







    :param symbol: 股票代碼



    :param market: 市場 ('US' 或 'HK')



    :return: (DataFrame, dict, dict) -> (hist, info, news)



    """



    cache_dir = os.path.join('data_cache', market.upper())



    safe_symbol = symbol.replace(":", "_")



    cache_file = os.path.join(cache_dir, f"{safe_symbol}.csv")



    



    today = datetime.now().date()



    ticker = yf.Ticker(symbol)



    hist, info, news = pd.DataFrame(), {}, []







    try:



        info = ticker.info



        news = ticker.news



    except Exception as e:



        print(f" - 無法獲取 {symbol} 的 info/news: {e}", end='')



        # 即使info獲取失敗，我們仍然可以嘗試獲取歷史數據







    if os.path.exists(cache_file):



        hist = pd.read_csv(cache_file, index_col='Date', parse_dates=True)



        last_cached_date = hist.index.max().date()



        



        if last_cached_date >= today:



            print(f" - 從緩存加載 {len(hist)} 條數據", end='')



        else:



            start_date = last_cached_date + timedelta(days=1)



            print(f" - 緩存數據過舊，正在從 {start_date.strftime('%Y-%m-%d')} 下載增量數據...", end='')



            



            new_hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), auto_adjust=True)



            if not new_hist.empty:



                float_shares = info.get('floatShares', None)



                if 'FloatShares' not in hist.columns:



                    hist['FloatShares'] = None



                



                hist = pd.concat([hist, new_hist])



                hist['FloatShares'] = float_shares



                hist.to_csv(cache_file)



                print(f"下載了 {len(new_hist)} 條新數據", end='')



            else:



                print("沒有新的數據可下載", end='')



    else:



        print(" - 緩存不存在，正在下載全部歷史數據...", end='')



        hist = ticker.history(period="max", auto_adjust=True)



        if not hist.empty:



            float_shares = info.get('floatShares', None)



            hist['FloatShares'] = float_shares



            hist.to_csv(cache_file)



            print(f"下載了 {len(hist)} 條數據", end='')



            



    return hist, info, news







def run_analysis(market: str):



    """



    對指定市場執行所有選股策略分析



    """



    if market.upper() == 'US':



        tickers = us_loader.get_us_tickers()



        market_ticker = '^GSPC'



    elif market.upper() == 'HK':



        tickers = hk_loader.get_hk_tickers()



        market_ticker = '^HSI'



    else:



        print(f"錯誤: 不支援的市場 '{market}'。請使用 'US' 或 'HK'。")



        return []







    is_market_healthy = False



    market_latest_return = 0.0



    try:



        market_hist = yf.Ticker(market_ticker).history(period='1y', auto_adjust=True)



        if market_hist.empty or len(market_hist) < 200:



            raise ValueError("大盤歷史數據不足以計算200MA")



        



        market_latest_return = market_hist['Close'].pct_change().iloc[-1] * 100



        market_hist['MA200'] = market_hist['Close'].rolling(window=200).mean()



        latest_market_data = market_hist.iloc[-1]



        is_market_healthy = latest_market_data['Close'] > latest_market_data['MA200']



        



        market_status_str = "多頭" if is_market_healthy else "空頭"



        print(f"已成功獲取大盤({market_ticker})數據。今日漲跌: {market_latest_return:.2f}%。市場趨勢: {market_status_str}")







    except Exception as e:



        print(f"無法下載或分析大盤數據 ({market_ticker})，策略中的大盤濾網將不會啟用。錯誤: {e}")







    strategies_to_run = get_strategies()



    if not strategies_to_run:



        print("警告: 在 'strategies' 文件夾中沒有找到任何策略。")



        return []



    print(f"已加載 {len(strategies_to_run)} 個策略: {[s.name for s in strategies_to_run]}")







    qualified_stocks = []



    total_stocks = len(tickers)







    for i, symbol in enumerate(tickers):



        progress = (i + 1) / total_stocks



        print(f"\r進度: [{int(progress * 20) * '#'}{int((1 - progress) * 20) * '-'}] {i+1}/{total_stocks} - 正在分析 {symbol}...", end='')







        try:



            hist, info, news = get_data_with_cache(symbol, market)



            



            if hist.empty or len(hist) < 2 or not info:



                continue







            passed_strategies = []



            for strategy in strategies_to_run:



                if strategy.run(hist.copy(), info=info, market_return=market_latest_return, is_market_healthy=is_market_healthy):



                    passed_strategies.append(strategy.name)



            



            if passed_strategies:



                exchange = info.get('exchange', 'UNKNOWN')



                qualified_stocks.append({



                    'symbol': symbol,



                    'exchange': exchange,



                    'strategies': passed_strategies,



                    'info': info,



                    'news': news



                })



                print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {passed_strategies}")







        except Exception as e:



            print(f"\r{' ' * 80}\r❌ 分析 {symbol} 時發生錯誤: {e}")



            pass



            



    print("\n分析完成！")



    return qualified_stocks






