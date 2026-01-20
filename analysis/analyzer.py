import yfinance as yf
import pandas as pd
import pkgutil
import importlib
import inspect
import os
import json
import subprocess
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

def get_data_with_cache(symbol: str, market: str, fast_mode: bool = False) -> (pd.DataFrame, dict, dict):
    """
    獲取股票數據，根據模式選擇快速加載或同步更新。
    """
    cache_dir = os.path.join('data_cache', market.upper())
    safe_symbol = symbol.replace(":", "_")
    csv_file = os.path.join(cache_dir, f"{safe_symbol}.csv")
    json_file = os.path.join(cache_dir, f"{safe_symbol}.json")
    
    ticker = yf.Ticker(symbol)

    if fast_mode:
        try:
            # print(f" - [快速模式] 從緩存加載", end='')
            hist = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
            with open(json_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            news = ticker.news # 新聞總是獲取最新的
            return hist, info, news
        except FileNotFoundError:
            # print(f" - [快速模式] 緩存文件未找到，切換到正常模式下載", end='')
            return get_data_with_cache(symbol, market, fast_mode=False)

    # --- 正常同步模式 ---
    today = datetime.now().date()
    hist, info, news = pd.DataFrame(), {}, []

    try:
        info = ticker.info
        news = ticker.news
    except Exception as e:
        print(f" - 無法獲取 {symbol} 的 info/news: {e}", end='')

    if os.path.exists(csv_file):
        hist = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        last_cached_date = hist.index.max().date()
        
        if last_cached_date >= today:
            print(f" - 從緩存加載 {len(hist)} 條數據", end='')
        else:
            start_date = last_cached_date + timedelta(days=1)
            print(f" - 緩存數據過舊，正在從 {start_date.strftime('%Y-%m-%d')} 下載增量數據...", end='')
            new_hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), auto_adjust=True)
            if not new_hist.empty:
                hist = pd.concat([hist, new_hist])
                print(f"下載了 {len(new_hist)} 條新數據", end='')
            else:
                print("沒有新的數據可下載", end='')
    else:
        print(" - 緩存不存在，正在下載全部歷史數據...", end='')
        hist = ticker.history(period="max", auto_adjust=True)
        print(f"下載了 {len(hist)} 條數據", end='')

    # 無論是更新還是新增，都用最新的數據覆蓋緩存
    if not hist.empty:
        float_shares = info.get('floatShares', None)
        hist['FloatShares'] = float_shares
        hist.to_csv(csv_file)
    
    if info:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
            
    return hist, info, news

def run_analysis(market: str, force_fast_mode: bool = False):
    """
    對指定市場執行所有選股策略分析
    
    Args:
        market: 市場代碼 ('US' 或 'HK')
        force_fast_mode: 是否強制跳過緩存更新，直接使用快速模式
    """
    # --- 全局緩存版本檢查 ---
    version_file = os.path.join('data_cache', market.upper(), 'version.txt')
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
            print("--- 未找到緩存版本文件，將執行首次同步 ---")

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
    analyzed_count = 0
    
    for i, symbol in enumerate(tickers):
        progress = (i + 1) / total_stocks
        print(f"\r分析進度: [{int(progress * 20) * '#'}{int((1 - progress) * 20) * '-'}] {i+1}/{total_stocks} - 正在分析 {symbol}...", end='')

        try:
            # 獲取股票數據（會自動處理緩存）
            hist, info, news = get_data_with_cache(symbol, market, fast_mode=not is_sync_needed)
            
            if hist.empty or len(hist) < 2 or not info:
                continue
            
            analyzed_count += 1
            
            # 執行所有策略
            passed_strategies = []
            for strategy in strategies_to_run:
                if strategy.run(hist.copy(), info=info, market_return=market_latest_return, is_market_healthy=is_market_healthy):
                    passed_strategies.append(strategy.name)
            
            if passed_strategies:
                # 調用 Kronos 預測（僅港股）
                kronos_prediction = "N/A"
                KRONOS_SCRIPT_PATH = "/Users/addison/Develop/yfinace/Kronos/scripts/prediction_hk.py"

                if market.upper() == 'HK':
                    try:
                        command = ["python3", KRONOS_SCRIPT_PATH, symbol]
                        process = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=300
                        )
                        kronos_prediction = process.stdout.strip()
                    except subprocess.CalledProcessError as e:
                        error_output = e.stderr.strip()
                        kronos_prediction = f"預測失敗: {error_output}"
                    except subprocess.TimeoutExpired:
                        kronos_prediction = "預測超時"
                    except Exception as pred_e:
                        kronos_prediction = f"調用外部腳本時出錯: {pred_e}"

                exchange = info.get('exchange', 'UNKNOWN')
                qualified_stocks.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'strategies': passed_strategies,
                    'info': info,
                    'news': news,
                    'kronos_prediction': kronos_prediction
                })
                print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {passed_strategies}, Kronos預測: {kronos_prediction}")

        except Exception as e:
            print(f"\r{' ' * 80}\r❌ 分析 {symbol} 時發生錯誤: {e}")
            pass
    
    # --- 更新緩存版本文件 ---
    if is_sync_needed:
        print(f"\n--- 更新緩存版本至 {today_str} ---")
        with open(version_file, 'w') as f:
            f.write(today_str)
    
    print(f"\n--- 分析完成！成功分析 {analyzed_count}/{total_stocks} 支股票，找到 {len(qualified_stocks)} 支符合條件的股票 ---")
    return qualified_stocks