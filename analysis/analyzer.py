import yfinance as yf
import pandas as pd
import pkgutil
import importlib
import inspect
import os
import json
import subprocess
import re
import time
from datetime import datetime, timedelta, date
from strategies.base_strategy import BaseStrategy
from data_loader import us_loader, hk_loader
from ai_analyzer import analyze_stock_with_ai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np

def parse_kronos_prediction(prediction_text: str) -> tuple[float, float]:
    """
    解析 Kronos 预测输出，提取上升和下跌机率

    Args:
        prediction_text: Kronos 预测脚本的输出文本

    Returns:
        (上升机率, 下跌机率) 的元组，如果解析失败返回 (0, 0)
    """
    try:
        # 使用正则表达式提取机率
        rise_match = re.search(r'價格上升機率:\s*([\d.]+)%', prediction_text)
        fall_match = re.search(r'價格下跌機率:\s*([\d.]+)%', prediction_text)

        if rise_match and fall_match:
            rise_prob = float(rise_match.group(1))
            fall_prob = float(fall_match.group(1))
            return rise_prob, fall_prob
        else:
            return 0.0, 0.0
    except Exception as e:
        print(f"解析 Kronos 预测机率时出错: {e}")
        return 0.0, 0.0

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

def _read_csv_with_auto_index(csv_file: str) -> pd.DataFrame:
    """
    读取 CSV 文件，自动检测并使用正确的索引列名（Date 或 Datetime）

    Args:
        csv_file: CSV 文件路径

    Returns:
        DataFrame
    """
    # 先读取第一行来检测列名
    with open(csv_file, 'r') as f:
        first_line = f.readline()
    
    # 检测索引列名
    if 'Datetime,' in first_line:
        index_col = 'Datetime'
    else:
        index_col = 'Date'
    
    # 使用正确的索引列名读取
    return pd.read_csv(csv_file, index_col=index_col, parse_dates=True)

def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return {
            "api": {
                "base_delay": 0.5,
                "max_delay": 2.0,
                "min_delay": 0.1,
                "retry_attempts": 3,
                "max_workers": 4
            },
            "data": {
                "max_cache_days": 7,
                "float_dtype": "float32"
            },
            "analysis": {
                "enable_realtime_output": true,
                "enable_data_preprocessing": true,
                "min_volume_threshold": 100000
            }
        }

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化 DataFrame 的内存使用，通过使用更高效的数据类型

    Args:
        df: 原始 DataFrame

    Returns:
        优化后的 DataFrame
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.float64)
    
    return df_optimized

def serialize_for_json(obj):
    """
    将对象转换为可 JSON 序列化的格式（递归处理所有层级）

    Args:
        obj: 要序列化的对象

    Returns:
        可 JSON 序列化的对象
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, date

    # 首先检查是否为 NaN（标量）
    try:
        if isinstance(obj, (float, int)) and np.isnan(obj):
            return None
    except (TypeError, ValueError):
        pass

    # 处理日期时间对象
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()

    # 处理 Series 或 DataFrame - 先转换为字典，然后递归处理
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        result_dict = obj.to_dict()
        return serialize_for_json(result_dict)  # 关键：递归处理转换后的字典

    # 处理字典 - 递归处理键和值
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # 处理键
            if isinstance(k, (pd.Timestamp, datetime, date)):
                k = k.isoformat()
            # 递归处理值
            result[k] = serialize_for_json(v)
        return result

    # 处理列表/元组 - 递归处理每个元素
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]

    # 处理 numpy 类型
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # 其他类型直接返回
    else:
        return obj

def get_enhanced_financial_data(ticker: yf.Ticker) -> dict:
    """
    获取增强的财务数据，包括财务报表和关键指标

    Args:
        ticker: yfinance Ticker 对象

    Returns:
        包含增强财务数据的字典
    """
    enhanced_data = {}

    try:
        # 只获取最关键的财务报表数据，减少 API 调用
        financials = ticker.financials
        if financials is not None and isinstance(financials, pd.DataFrame) and not financials.empty:
            enhanced_data['financials'] = financials.to_dict()

        # 获取资产负债表
        balance_sheet = ticker.balance_sheet
        if balance_sheet is not None and isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
            enhanced_data['balance_sheet'] = balance_sheet.to_dict()

        # 获取现金流量表
        cashflow = ticker.cashflow
        if cashflow is not None and isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
            enhanced_data['cashflow'] = cashflow.to_dict()

    except Exception as e:
        print(f" - [增强数据] 获取失败: {e}", end='')

    return enhanced_data

def get_data_with_cache(symbol: str, market: str, fast_mode: bool = False, interval: str = '1d') -> (pd.DataFrame, dict, dict):
    """
    獲取股票數據，根據模式選擇快速加載或同步更新。

    Args:
        symbol: 股票代碼
        market: 市場代碼 ('US' 或 'HK')
        fast_mode: 是否使用快速模式
        interval: 數據時段類型 ('1d' 日線, '1h' 小時線, '1m' 分鐘線)
    """
    cache_dir = os.path.join('data_cache', market.upper())
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.replace(":", "_")
    csv_file = os.path.join(cache_dir, f"{safe_symbol}_{interval}.csv")  # 添加 interval 到文件名
    json_file = os.path.join(cache_dir, f"{safe_symbol}.json")

    ticker = yf.Ticker(symbol)

    if fast_mode:
        try:
            # print(f" - [快速模式] 從緩存加載", end='')
            # 自动检测索引列名（Date 或 Datetime）
            hist = _read_csv_with_auto_index(csv_file)
            with open(json_file, 'r', encoding='utf-8') as f:
                info = json.load(f)

            # 确保 info 是字典
            if not isinstance(info, dict):
                info = {}

            # 验证关键字段
            required_fields = [
                'marketCap', 'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook',
                'profitMargins', 'returnOnEquity', 'revenueGrowth', 'earningsGrowth',
                'dividendYield', 'beta', '52WeekChange', 'targetMeanPrice',
                'volume', 'floatShares', 'shortRatio'
            ]
            for field in required_fields:
                if field not in info:
                    info[field] = None

            # 移除 news 调用以减少 API 请求
            news = []
            # 优化内存使用 - 转换数据类型
            hist = optimize_dataframe_memory(hist)
            return hist, info, news
        except FileNotFoundError:
            # print(f" - [快速模式] 緩存文件未找到，切換到正常模式下載", end='')
            return get_data_with_cache(symbol, market, fast_mode=False, interval=interval)
        except (json.JSONDecodeError, ValueError) as e:
            print(f" - [快速模式] JSON 解析失败: {e}，重新下載", end='')
            # 删除损坏的缓存文件
            try:
                os.remove(json_file)
            except:
                pass
            return get_data_with_cache(symbol, market, fast_mode=False, interval=interval)

    # --- 正常同步模式 ---
    today = datetime.now().date()
    hist, info, news = pd.DataFrame(), {}, []

    # 首先获取历史价格数据（这个通常比 info 更容易获取）
    if os.path.exists(csv_file):
        # 自动检测索引列名（Date 或 Datetime）
        hist = _read_csv_with_auto_index(csv_file)
        last_cached_date = hist.index.max().date()

        if last_cached_date >= today:
            print(f" - 從緩存加載 {len(hist)} 條數據", end='')
        else:
            start_date = last_cached_date + timedelta(days=1)
            print(f" - 緩存數據過舊，正在從 {start_date.strftime('%Y-%m-%d')} 下載增量數據...", end='')
            new_hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), interval=interval, auto_adjust=True)
            if not new_hist.empty:
                hist = pd.concat([hist, new_hist])
                print(f"下載了 {len(new_hist)} 條新數據", end='')
            else:
                print("沒有新的數據可下載", end='')
    else:
        print(" - 緩存不存在，正在下載全部歷史數據...", end='')
        # 根據 interval 設置不同的 period
        if interval == '1m':
            period = '7d'  # 分鐘線只下載最近7天
        elif interval == '1h':
            period = '730d'  # 小時線下載最近2年
        else:
            period = 'max'  # 日線下載全部歷史
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        print(f"下載了 {len(hist)} 條數據", end='')

    # 优化内存使用 - 转换数据类型
    if not hist.empty:
        hist = optimize_dataframe_memory(hist)
        float_shares = None  # 暂时设置为 None
        hist['FloatShares'] = float_shares
        hist.to_csv(csv_file)

    # 尝试获取 info 数据（如果失败则使用空字典）
    try:
        info = ticker.info
        # 确保 info 不为空 - 使用更安全的方式
        if info is None:
            print(f" - info 数据为空", end='')
            info = {}
        elif not isinstance(info, dict):
            # 如果 info 不是字典，转换为字典
            print(f" - info 格式异常，转换为字典", end='')
            info = {}
        elif isinstance(info, dict) and len(info) == 0:
            print(f" - info 字典为空", end='')
            # 保持为空字典，继续尝试获取增强数据

        # 验证关键字段是否存在，如果不存在则设置为 None
        required_fields = [
            'marketCap', 'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook',
            'profitMargins', 'returnOnEquity', 'revenueGrowth', 'earningsGrowth',
            'dividendYield', 'beta', '52WeekChange', 'targetMeanPrice',
            'volume', 'floatShares', 'shortRatio'
        ]
        for field in required_fields:
            if field not in info:
                info[field] = None

        # 获取增强的财务数据
        enhanced_data = get_enhanced_financial_data(ticker)
        if enhanced_data:
            info['enhanced_financial_data'] = enhanced_data

        # 保存 info 到缓存 - 只在有有效数据时保存
        if isinstance(info, dict) and len(info) > 0:
            try:
                # 使用递归的 serialize_for_json 处理所有嵌套层级
                processed_info = serialize_for_json(info)

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_info, f, ensure_ascii=False, indent=4)
            except Exception as save_error:
                print(f" - 保存 info 失败: {save_error}", end='')
                # 保存失败不影响主流程

    except Exception as e:
        print(f" - 無法獲取 info: {e}，將使用空數據", end='')
        info = {}

    news = []

    if os.path.exists(csv_file):
        # 自动检测索引列名（Date 或 Datetime）
        hist = _read_csv_with_auto_index(csv_file)
        last_cached_date = hist.index.max().date()

        if last_cached_date >= today:
            print(f" - 從緩存加載 {len(hist)} 條數據", end='')
        else:
            start_date = last_cached_date + timedelta(days=1)
            print(f" - 緩存數據過舊，正在從 {start_date.strftime('%Y-%m-%d')} 下載增量數據...", end='')
            new_hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), interval=interval, auto_adjust=True)
            if not new_hist.empty:
                hist = pd.concat([hist, new_hist])
                print(f"下載了 {len(new_hist)} 條新數據", end='')
            else:
                print("沒有新的數據可下載", end='')
    else:
        print(" - 緩存不存在，正在下載全部歷史數據...", end='')
        # 根據 interval 設置不同的 period
        if interval == '1m':
            period = '7d'  # 分鐘線只下載最近7天
        elif interval == '1h':
            period = '730d'  # 小時線下載最近2年
        else:
            period = 'max'  # 日線下載全部歷史
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        print(f"下載了 {len(hist)} 條數據", end='')

    # 优化内存使用 - 转换数据类型
    if not hist.empty:
        hist = optimize_dataframe_memory(hist)
        float_shares = info.get('floatShares', None)
        hist['FloatShares'] = float_shares
        hist.to_csv(csv_file)

    if info:
        # 使用递归的 serialize_for_json 处理所有嵌套层级
        processed_info = serialize_for_json(info)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(processed_info, f, ensure_ascii=False, indent=4)

    return hist, info, news

def run_analysis(market: str, force_fast_mode: bool = False, use_kronos: bool = True, symbol_filter: str = None, interval: str = '1d', max_workers: int = None):
    """
    對指定市場執行所有選股策略分析

    Args:
        market: 市場代碼 ('US' 或 'HK')
        force_fast_mode: 是否強制跳過緩存更新，直接使用快速模式
        use_kronos: 是否使用 Kronos 預測（僅適用於港股）
        symbol_filter: 指定分析單一股票代碼（例如：0017.HK）
        interval: 數據時段類型 ('1d' 日線, '1h' 小時線, '1m' 分鐘線)
        max_workers: 最大并行工作线程数，默认为None（从配置文件读取）
    """
    # 加载配置
    config = load_config()
    
    # 如果未指定max_workers，从配置中获取
    if max_workers is None:
        max_workers = config['api']['max_workers']
    
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
            tickers = us_loader.get_us_tickers()
        elif market.upper() == 'HK':
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
    realtime_output_enabled = config['analysis']['enable_realtime_output']
    if realtime_output_enabled:
        output_file = f"{datetime.now().strftime('%Y-%m-%d')}_{market.lower()}_qualified_stocks.txt"
    
    # 使用线程池并行处理股票
    def analyze_single_stock(symbol):
        """分析单个股票的函数"""
        try:
            # 获取股票數據（會自動處理緩存）
            hist, info, news = get_data_with_cache(symbol, market, fast_mode=not is_sync_needed, interval=interval)
            
            # 数据质量检查
            if hist.empty or len(hist) < 2 or info is None or (isinstance(info, dict) and len(info) == 0):
                return None, 0  # 返回None表示该股票未通过筛选，0表示未分析成功
            
            # 数据预处理优化：基础筛选
            config = load_config()
            enable_preprocessing = config['analysis']['enable_data_preprocessing']
            min_volume_threshold = config['analysis']['min_volume_threshold']
            
            if enable_preprocessing:
                # 基础数据质量检查
                if 'Volume' in hist.columns and not hist['Volume'].empty:
                    recent_volume = hist['Volume'].tail(5).mean()  # 最近5天平均成交量
                    if recent_volume < min_volume_threshold:
                        return None, 1  # 成交量过低，跳过分析，但计入已分析计数
                
                # 检查价格数据是否有效
                if 'Close' in hist.columns:
                    recent_prices = hist['Close'].tail(10)  # 最近10天价格
                    if recent_prices.isna().all() or (recent_prices <= 0).any():
                        return None, 1  # 价格数据无效，跳过分析
                
                # 检查是否有足够的有效数据点
                if len(hist.dropna()) < 20:  # 至少需要20个有效数据点
                    return None, 1  # 数据点不足，跳过分析
            
            # 執行所有策略
            passed_strategies = []
            for strategy in strategies_to_run:
                if strategy.run(hist.copy(), info=info, market_return=market_latest_return, is_market_healthy=is_market_healthy):
                    passed_strategies.append(strategy.name)
            
            if passed_strategies:
                # 步骤 1: AI 分析（在 Kronos 预测之前）
                ai_analysis = None
                try:
                    ai_analysis = analyze_stock_with_ai({
                        'symbol': symbol,
                        'strategies': passed_strategies,
                        'info': info,
                        'market': market
                    }, hist, interval)
                except Exception as ai_e:
                    print(f" - AI 分析出错: {ai_e}", end='')

                # 步骤 2: 调用 Kronos 预测（仅港股且启用 Kronos）
                kronos_prediction = "N/A"
                rise_prob = 0.0
                fall_prob = 0.0
                KRONOS_SCRIPT_PATH = "/Users/addison/Develop/yfinace/Kronos/scripts/prediction_hk.py"

                if market.upper() == 'HK' and use_kronos:
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
                        # 解析上升/下跌机率
                        rise_prob, fall_prob = parse_kronos_prediction(kronos_prediction)
                    except subprocess.CalledProcessError as e:
                        error_output = e.stderr.strip()
                        kronos_prediction = f"預測失敗: {error_output}"
                    except subprocess.TimeoutExpired:
                        kronos_prediction = "預測超時"
                    except Exception as pred_e:
                        kronos_prediction = f"調用外部腳本時出錯: {pred_e}"

                # 步骤 3: 仅当上升机率 > 下跌机率时才加入 qualified_stocks（如果启用了 Kronos）
                # 如果未启用 Kronos，则直接加入 qualified_stocks
                if not use_kronos or rise_prob > fall_prob:
                    exchange = info.get('exchange', 'UNKNOWN')
                    stock_result = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'strategies': passed_strategies,
                        'info': info,
                        'news': news,
                        'kronos_prediction': kronos_prediction,
                        'rise_prob': rise_prob,
                        'fall_prob': fall_prob,
                        'ai_analysis': ai_analysis
                    }
                    
                    # 实时输出符合条件的股票
                    if realtime_output_enabled:
                        with threading.Lock():
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(f"{symbol} 符合策略: {passed_strategies}\n")
                                if ai_analysis:
                                    f.write(f"AI 分析: {ai_analysis['summary']}\n")
                                f.write("-" * 50 + "\n")
                    
                    if use_kronos:
                        print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {passed_strategies}, 上升機率: {rise_prob:.2f}% vs 下跌機率: {fall_prob:.2f}%")
                    else:
                        print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {passed_strategies}")
                    # 输出 AI 分析结果到 console
                    if ai_analysis:
                        print(f"   🤖 AI 分析: {ai_analysis['summary']}")
                        print(f"   🤖 AI 模型: {ai_analysis['model_used']}")
                    else:
                        print(f"   🤖 AI 分析: 未能完成")
                    return stock_result, 1
                else:
                    print(f"\r{' ' * 80}\r⏭️  {symbol} 符合策略但上升機率({rise_prob:.2f}%) ≤ 下跌機率({fall_prob:.2f}%)，已跳過")
                    return None, 1
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
        future_to_symbol = {executor.submit(analyze_single_stock, symbol): symbol for symbol in tickers}
        
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