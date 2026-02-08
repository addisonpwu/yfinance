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
from src.strategies.base_strategy import BaseStrategy
from src.data_loader import us_loader, hk_loader
from src.ai.analyzer.ai_analyzer import analyze_stock_with_ai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import logging

# 配置日志
def setup_logging():
    """设置日志配置"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"analyzer_{datetime.now().strftime('%Y-%m-%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)


def get_strategies():
    """
    動態從 strategies 模組加載所有策略類別的實例。
    """
    strategies = []
    import src.strategies as strategies_module
    strategy_path = strategies_module.__path__

    for _, name, _ in pkgutil.iter_modules(strategy_path):
        if name != 'base_strategy':
            module = importlib.import_module(f"src.strategies.{name}")
            for item_name, item in inspect.getmembers(module, inspect.isclass):
                if issubclass(item, BaseStrategy) and item is not BaseStrategy:
                    strategies.append(item())
    
    if not strategies:
        print("警告: 没有找到任何策略，系统将只执行AI分析")
                
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

def validate_config(config):
    """
    验证配置值是否在合理范围内
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 当配置值不在合理范围内时抛出异常
    """
    errors = []
    
    # API配置验证
    api_config = config['api']
    if api_config['min_delay'] < 0:
        errors.append("min_delay 不能为负数")
    if api_config['max_delay'] < api_config['min_delay']:
        errors.append("max_delay 不能小于 min_delay")
    if api_config['base_delay'] < api_config['min_delay'] or api_config['base_delay'] > api_config['max_delay']:
        errors.append("base_delay 应在 min_delay 和 max_delay 之间")
    if api_config['retry_attempts'] < 0:
        errors.append("retry_attempts 不能为负数")
    if api_config['max_workers'] <= 0:
        errors.append("max_workers 必须大于0")
    
    # 数据配置验证
    if config['data']['max_cache_days'] <= 0:
        errors.append("max_cache_days 必须大于0")
    
    # 分析配置验证
    if config['analysis']['min_volume_threshold'] < 0:
        errors.append("min_volume_threshold 不能为负数")
    
    if errors:
        raise ValueError(f"配置验证失败: {'; '.join(errors)}")
    
    return True

# 配置缓存变量
_config_cache = None
_config_timestamp = None

def load_config(cached=True):
    """
    加载配置文件
    
    Args:
        cached: 是否使用缓存的配置，默认为True
    """
    global _config_cache, _config_timestamp
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    # 如果启用缓存且缓存存在，检查文件是否被修改
    if cached and _config_cache:
        try:
            import stat
            mtime = os.path.getmtime(config_path)
            if _config_timestamp and mtime <= _config_timestamp:
                return _config_cache
        except:
            pass  # 如果文件不存在或其他错误，继续加载
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 确保所有必要的配置项都存在，如果不存在则使用默认值
        default_config = {
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
                "enable_realtime_output": True,
                "enable_data_preprocessing": True,
                "min_volume_threshold": 100000
            }
        }
        
        # 合并配置：使用配置文件中的值，对于缺失的配置使用默认值
        for section, section_data in default_config.items():
            if section not in config:
                config[section] = {}
            for key, value in section_data.items():
                if key not in config[section]:
                    config[section][key] = value
        
        # 验证配置
        validate_config(config)
        
        # 更新缓存
        _config_cache = config
        _config_timestamp = os.path.getmtime(config_path) if os.path.exists(config_path) else None
        
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        default_config = {
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
                "enable_realtime_output": True,
                "enable_data_preprocessing": True,
                "min_volume_threshold": 100000
            }
        }
        
        # 验证默认配置
        validate_config(default_config)
        
        # 更新缓存
        _config_cache = default_config
        _config_timestamp = None
        
        return default_config

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

class APIDelayer:
    """
    API延迟管理类，实现智能延迟策略
    """
    def __init__(self, config):
        self.config = config['api']
        self.last_response_time = time.time()
        self.failure_count = 0
        self.successful_requests = 0
        self.total_delay = 0

    def calculate_delay(self, is_failure=False):
        """
        计算API延迟时间
        
        Args:
            is_failure: 是否是失败请求，如果是则增加延迟
        """
        base_delay = self.config['base_delay']
        
        # 基于失败次数的指数退避
        if is_failure:
            self.failure_count += 1
            current_delay = base_delay * (1.5 ** min(self.failure_count, 5))  # 最大退避5次
        else:
            # 成功请求时减少失败计数器（但不重置）
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 0.1)
            current_delay = base_delay * (0.95 ** min(self.successful_requests, 10))  # 初始成功时可略微降低延迟

        # 应用最小和最大延迟限制
        current_delay = max(self.config['min_delay'], 
                           min(current_delay, self.config['max_delay']))
        
        return current_delay

    def apply_delay(self, is_failure=False):
        """
        应用API延迟
        
        Args:
            is_failure: 是否是失败请求，如果是则增加延迟
        """
        delay = self.calculate_delay(is_failure)
        time.sleep(delay)
        self.total_delay += delay

    def record_request_result(self, is_success):
        """
        记录请求结果以调整延迟策略
        
        Args:
            is_success: 请求是否成功
        """
        if is_success:
            self.successful_requests += 1
        else:
            self.failure_count += 1


def calculate_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """
    预计算技术指标并添加到历史数据中
    
    Args:
        hist: 包含OHLCV数据的DataFrame
        
    Returns:
        添加了技术指标的DataFrame
    """
    if hist is None or hist.empty or 'Close' not in hist.columns:
        return hist
    
    # 复制数据以避免修改原始数据
    result = hist.copy()
    
    try:
        # RSI (14)
        delta = result['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        result['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp12 = result['Close'].ewm(span=12, adjust=False).mean()
        exp26 = result['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Histogram'] = macd - signal
        
        # ATR (14)
        high_low = result['High'] - result['Low']
        high_close = abs(result['High'] - result['Close'].shift())
        low_close = abs(result['Low'] - result['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        
        # 布林带 (20, 2)
        sma20 = result['Close'].rolling(window=20, min_periods=1).mean()
        std20 = result['Close'].rolling(window=20, min_periods=1).std()
        result['BB_Middle'] = sma20
        result['BB_Upper'] = sma20 + (std20 * 2)
        result['BB_Lower'] = sma20 - (std20 * 2)
        
        # 移动平均线
        result['MA_5'] = result['Close'].rolling(window=5, min_periods=1).mean()
        result['MA_10'] = result['Close'].rolling(window=10, min_periods=1).mean()
        result['MA_20'] = result['Close'].rolling(window=20, min_periods=1).mean()
        result['MA_50'] = result['Close'].rolling(window=50, min_periods=1).mean()
        result['MA_200'] = result['Close'].rolling(window=200, min_periods=1).mean()
        
        # 成交量移动平均
        result['Volume_MA_20'] = result['Volume'].rolling(window=20, min_periods=1).mean()
        
        # 价格变化率
        result['Price_Change_Pct'] = result['Close'].pct_change(fill_method=None)
        result['Price_Change_Pct_5D'] = result['Close'].pct_change(periods=5, fill_method=None)
        
    except Exception as e:
        print(f" - [技术指标计算] 计算技术指标时出错: {e}")
        # 如果计算失败，返回原始数据
        return hist
    
    return result


def get_data_with_cache(symbol: str, market: str, fast_mode: bool = False, interval: str = '1d', config=None) -> (pd.DataFrame, dict, dict):
    """
    獲取股票數據，根據模式選擇快速加載或同步更新。

    Args:
        symbol: 股票代碼
        market: 市場代碼 ('US' 或 'HK')
        fast_mode: 是否使用快速模式
        interval: 數據時段類型 ('1d' 日線, '1h' 小時線, '1m' 分鐘線)
        config: 配置对象，如果为None则加载配置
    """
    cache_dir = os.path.join('data_cache', market.upper())
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.replace(":", "_")
    csv_file = os.path.join(cache_dir, f"{safe_symbol}_{interval}.csv")  # 添加 interval 到文件名
    json_file = os.path.join(cache_dir, f"{safe_symbol}.json")

    # 获取配置
    if config is None:
        config = load_config()
    
    api_config = config['api']
    retry_attempts = api_config['retry_attempts']

    ticker = yf.Ticker(symbol)

    if fast_mode:
        try:
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
            # 预计算技术指标
            hist = calculate_technical_indicators(hist, config)
            logger.info(f"快速模式加载 {symbol} 数据成功: {len(hist)} 条记录")
            return hist, info, news
        except FileNotFoundError:
            logger.info(f"快速模式 - 缓存文件未找到: {csv_file} 或 {json_file}")
            return get_data_with_cache(symbol, market, fast_mode=False, interval=interval, config=config)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"快速模式 - JSON 解析失败 {symbol}: {e}")
            # 删除损坏的缓存文件
            try:
                os.remove(json_file)
                logger.info(f"已删除损坏的缓存文件: {json_file}")
            except Exception as rm_e:
                logger.warning(f"删除损坏缓存文件失败: {rm_e}")
            return get_data_with_cache(symbol, market, fast_mode=False, interval=interval, config=config)
        except Exception as e:
            logger.error(f"快速模式 - 加载 {symbol} 数据时发生未知错误: {e}")
            return get_data_with_cache(symbol, market, fast_mode=False, interval=interval, config=config)

    # --- 正常同步模式 ---
    today = datetime.now().date()
    hist, info, news = pd.DataFrame(), {}, []

    # 创建API延迟管理器
    delayer = APIDelayer(config)

    # 首先获取历史价格数据（这个通常比 info 更容易获取）
    if os.path.exists(csv_file):
        try:
            # 自动检测索引列名（Date 或 Datetime）
            hist = _read_csv_with_auto_index(csv_file)
            last_cached_date = hist.index.max().date()

            if last_cached_date >= today:
                logger.info(f"从缓存加载 {symbol} {len(hist)} 条数据")
            else:
                start_date = last_cached_date + timedelta(days=1)
                logger.info(f"缓存数据过旧，正在从 {start_date.strftime('%Y-%m-%d')} 下载增量数据...")
                
                # 应用API延迟
                delayer.apply_delay()
                
                new_hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), interval=interval, auto_adjust=True)
                if not new_hist.empty:
                    hist = pd.concat([hist, new_hist])
                    logger.info(f"下载了 {len(new_hist)} 条新数据")
                else:
                    logger.info("没有新的数据可下载")
        except Exception as e:
            logger.warning(f"加载历史缓存数据失败，将重新下载: {e}")
            hist = pd.DataFrame()  # 重置为新的数据

    if hist.empty or hist.shape[0] == 0:
        logger.info(f"缓存不存在或加载失败，正在下载 {symbol} 全部历史数据...")
        # 根據 interval 設置不同的 period
        if interval == '1m':
            period = '7d'  # 分鐘線只下載最近7天
        elif interval == '1h':
            period = '730d'  # 小時線下載最近2年
        else:
            period = 'max'  # 日線下載全部歷史
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        logger.info(f"下载了 {len(hist)} 条数据")

    # 预计算技术指标
    hist = calculate_technical_indicators(hist, config)
    
    # 优化内存使用 - 转换数据类型
    if not hist.empty:
        hist = optimize_dataframe_memory(hist)
        float_shares = None  # 暂时设置为 None
        hist['FloatShares'] = float_shares
        hist.to_csv(csv_file)

    # 尝试获取 info 数据（如果失败则使用空字典）
    info_loaded = False
    for attempt in range(retry_attempts):
        try:
            logger.debug(f"尝试获取 {symbol} info 数据 (第 {attempt + 1}/{retry_attempts} 次)")
            # 应用API延迟
            delayer.apply_delay()
            
            info = ticker.info
            # 确保 info 不为空 - 使用更安全的方式
            if info is None:
                logger.warning(f"{symbol} info 数据为空")
                info = {}
            elif not isinstance(info, dict):
                # 如果 info 不是字典，转换为字典
                logger.warning(f"{symbol} info 格式异常，转换为字典")
                info = {}
            elif isinstance(info, dict) and len(info) == 0:
                logger.warning(f"{symbol} info 字典为空")
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
                    logger.debug(f"成功保存 {symbol} info 到缓存")
                except Exception as save_error:
                    logger.error(f"保存 {symbol} info 失败: {save_error}")
                    # 保存失败不影响主流程

            # 记录成功请求
            delayer.record_request_result(True)
            info_loaded = True
            break  # 成功获取info，跳出重试循环
        except Exception as e:
            logger.error(f"无法获取 {symbol} info (尝试 {attempt + 1}/{retry_attempts}): {e}")
            # 记录失败请求
            delayer.record_request_result(False)
            if attempt < retry_attempts - 1:
                logger.debug(f"重试前等待: 第 {attempt + 1} 次")
                delayer.apply_delay(is_failure=True)  # 重试前等待更长时间
            else:
                logger.error(f"{symbol} 所有重试均失败，将使用空数据")
                info = {}

    if not info_loaded:
        logger.warning(f"未能获取 {symbol} 的 info 数据，使用默认空数据")

    news = []

    # 最后再检查一次缓存，确保数据完整性
    if os.path.exists(csv_file) and hist.empty:
        try:
            # 自动检测索引列名（Date 或 Datetime）
            hist = _read_csv_with_auto_index(csv_file)
            logger.info(f"从CSV文件重新加载 {symbol} 的历史数据: {len(hist)} 条")
        except Exception as e:
            logger.error(f"重新从CSV加载 {symbol} 数据失败: {e}")

    # 预计算技术指标（如果尚未计算）
    if not hist.empty and 'RSI_14' not in hist.columns:
        hist = calculate_technical_indicators(hist, config)
    
    # 优化内存使用 - 转换数据类型
    if not hist.empty:
        hist = optimize_dataframe_memory(hist)
        float_shares = info.get('floatShares', None)
        hist['FloatShares'] = float_shares
        hist.to_csv(csv_file)

    if info:
        try:
            # 使用递归的 serialize_for_json 处理所有嵌套层级
            processed_info = serialize_for_json(info)

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(processed_info, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存 {symbol} JSON 缓存失败: {e}")

    logger.info(f"成功获取 {symbol} 数据: {len(hist)} 条记录, info字段数: {len(info) if info else 0}")
    return hist, info, news

def run_analysis(market: str, force_fast_mode: bool = False, skip_strategies: bool = False, symbol_filter: str = None, interval: str = '1d', max_workers: int = None, model: str = 'iflow-rome-30ba3b'):
    """
    對指定市場執行所有選股策略分析

    Args:
        market: 市場代碼 ('US' 或 'HK')
        force_fast_mode: 是否強制跳過緩存更新，直接使用快速模式
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
    
    # 实时输出符合条件的股票到主报告文件
    realtime_output_enabled = config['analysis']['enable_realtime_output']
    main_report_file = f"{market.lower()}_stocks_{datetime.now().strftime('%Y-%m-%d')}.txt"
    if realtime_output_enabled:
        print(f"--- 實時輸出已啟用，將記錄到主報告文件: {main_report_file} ---")
    
    # 使用线程池并行处理股票
    def analyze_single_stock(symbol, config, skip_strategies=False, model='iflow-rome-30ba3b'):
        """分析单个股票的函数，接受配置参数
        
        Args:
            symbol: 股票代码
            config: 配置对象
            skip_strategies: 是否跳过策略筛选，所有股票都进行AI分析
            model: 要使用的AI模型名称
        """
        try:
            # 获取股票數據（會自動處理緩存）
            hist, info, news = get_data_with_cache(symbol, market, fast_mode=not is_sync_needed, interval=interval, config=config)
            
            # 数据质量检查
            if hist.empty or len(hist) < 2 or info is None or (isinstance(info, dict) and len(info) == 0):
                return None, 0  # 返回None表示该股票未通过筛选，0表示未分析成功
            
            # 数据预处理优化：基础筛选
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
            
            # 執行所有策略或跳過策略
            passed_strategies = []
            if skip_strategies:
                # 如果跳過策略，則所有股票都標記為通過空策略列表
                passed_strategies = ["跳過策略"]
                print(f"\r{' ' * 80}\r🔍 {symbol} 已跳過策略篩選，直接進行AI分析")
            else:
                # 執行所有策略
                for strategy in strategies_to_run:
                    if strategy.run(hist.copy(), info=info, market_return=market_latest_return, is_market_healthy=is_market_healthy):
                        passed_strategies.append(strategy.name)
            
            # 无论是否跳过策略，只要通过了基础筛选，都需要进行AI分析
            if passed_strategies or skip_strategies:
            
                # 步骤 1: AI 分析
                ai_analysis = None
                
                try:
                    ai_analysis = analyze_stock_with_ai({
                        'symbol': symbol,
                        'strategies': passed_strategies,
                        'info': info,
                        'market': market
                    }, hist, interval, model)
                except Exception as ai_e:
                    print(f" - AI 分析出错: {ai_e}", end='')

                # 步骤 2: 将股票添加到结果中（当启用 skip_strategies 时，所有股票都添加）
                # 如果启用了 skip_strategies，所有通过基础筛选的股票都添加到结果中
                exchange = info.get('exchange', 'UNKNOWN')
                stock_result = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'strategies': passed_strategies,
                    'info': info,
                    'news': news,
                    'ai_analysis': ai_analysis
                }
                
                # 实时输出符合条件的股票到主报告文件
                if realtime_output_enabled:
                    with threading.Lock():
                        with open(main_report_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n--- 實時輸出 ({datetime.now().strftime('%H:%M:%S')}) ---\n")
                            f.write(f"{symbol} 符合策略: {passed_strategies}\n")
                            if ai_analysis:
                                f.write(f"AI 分析: {ai_analysis['summary']}\n")
                            f.write("-" * 50 + "\n")
                
                if skip_strategies:
                    print(f"\r{' ' * 80}\r✅ {symbol} 跳過策略篩選，已進行AI分析")
                else:
                    print(f"\r{' ' * 80}\r✅ {symbol} 符合策略: {passed_strategies}")
                # 仅输出简要AI分析信息，详细内容在最终报告中显示
                if ai_analysis:
                    print(f"   🤖 AI 分析: 已完成 (模型: {ai_analysis['model_used']})")
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
        # 提交所有任务，传递配置参数、skip_strategies参数和model参数
        future_to_symbol = {executor.submit(analyze_single_stock, symbol, config, skip_strategies, model): symbol for symbol in tickers}
        
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