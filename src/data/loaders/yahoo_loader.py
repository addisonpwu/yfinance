import yfinance as yf
import pandas as pd
import json
from typing import Dict, List, Optional
import time
from datetime import datetime, timedelta
import os
import numpy as np
from pathlib import Path
from src.data.external.stock_repository import StockRepository
from src.data.cache.cache_service import OptimizedCache
from src.config.settings import config_manager
from src.utils.exceptions import DataFetchException, CacheException
from src.utils.logger import get_data_logger

class YahooFinanceRepository(StockRepository):
    def __init__(self):
        # 使用配置中的 enable_cache 设置来初始化缓存服务
        config = config_manager.get_config()
        self.cache_service = OptimizedCache(enabled=config.data.enable_cache)
        self.config = config_manager.get_config()
        self.logger = get_data_logger()
        # 总是创建缓存目录，以便保存CSV文件
        self._setup_cache_dirs()
        
    def _setup_cache_dirs(self):
        """设置缓存目录结构"""
        os.makedirs(f"data_cache/{'US'}", exist_ok=True)
        os.makedirs(f"data_cache/{'HK'}", exist_ok=True)
        os.makedirs("data_cache/ai_analysis", exist_ok=True)
    
    def _save_historical_data_as_csv(self, symbol: str, data: pd.DataFrame, market: str, interval: str = '1d') -> None:
        """保存历史数据为CSV文件（总是保存，不依赖于缓存设置）"""
        try:
            cache_dir = os.path.join('data_cache', market.upper())
            safe_symbol = symbol.replace(":", "_")
            csv_file = os.path.join(cache_dir, f"{safe_symbol}_{interval}.csv")
            
            data.to_csv(csv_file)
        except Exception as e:
            self.logger.error(f"保存 {symbol} 历史数据为CSV失败: {e}")
            raise CacheException(f"保存历史数据为CSV失败: {e}")
    
    def get_historical_data(self, symbol: str, market: str, interval: str = '1d') -> pd.DataFrame:
        try:
            cache_key = f"{symbol}_{interval}_{market}"
            cached_data = self.cache_service.get(cache_key)
            
            if cached_data is not None:
                self.logger.info(f"从缓存获取 {symbol} ({interval}) 的历史数据")
                return cached_data
            
            # 获取数据
            hist = self._fetch_historical_data(symbol, market, interval)
            
            # 保存为CSV文件（总是保存，不依赖于缓存设置）
            self._save_historical_data_as_csv(symbol, hist, market, interval)
            
            # 根据缓存设置决定是否缓存数据
            if self.cache_service.enabled:
                self.cache_service.set(cache_key, hist)
            
            return hist
        except Exception as e:
            self.logger.error(f"获取 {symbol} 历史数据失败: {e}")
            raise DataFetchException(f"获取历史数据失败: {e}")
    
    def get_financial_info(self, symbol: str) -> Dict:
        try:
            cache_key = f"info_{symbol}"
            cached_info = self.cache_service.get(cache_key)
            
            if cached_info is not None:
                self.logger.info(f"从缓存获取 {symbol} 的财务信息")
                return cached_info
            
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            
            # 确保 info 不为空 - 使用更安全的方式
            if info is None:
                self.logger.warning(f"{symbol} info 数据为空")
                info = {}
            elif not isinstance(info, dict):
                # 如果 info 不是字典，转换为字典
                self.logger.warning(f"{symbol} info 格式异常，转换为字典")
                info = {}
            elif isinstance(info, dict) and len(info) == 0:
                self.logger.warning(f"{symbol} info 字典为空")
            
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
            
            # 缓存信息
            self.cache_service.set(cache_key, info)
            
            return info
        except Exception as e:
            self.logger.error(f"获取 {symbol} 财务信息失败: {e}")
            raise DataFetchException(f"获取财务信息失败: {e}")
    
    def get_news(self, symbol: str) -> List:
        # 暂时返回空列表，可根据需要实现新闻获取逻辑
        return []
    
    def save_historical_data(self, symbol: str, data: pd.DataFrame, market: str, interval: str = '1d') -> None:
        try:
            # 保存为CSV文件（总是保存，不依赖于缓存设置）
            self._save_historical_data_as_csv(symbol, data, market, interval)
            
            # 根据缓存设置决定是否缓存数据
            if self.cache_service.enabled:
                cache_key = f"{symbol}_{interval}_{market}"
                self.cache_service.set(cache_key, data)
        except Exception as e:
            self.logger.error(f"保存 {symbol} 历史数据失败: {e}")
            raise CacheException(f"保存历史数据失败: {e}")
    
    def save_financial_info(self, symbol: str, info: Dict) -> None:
        try:
            cache_key = f"info_{symbol}"
            self.cache_service.set(cache_key, info)
            
            # 同时保存到JSON文件
            cache_dir = os.path.join('data_cache', 'US')  # 默认保存到US目录
            json_file = os.path.join(cache_dir, f"{symbol.replace(':', '_')}.json")
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.error(f"保存 {symbol} 财务信息失败: {e}")
            raise CacheException(f"保存财务信息失败: {e}")
    
    def _fetch_historical_data(self, symbol: str, market: str, interval: str = '1d') -> pd.DataFrame:
        """获取历史数据的内部方法"""
        ticker = yf.Ticker(symbol)
        
        # 根据 interval 设置不同的 period，从配置中获取
        config = self.config
        data_period_config = config.data.data_download_period
        if interval == '1m':
            period = data_period_config.m1  # 分鐘線從配置獲取
        elif interval == '1h':
            period = data_period_config.h1  # 小時線從配置獲取
        else:
            period = data_period_config.d1  # 日線從配置獲取
        
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        
        # 应用API延迟
        time.sleep(self.config.api.base_delay)
        
        return hist

def calculate_technical_indicators(hist: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    预计算技术指标并添加到历史数据中
    
    Args:
        hist: 包含OHLCV数据的DataFrame
        config: 配置对象，如果为None则使用默认配置
        
    Returns:
        添加了技术指标的DataFrame
    """
    if hist is None or hist.empty or 'Close' not in hist.columns:
        return hist
    
    # 复制数据以避免修改原始数据
    result = hist.copy()
    
    # 获取配置
    if config is None:
        from src.config.settings import config_manager
        config = config_manager.get_config()
    
    # 从配置获取技术指标参数
    ti_config = config.technical_indicators
    
    try:
        # RSI (可配置周期)
        delta = result['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=ti_config.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=ti_config.rsi_period, min_periods=1).mean()
        rs = gain / loss
        result[f'RSI_{ti_config.rsi_period}'] = 100 - (100 / (1 + rs))
        
        # MACD (可配置参数)
        exp_fast = result['Close'].ewm(span=ti_config.macd_fast, adjust=False).mean()
        exp_slow = result['Close'].ewm(span=ti_config.macd_slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        signal = macd.ewm(span=ti_config.macd_signal, adjust=False).mean()
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Histogram'] = macd - signal
        
        # ATR (可配置周期)
        high_low = result['High'] - result['Low']
        high_close = abs(result['High'] - result['Close'].shift())
        low_close = abs(result['Low'] - result['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result[f'ATR_{ti_config.atr_period}'] = tr.rolling(window=ti_config.atr_period, min_periods=1).mean()
        
        # 布林带 (可配置参数)
        sma_bb = result['Close'].rolling(window=ti_config.bb_period, min_periods=1).mean()
        std_bb = result['Close'].rolling(window=ti_config.bb_period, min_periods=1).std()
        result['BB_Middle'] = sma_bb
        result['BB_Upper'] = sma_bb + (std_bb * ti_config.bb_std_dev)
        result['BB_Lower'] = sma_bb - (std_bb * ti_config.bb_std_dev)
        
        # 移动平均线 (可配置周期)
        for period in ti_config.ma_periods:
            result[f'MA_{period}'] = result['Close'].rolling(window=period, min_periods=1).mean()
        
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