"""
市场数据服务

负责股票列表获取、大盘数据分析、市场健康状态判断
"""
from typing import List, Dict, Optional, Tuple
import yfinance as yf
import pandas as pd
from dataclasses import dataclass


@dataclass
class MarketData:
    """市场数据"""
    ticker: str
    latest_return: float
    is_healthy: bool
    trend: str  # "多頭" 或 "空頭"


class MarketDataService:
    """市场数据服务"""
    
    # 市场配置
    MARKET_CONFIG = {
        'US': {
            'ticker': '^GSPC',
            'name': 'S&P 500'
        },
        'HK': {
            'ticker': '^HSI',
            'name': '恒生指数'
        }
    }
    
    def __init__(self):
        pass
    
    def get_stock_list(self, market: str, symbol_filter: str = None) -> List[str]:
        """
        获取股票列表
        
        Args:
            market: 市场代码 ('US' 或 'HK')
            symbol_filter: 指定单一股票代码
        
        Returns:
            股票代码列表
        """
        if symbol_filter:
            return [symbol_filter]
        
        market = market.upper()
        if market == 'US':
            from src.data.loaders import us_loader
            return us_loader.get_us_tickers()
        elif market == 'HK':
            from src.data.loaders import hk_loader
            return hk_loader.get_hk_tickers()
        else:
            raise ValueError(f"不支持的市场: {market}")
    
    def get_market_data(self, market: str) -> Optional[MarketData]:
        """
        获取大盘数据
        
        Args:
            market: 市场代码 ('US' 或 'HK')
        
        Returns:
            MarketData 对象，如果获取失败返回 None
        """
        market = market.upper()
        config = self.MARKET_CONFIG.get(market)
        
        if not config:
            print(f"错误: 不支持的市场 '{market}'。请使用 'US' 或 'HK'。")
            return None
        
        ticker = config['ticker']
        
        try:
            market_hist = yf.Ticker(ticker).history(period='1y', auto_adjust=True)
            
            if market_hist.empty or len(market_hist) < 200:
                print(f"大盘历史数据不足以计算200MA")
                return None
            
            # 计算最新回报率
            latest_return = market_hist['Close'].pct_change(fill_method=None).iloc[-1] * 100
            
            # 计算200日均线
            market_hist['MA200'] = market_hist['Close'].rolling(window=200).mean()
            latest_data = market_hist.iloc[-1]
            
            is_healthy = latest_data['Close'] > latest_data['MA200']
            trend = "多頭" if is_healthy else "空頭"
            
            print(f"已成功获取大盘({ticker})数据。今日涨跌: {latest_return:.2f}%。市场趋势: {trend}")
            
            return MarketData(
                ticker=ticker,
                latest_return=latest_return,
                is_healthy=is_healthy,
                trend=trend
            )
            
        except Exception as e:
            print(f"无法下载或分析大盘数据 ({ticker})，策略中的大盘滤网将不会启用。错误: {e}")
            return None
    
    def get_market_ticker(self, market: str) -> str:
        """获取市场对应的指数代码"""
        market = market.upper()
        config = self.MARKET_CONFIG.get(market)
        return config['ticker'] if config else None
