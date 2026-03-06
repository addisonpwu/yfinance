"""
新闻服务统一接口

提供统一的新闻获取接口，支持 Yahoo/Google/both 多种来源

使用方法：
    from src.data.loaders.news_service import NewsService, get_news_service
    
    # 获取服务实例
    service = get_news_service()
    
    # 获取新闻
    news = service.get_news("0700", "HK")
    
    # 或者使用便捷函数
    from src.data.loaders.news_service import get_stock_news
    news = get_stock_news("0700", "HK")
"""
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.config.settings import config_manager
from src.utils.logger import get_data_logger


@dataclass
class NewsResult:
    """新闻获取结果"""
    news: List[Dict]
    provider: str  # "google" | "yahoo" | "both"
    sources_used: List[str]  # 实际使用的来源


class NewsService:
    """
    统一新闻服务
    
    根据配置选择不同的新闻来源：
    - "google": 仅使用 Google News RSS
    - "yahoo": 仅使用 Yahoo Finance RSS
    - "both": 优先使用 Google，失败时回退到 Yahoo
    """
    
    def __init__(self, provider: str = None):
        """
        初始化新闻服务
        
        Args:
            provider: 新闻来源 ("google" | "yahoo" | "both")，默认从配置读取
        """
        self.config = config_manager.get_config()
        self.logger = get_data_logger()
        
        # 从配置获取新闻设置
        news_config = getattr(self.config, 'news', None)
        self.provider = provider or getattr(news_config, 'provider', 'both')
        
        # 初始化各个新闻源
        self._init_providers()
        
        self.logger.info(f"NewsService 初始化完成，提供商: {self.provider}")
    
    def _init_providers(self):
        """初始化新闻提供者"""
        # 延迟导入，避免循环依赖
        from src.data.loaders.yahoo_loader import YahooFinanceRepository
        from src.data.loaders.google_news_loader import GoogleNewsRepository
        
        self._yahoo_repo = YahooFinanceRepository()
        self._google_repo = GoogleNewsRepository()
    
    def get_news(
        self,
        symbol: str,
        market: str = "HK",
        days_back: int = None,
        max_items: int = None
    ) -> List[Dict]:
        """
        获取股票新闻
        
        Args:
            symbol: 股票代码
            market: 市场代码 (HK/US)
            days_back: 回溯天数，默认从配置读取
            max_items: 最大新闻数量，默认从配置读取
            
        Returns:
            List[Dict]: 新闻列表
        """
        # 使用默认值
        news_config = getattr(self.config, 'news', None)
        if days_back is None:
            days_back = getattr(news_config, 'days_back', 14) if news_config else 14
        if max_items is None:
            max_items = getattr(news_config, 'max_news_items', 10) if news_config else 10
        
        provider = self.provider.lower()
        
        if provider == "google":
            return self._get_google_news(symbol, market, days_back, max_items)
        elif provider == "yahoo":
            return self._get_yahoo_news(symbol, market, days_back, max_items)
        elif provider == "both":
            return self._get_news_with_fallback(symbol, market, days_back, max_items)
        else:
            self.logger.warning(f"未知的新闻提供商: {provider}，使用 'both'")
            return self._get_news_with_fallback(symbol, market, days_back, max_items)
    
    def _get_google_news(
        self,
        symbol: str,
        market: str,
        days_back: int,
        max_items: int
    ) -> List[Dict]:
        """从 Google News 获取新闻"""
        try:
            news = self._google_repo.get_news(symbol, market, days_back, max_items)
            self.logger.info(f"[NewsService] Google News: {symbol} 获取 {len(news)} 条")
            return news
        except Exception as e:
            self.logger.error(f"[NewsService] Google News 获取失败: {e}")
            return []
    
    def _get_yahoo_news(
        self,
        symbol: str,
        market: str,
        days_back: int,
        max_items: int
    ) -> List[Dict]:
        """从 Yahoo Finance 获取新闻"""
        try:
            news = self._yahoo_repo.get_news(symbol, market, days_back, max_items)
            self.logger.info(f"[NewsService] Yahoo News: {symbol} 获取 {len(news)} 条")
            return news
        except Exception as e:
            self.logger.error(f"[NewsService] Yahoo News 获取失败: {e}")
            return []
    
    def _get_news_with_fallback(
        self,
        symbol: str,
        market: str,
        days_back: int,
        max_items: int
    ) -> List[Dict]:
        """
        优先使用 Google News，失败时回退到 Yahoo
        
        这是默认行为，兼顾覆盖率和稳定性
        """
        # 首先尝试 Google News
        google_news = self._get_google_news(symbol, market, days_back, max_items)
        
        if google_news:
            return google_news
        
        # Google 失败，回退到 Yahoo
        self.logger.warning(f"[NewsService] Google News 无数据，回退到 Yahoo: {symbol}")
        yahoo_news = self._get_yahoo_news(symbol, market, days_back, max_items)
        
        return yahoo_news
    
    def get_news_with_details(
        self,
        symbol: str,
        market: str = "HK",
        days_back: int = None,
        max_items: int = None
    ) -> NewsResult:
        """
        获取新闻并返回详细信息
        
        Args:
            symbol: 股票代码
            market: 市场代码
            days_back: 回溯天数
            max_items: 最大新闻数量
            
        Returns:
            NewsResult: 包含新闻、使用的提供商、来源列表
        """
        provider = self.provider.lower()
        
        if provider == "both":
            # 对于 "both" 模式，先尝试 Google
            google_news = self._get_google_news(symbol, market, days_back or 14, max_items or 10)
            if google_news:
                return NewsResult(
                    news=google_news,
                    provider="google",
                    sources_used=["google"]
                )
            # 回退到 Yahoo
            yahoo_news = self._get_yahoo_news(symbol, market, days_back or 14, max_items or 10)
            return NewsResult(
                news=yahoo_news,
                provider="yahoo",
                sources_used=["yahoo"]
            )
        else:
            news = self.get_news(symbol, market, days_back, max_items)
            return NewsResult(
                news=news,
                provider=provider,
                sources_used=[provider]
            )
    
    def get_multi_source_news(
        self,
        symbol: str,
        market: str = "HK",
        days_back: int = None,
        max_items: int = None
    ) -> Dict[str, List[Dict]]:
        """
        从多个来源获取新闻并合并
        
        Args:
            symbol: 股票代码
            market: 市场代码
            days_back: 回溯天数
            max_items: 最大新闻数量
            
        Returns:
            Dict: {
                "google": [...],
                "yahoo": [...],
                "all": [...]  # 合并去重后的结果
            }
        """
        try:
            results = self._google_repo.get_news_multi_source(
                symbol, market, days_back, max_items, sources=["google", "yahoo"]
            )
            return results
        except Exception as e:
            self.logger.error(f"[NewsService] 多来源获取失败: {e}")
            return {"google": [], "yahoo": [], "all": []}


# 全局服务实例
_news_service: Optional[NewsService] = None


def get_news_service(provider: str = None) -> NewsService:
    """
    获取 NewsService 全局实例
    
    Args:
        provider: 可选的提供商覆盖
        
    Returns:
        NewsService 实例
    """
    global _news_service
    if _news_service is None:
        _news_service = NewsService(provider)
    return _news_service


def reset_news_service():
    """重置全局 NewsService 实例"""
    global _news_service
    _news_service = None


def get_stock_news(
    symbol: str,
    market: str = "HK",
    days_back: int = None,
    max_items: int = None,
    provider: str = None
) -> List[Dict]:
    """
    便捷函数：获取股票新闻
    
    Args:
        symbol: 股票代码
        market: 市场代码
        days_back: 回溯天数
        max_items: 最大新闻数量
        provider: 新闻来源 ("google" | "yahoo" | "both")
        
    Returns:
        List[Dict]: 新闻列表
    """
    service = get_news_service(provider)
    return service.get_news(symbol, market, days_back, max_items)


# 兼容旧接口：直接导入 YahooFinanceRepository 的 get_news
def get_yahoo_news(
    symbol: str,
    market: str = "HK",
    days_back: int = None,
    max_items: int = None
) -> List[Dict]:
    """
    兼容函数：仅使用 Yahoo Finance 获取新闻
    
    Args:
        symbol: 股票代码
        market: 市场代码
        days_back: 回溯天数
        max_items: 最大新闻数量
        
    Returns:
        List[Dict]: 新闻列表
    """
    service = NewsService(provider="yahoo")
    return service.get_news(symbol, market, days_back, max_items)


def get_google_news(
    symbol: str,
    market: str = "HK",
    days_back: int = None,
    max_items: int = None
) -> List[Dict]:
    """
    兼容函数：仅使用 Google News 获取新闻
    
    Args:
        symbol: 股票代码
        market: 市场代码
        days_back: 回溯天数
        max_items: 最大新闻数量
        
    Returns:
        List[Dict]: 新闻列表
    """
    service = NewsService(provider="google")
    return service.get_news(symbol, market, days_back, max_items)


if __name__ == "__main__":
    # 测试代码
    print("测试 NewsService...")
    
    # 测试默认配置
    service = get_news_service()
    print(f"当前提供商: {service.provider}")
    
    # 测试港股
    print("\n=== 测试港股 0700 ===")
    news = service.get_news("0700", "HK", days_back=7, max_items=5)
    print(f"获取新闻 {len(news)} 条:")
    for i, n in enumerate(news, 1):
        print(f"  {i}. {n['title'][:50]}...")
        print(f"     来源: {n.get('source', 'unknown')}")
    
    # 测试美股
    print("\n=== 测试美股 AAPL ===")
    news = service.get_news("AAPL", "US", days_back=7, max_items=5)
    print(f"获取新闻 {len(news)} 条:")
    for i, n in enumerate(news, 1):
        print(f"  {i}. {n['title'][:50]}...")
        print(f"     来源: {n.get('source', 'unknown')}")
    
    # 测试指定提供商
    print("\n=== 测试指定 Yahoo ===")
    yahoo_service = NewsService(provider="yahoo")
    news = yahoo_service.get_news("0700", "HK", days_back=7, max_items=3)
    print(f"Yahoo 新闻 {len(news)} 条")
    
    print("\n=== 测试指定 Google ===")
    google_service = NewsService(provider="google")
    news = google_service.get_news("AAPL", "US", days_back=7, max_items=3)
    print(f"Google 新闻 {len(news)} 条")
    
    print("\n测试完成!")
