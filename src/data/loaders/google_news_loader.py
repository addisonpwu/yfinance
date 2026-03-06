"""
Google News RSS 新闻获取模块

通过 Google News RSS 获取股票相关新闻

使用方法：
    from src.data.loaders.google_news_loader import GoogleNewsRepository
    
    repo = GoogleNewsRepository()
    news = repo.get_news("0700", "HK", days_back=7, max_items=10)
"""
import feedparser
import ssl
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import urllib.request
from urllib.parse import quote_plus, urlencode

from src.config.settings import config_manager
from src.utils.logger import get_data_logger

# 安全修复：使用 certifi 提供证书
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass


class GoogleNewsRepository:
    """Google News RSS 新闻获取器"""
    
    def __init__(self):
        self.config = config_manager.get_config()
        self.logger = get_data_logger()
        
        # 从配置获取 Google News 默认参数
        news_config = getattr(self.config, 'news', None)
        google_config = getattr(news_config, 'google', None) if news_config else None
        
        # 默认语言和地区设置
        self.default_language = getattr(google_config, 'default_language', 'zh-TW') if google_config else 'zh-TW'
        self.default_region = getattr(google_config, 'default_region', 'TW') if google_config else 'TW'
        
        # 新闻配置
        self.default_days_back = getattr(news_config, 'days_back', 14) if news_config else 14
        self.default_max_items = getattr(news_config, 'max_news_items', 10) if news_config else 10
    
    def _build_search_query(self, symbol: str, market: str, days_back: int) -> str:
        """
        构建 Google News RSS 搜索查询
        
        Args:
            symbol: 股票代码
            market: 市场代码 (HK/US)
            days_back: 回溯天数
            
        Returns:
            编码后的搜索查询字符串
        """
        symbol = symbol.upper().replace(".HK", "")
        
        # 根据市场选择搜索关键词
        if market.upper() == "HK":
            # 港股：搜索 股票代码 + 股票名 + stock
            query = f"({symbol} OR {symbol} HK) 股票"
        else:
            # 美股：搜索 股票代码 + stock
            query = f"({symbol} OR {symbol} stock)"
        
        # 添加时间过滤
        query += f" when:{days_back}d"
        
        return query
    
    def _get_lang_and_region(self, language: str = None, region: str = None) -> tuple:
        """
        获取语言和地区代码
        
        Args:
            language: 语言代码 (如 'zh-TW', 'en-US')
            region: 地区代码 (如 'TW', 'US')
            
        Returns:
            (lang, geo) 元组
        """
        lang = language or self.default_language
        geo = region or self.default_region
        
        # 映射到 Google News RSS 使用的代码
        lang_map = {
            'zh-TW': 'zh-Hant',
            'zh-CN': 'zh-CN',
            'en-US': 'en-US',
            'en-GB': 'en-GB',
            'ja-JP': 'ja',
            'ko-KR': 'ko'
        }
        
        geo_map = {
            'TW': 'TW',
            'HK': 'HK',
            'US': 'US',
            'CN': 'CN',
            'JP': 'JP',
            'KR': 'KR',
            'GB': 'GB'
        }
        
        return lang_map.get(lang, 'zh-Hant'), geo_map.get(geo, 'TW')
    
    def _parse_rss_entry(self, entry, cutoff: datetime) -> Optional[Dict]:
        """
        解析单个 RSS 条目
        
        Args:
            entry: RSS 条目
            cutoff: 时间截止点
            
        Returns:
            解析后的新闻字典，或 None 如果已过期
        """
        try:
            # 解析发布时间
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_time = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            else:
                # 尝试解析日期字符串
                pub_time = None
            
            # 过滤过期新闻
            if pub_time and pub_time < cutoff:
                return None
            
            # 解析摘要/描述
            summary = ""
            if hasattr(entry, 'summary'):
                summary = entry.summary
            elif hasattr(entry, 'description'):
                summary = entry.description
            
            # 清理 HTML 标签
            import re
            summary = re.sub(r'<[^>]+>', '', summary)
            summary = summary.strip()[:200]  # 截取摘要
            
            # 获取发布时间字符串
            if pub_time:
                published_str = pub_time.strftime("%Y-%m-%d %H:%M")
            else:
                published_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            return {
                "title": entry.get("title", "").strip(),
                "link": entry.get("link", ""),
                "published": published_str,
                "summary": summary,
                "publisher": entry.get("source", {}).get("title", "Google News") if hasattr(entry, 'source') else "Google News",
                "source": "google_rss"
            }
            
        except Exception as e:
            self.logger.debug(f"解析 RSS 条目失败: {e}")
            return None
    
    def get_news(
        self,
        symbol: str,
        market: str = "HK",
        days_back: int = None,
        max_items: int = None,
        language: str = None,
        region: str = None
    ) -> List[Dict]:
        """
        从 Google News RSS 获取股票相关新闻
        
        Args:
            symbol: 股票代码 (如 "0700" 或 "AAPL")
            market: 市场代码 ("HK" 或 "US")
            days_back: 回溯天数，默认从配置读取
            max_items: 最大新闻数量，默认从配置读取
            language: 语言代码，默认从配置读取
            region: 地区代码，默认从配置读取
            
        Returns:
            List[Dict]: 新闻列表，每条包含 title, link, published, summary, publisher, source
        """
        # 使用默认值
        days_back = days_back if days_back is not None else self.default_days_back
        max_items = max_items if max_items is not None else self.default_max_items
        
        # 获取语言和地区
        lang, geo = self._get_lang_and_region(language, region)
        
        # 构建搜索查询
        query = self._build_search_query(symbol, market, days_back)
        encoded_query = quote_plus(query)
        
        # 构建 RSS URL
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang}&gl={geo}"
        
        self.logger.info(f"Google News RSS URL: {rss_url}")
        
        try:
            # 解析 RSS
            feed = feedparser.parse(rss_url)
            
            if feed.bozo:
                self.logger.warning(f"Google News RSS 解析警告 ({symbol}): {feed.bozo_exception}")
            
            # 检查是否有条目
            if not feed.entries:
                self.logger.info(f"Google News 未找到 {symbol} 的相关新闻")
                return []
            
            # 时间过滤截止点
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            news_list = []
            for entry in feed.entries[:max_items * 3]:  # 多获取一些，过滤后截取
                parsed = self._parse_rss_entry(entry, cutoff)
                if parsed:
                    news_list.append(parsed)
                    
                if len(news_list) >= max_items:
                    break
            
            # 按时间降序排序
            news_list.sort(key=lambda x: x["published"], reverse=True)
            
            self.logger.info(f"Google News 获取 {symbol} 新闻 {len(news_list)} 条")
            
            return news_list
            
        except Exception as e:
            self.logger.error(f"获取 {symbol} Google News 失败: {e}")
            return []
    
    def get_news_multi_source(
        self,
        symbol: str,
        market: str = "HK",
        days_back: int = None,
        max_items: int = None,
        sources: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        从多个来源获取新闻
        
        Args:
            symbol: 股票代码
            market: 市场代码
            days_back: 回溯天数
            max_items: 最大新闻数量
            sources: 来源列表 ["google", "yahoo"] 或 ["google"]
            
        Returns:
            Dict: {"google": [...], "yahoo": [...], "all": [...]} 
            all 是合并去重后的结果
        """
        from src.data.loaders.yahoo_loader import YahooFinanceRepository
        
        if sources is None:
            sources = ["google", "yahoo"]
        
        results = {}
        all_news = []
        
        if "google" in sources:
            google_news = self.get_news(symbol, market, days_back, max_items)
            results["google"] = google_news
            all_news.extend(google_news)
        
        if "yahoo" in sources:
            yahoo_repo = YahooFinanceRepository()
            yahoo_news = yahoo_repo.get_news(symbol, market, days_back, max_items)
            results["yahoo"] = yahoo_news
            all_news.extend(yahoo_news)
        
        # 按时间降序排序并去重（根据 title）
        seen_titles = set()
        unique_news = []
        for news in sorted(all_news, key=lambda x: x["published"], reverse=True):
            title_key = news["title"][:50]  # 使用前50字符作为key
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)
        
        results["all"] = unique_news[:max_items or self.default_max_items]
        
        return results


# 便捷函数
def get_google_news(symbol: str, market: str = "HK", days_back: int = 7, max_items: int = 10) -> List[Dict]:
    """
    便捷函数：获取 Google News 股票新闻
    
    Args:
        symbol: 股票代码
        market: 市场代码
        days_back: 回溯天数
        max_items: 最大新闻数量
        
    Returns:
        List[Dict]: 新闻列表
    """
    repo = GoogleNewsRepository()
    return repo.get_news(symbol, market, days_back, max_items)


if __name__ == "__main__":
    # 测试代码
    print("测试 Google News RSS 获取...")
    
    # 测试港股
    news_hk = get_google_news("0700", "HK", days_back=7, max_items=5)
    print(f"\n港股 0700 (腾讯) 新闻:")
    for i, n in enumerate(news_hk, 1):
        print(f"  {i}. {n['title'][:50]}...")
        print(f"     时间: {n['published']}")
    
    # 测试美股
    news_us = get_google_news("AAPL", "US", days_back=7, max_items=5)
    print(f"\n美股 AAPL (Apple) 新闻:")
    for i, n in enumerate(news_us, 1):
        print(f"  {i}. {n['title'][:50]}...")
        print(f"     时间: {n['published']}")
    
    print("\n测试完成!")
