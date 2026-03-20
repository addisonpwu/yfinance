"""
增強版股票列表 JSON 加載器

從 JSON 文件加載股票列表和新聞數據，並過濾掉盈警和負面消息
"""
import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StockNewsItem:
    """股票新聞項"""
    agency: str
    rating: str
    profit: str
    title: str
    publish_time: str
    url: str
    news_type: str  # "rating" | "profit"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agency': self.agency,
            'rating': self.rating,
            'profit': self.profit,
            'title': self.title,
            'publishTime': self.publish_time,
            'url': self.url,
            'type': self.news_type
        }


@dataclass
class StockListItem:
    """股票列表項"""
    stock_code: str
    stock_name: str
    news: List[StockNewsItem]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stockCode': self.stock_code,
            'stockName': self.stock_name,
            'news': [n.to_dict() for n in self.news]
        }


class EnhancedStockListLoader:
    """增強版股票列表 JSON 加載器 - 過濾盈警和負面消息"""
    
    def __init__(self, json_path: str = None):
        """
        初始化加載器
        
        Args:
            json_path: JSON 文件路徑
        """
        self.json_path = json_path
        
        # 負面消息關鍵詞列表
        self.negative_keywords = [
            # 盈警相關
            '盈警', '虧損', '轉虧', '減少', '跌', '下降', '下滑', '倒退',
            '縮水', '萎縮', '惡化', '衰退', '裁員', '減薪', '倒閉', '破產',
            '訴訟', '調查', '罰款', '處罰', '警告', '降級', '下調',
            '看淡', '看空', '看跌', '謹慎', '保守', '風險'
        ]
        
        # 正面消息關鍵詞列表
        self.positive_keywords = [
            '盈喜', '盈利', '增長', '增加', '上升', '上漲', '創新高',
            '突破', '擴大', '改善', '復甦', '回暖', '樂觀', '看好',
            '升級', '上調', '買入', '增持', '強烈買入', '推薦'
        ]
    
    def _is_negative_news(self, news_item: StockNewsItem) -> bool:
        """
        判斷是否為負面新聞
        
        Args:
            news_item: 新聞項
            
        Returns:
            是否為負面新聞
        """
        title = news_item.title
        profit = news_item.profit
        rating = news_item.rating
        
        # 檢查標題中的負面關鍵詞
        title_lower = title.lower()
        for keyword in self.negative_keywords:
            if keyword in title_lower:
                return True
        
        # 檢查業績描述中的負面信息
        if profit:
            profit_lower = profit.lower()
            for keyword in self.negative_keywords:
                if keyword in profit_lower:
                    return True
        
        # 檢查評級中的負面信息
        if rating:
            rating_lower = rating.lower()
            negative_ratings = ['減持', '賣出', '沽售', '跑輸大市', '弱於大市', '落後大市']
            for negative_rating in negative_ratings:
                if negative_rating in rating_lower:
                    return True
        
        return False
    
    def _is_positive_news(self, news_item: StockNewsItem) -> bool:
        """
        判斷是否為正面新聞
        
        Args:
            news_item: 新聞項
            
        Returns:
            是否為正面新聞
        """
        title = news_item.title
        profit = news_item.profit
        rating = news_item.rating
        
        # 檢查標題中的正面關鍵詞
        title_lower = title.lower()
        for keyword in self.positive_keywords:
            if keyword in title_lower:
                return True
        
        # 檢查業績描述中的正面信息
        if profit:
            profit_lower = profit.lower()
            for keyword in self.positive_keywords:
                if keyword in profit_lower:
                    return True
        
        # 檢查評級中的正面信息
        if rating:
            rating_lower = rating.lower()
            positive_ratings = ['買入', '增持', '強烈買入', '跑贏大市', '優於大市', '領先大市']
            for positive_rating in positive_ratings:
                if positive_rating in rating_lower:
                    return True
        
        return False
    
    def load(self, json_path: str = None, filter_negative_news: bool = True) -> List[StockListItem]:
        """
        加載股票列表
        
        Args:
            json_path: JSON 文件路徑（可選，覆蓋初始化時設置的路徑）
            filter_negative_news: 是否過濾負面新聞
        
        Returns:
            股票列表
        """
        path = json_path or self.json_path
        if not path:
            raise ValueError("未指定 JSON 文件路徑")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON 文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._parse_data(data, filter_negative_news)
    
    def _parse_data(self, data: List[Dict[str, Any]], filter_negative_news: bool = True) -> List[StockListItem]:
        """解析 JSON 數據"""
        items = []
        
        for item in data:
            stock_code = item.get('stockCode', '')
            stock_name = item.get('stockName', '')
            news_list = item.get('news', [])
            
            # 解析新聞列表
            news_items = []
            for news in news_list:
                news_item = StockNewsItem(
                    agency=news.get('agency', ''),
                    rating=news.get('rating', ''),
                    profit=news.get('profit', ''),
                    title=news.get('title', ''),
                    publish_time=news.get('publishTime', ''),
                    url=news.get('url', ''),
                    news_type=news.get('type', '')
                )
                
                # 過濾負面新聞
                if filter_negative_news and self._is_negative_news(news_item):
                    continue
                
                news_items.append(news_item)
            
            items.append(StockListItem(
                stock_code=stock_code,
                stock_name=stock_name,
                news=news_items
            ))
        
        return items
    
    def get_stock_codes(self, json_path: str = None, filter_negative_news: bool = True) -> List[str]:
        """
        獲取股票代碼列表
        
        Args:
            json_path: JSON 文件路徑
            filter_negative_news: 是否過濾負面新聞
        
        Returns:
            股票代碼列表
        """
        items = self.load(json_path, filter_negative_news)
        return [item.stock_code for item in items]
    
    def get_stock_with_news(self, json_path: str = None, filter_negative_news: bool = True) -> Dict[str, StockListItem]:
        """
        獲取股票代碼到股票信息的映射
        
        Args:
            json_path: JSON 文件路徑
            filter_negative_news: 是否過濾負面新聞
        
        Returns:
            {stock_code: StockListItem} 映射
        """
        items = self.load(json_path, filter_negative_news)
        return {item.stock_code: item for item in items}
    
    def convert_news_to_standard_format(self, news_items: List[StockNewsItem]) -> List[Dict[str, Any]]:
        """
        將新聞轉換為系統標準格式
        
        系統標準格式:
        {
            'title': str,
            'link': str,
            'published': str,
            'summary': str,
            'publisher': str,
            'source': str
        }
        
        Args:
            news_items: 新聞列表
        
        Returns:
            標準格式的新聞列表
        """
        standard_news = []
        
        for item in news_items:
            # 構建摘要
            summary_parts = []
            if item.rating:
                summary_parts.append(f"評級: {item.rating}")
            if item.profit:
                summary_parts.append(f"業績: {item.profit}")
            if item.agency:
                summary_parts.append(f"機構: {item.agency}")
            
            summary = " | ".join(summary_parts) if summary_parts else item.title
            
            standard_news.append({
                'title': item.title,
                'link': item.url,
                'published': item.publish_time,
                'summary': summary,
                'publisher': item.agency or '未知',
                'source': 'json_import',
                'type': item.news_type,
                'rating': item.rating,
                'profit': item.profit
            })
        
        return standard_news


def load_stock_list(json_path: str, filter_negative_news: bool = True) -> List[StockListItem]:
    """便捷函數：加載股票列表"""
    loader = EnhancedStockListLoader(json_path)
    return loader.load(filter_negative_news=filter_negative_news)


def get_stock_codes_from_json(json_path: str, filter_negative_news: bool = True) -> List[str]:
    """便捷函數：獲取股票代碼列表"""
    loader = EnhancedStockListLoader(json_path)
    return loader.get_stock_codes(filter_negative_news=filter_negative_news)