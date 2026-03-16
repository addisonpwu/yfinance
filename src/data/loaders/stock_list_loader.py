"""
股票列表 JSON 加载器

从 JSON 文件加载股票列表和新闻数据
"""
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StockNewsItem:
    """股票新闻项"""
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
    """股票列表项"""
    stock_code: str
    stock_name: str
    news: List[StockNewsItem]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stockCode': self.stock_code,
            'stockName': self.stock_name,
            'news': [n.to_dict() for n in self.news]
        }


class StockListLoader:
    """股票列表 JSON 加载器"""
    
    def __init__(self, json_path: str = None):
        """
        初始化加载器
        
        Args:
            json_path: JSON 文件路径
        """
        self.json_path = json_path
    
    def load(self, json_path: str = None) -> List[StockListItem]:
        """
        加载股票列表
        
        Args:
            json_path: JSON 文件路径（可选，覆盖初始化时设置的路径）
        
        Returns:
            股票列表
        """
        path = json_path or self.json_path
        if not path:
            raise ValueError("未指定 JSON 文件路径")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON 文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._parse_data(data)
    
    def _parse_data(self, data: List[Dict[str, Any]]) -> List[StockListItem]:
        """解析 JSON 数据"""
        items = []
        
        for item in data:
            stock_code = item.get('stockCode', '')
            stock_name = item.get('stockName', '')
            news_list = item.get('news', [])
            
            # 解析新闻列表
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
                news_items.append(news_item)
            
            items.append(StockListItem(
                stock_code=stock_code,
                stock_name=stock_name,
                news=news_items
            ))
        
        return items
    
    def get_stock_codes(self, json_path: str = None) -> List[str]:
        """
        获取股票代码列表
        
        Args:
            json_path: JSON 文件路径
        
        Returns:
            股票代码列表
        """
        items = self.load(json_path)
        return [item.stock_code for item in items]
    
    def get_stock_with_news(self, json_path: str = None) -> Dict[str, StockListItem]:
        """
        获取股票代码到股票信息的映射
        
        Args:
            json_path: JSON 文件路径
        
        Returns:
            {stock_code: StockListItem} 映射
        """
        items = self.load(json_path)
        return {item.stock_code: item for item in items}
    
    def convert_news_to_standard_format(self, news_items: List[StockNewsItem]) -> List[Dict[str, Any]]:
        """
        将新闻转换为系统标准格式
        
        系统标准格式:
        {
            'title': str,
            'link': str,
            'published': str,
            'summary': str,
            'publisher': str,
            'source': str
        }
        
        Args:
            news_items: 新闻列表
        
        Returns:
            标准格式的新闻列表
        """
        standard_news = []
        
        for item in news_items:
            # 构建摘要
            summary_parts = []
            if item.rating:
                summary_parts.append(f"评级: {item.rating}")
            if item.profit:
                summary_parts.append(f"业绩: {item.profit}")
            if item.agency:
                summary_parts.append(f"机构: {item.agency}")
            
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


def load_stock_list(json_path: str) -> List[StockListItem]:
    """便捷函数：加载股票列表"""
    loader = StockListLoader(json_path)
    return loader.load()


def get_stock_codes_from_json(json_path: str) -> List[str]:
    """便捷函数：获取股票代码列表"""
    loader = StockListLoader(json_path)
    return loader.get_stock_codes()
