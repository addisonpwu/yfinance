#!/usr/bin/env python3
"""
自动股票新闻搜索脚本
读取yfinance生成的股票列表，然后搜索每只股票的最新新闻
"""

import json
import os
import sys
from datetime import datetime

def read_stock_list():
    """读取股票列表文件"""
    report_file = f"/Users/addison/Dev/yfinance/reports/us_stocks_{datetime.now().strftime('%Y-%m-%d')}.json"
    
    if not os.path.exists(report_file):
        print(f"❌ 报告文件不存在: {report_file}")
        return []
    
    with open(report_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 找到 {data['total_stocks']} 只股票: {data['stocks']}")
    return data['stocks']

def search_stock_news(stocks):
    """搜索股票新闻"""
    news_results = {}
    
    for stock in stocks:
        print(f"🔍 搜索 {stock} 最新新闻...")
        # 这里可以集成OpenClaw的搜索功能
        # 目前先模拟搜索
        news_results[stock] = f"{stock} 的最新新闻搜索结果"
    
    return news_results

def main():
    """主函数"""
    print("🚀 开始自动股票新闻搜索")
    
    # 1. 读取股票列表
    stocks = read_stock_list()
    if not stocks:
        return
    
    # 2. 搜索新闻
    news_results = search_stock_news(stocks)
    
    # 3. 输出结果
    print("\n📰 新闻搜索结果:")
    for stock, result in news_results.items():
        print(f"   {stock}: {result}")
    
    print("\n✅ 搜索完成!")

if __name__ == "__main__":
    main()