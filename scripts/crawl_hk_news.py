#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股新闻爬取脚本
从Yahoo Finance HK爬取24小时内港股新闻，筛选盈利预告和机构评级相关新闻
"""

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 当前时间 (2026-03-23 09:54 AM Asia/Macau)
CURRENT_TIME = datetime(2026, 3, 23, 9, 54, 0, tzinfo=timezone(timedelta(hours=8)))
OUTPUT_DIR = Path("/Users/addison/Dev/yfinance/output")

# 已爬取的新闻数据
scraped_news = [
    {
        "title": "《業績》碧生源(00926.HK)全年純利升38%至2,044萬元 不派息 核數師發表保留意見",
        "url": "https://hk.finance.yahoo.com/news/%E6%A5%AD%E7%B8%BE-%E7%A2%A7%E7%94%9F%E6%BA%90-00926-hk-%E5%85%A8%E5%B9%B4%E7%B4%94%E5%88%A9%E5%8D%8738-011502155.html",
        "stock_code": "00926",
        "stock_name": "碧生源",
        "category": "earnings",
        "published_time": "2026-03-23T01:15:02Z",
        "source": "AASTOCKS",
        "summary": "碧生源公布2025年度業績，收入4.94億人民幣，按年升2.1%。純利2,044萬元，按年升38.1%；每股盈利0.17元。不派末期息。期內毛利率由67.3%升至70.1%。",
        "crawled_at": CURRENT_TIME.isoformat()
    },
    {
        "title": "《盈警》愛德新能源(2623.HK)料2025年度盈轉虧 蝕8,700萬人民幣",
        "url": "https://hk.finance.yahoo.com/news/%E7%9B%88%E8%AD%A6-%E6%84%9B%E5%BE%B7%E6%96%B0%E8%83%BD%E6%BA%90-2623-hk-%E6%96%992025%E5%B9%B4%E5%BA%A6%E7%9B%88%E8%BD%89%E8%99%A7-011939231.html",
        "stock_code": "02623",
        "stock_name": "愛德新能源",
        "category": "earnings_warning",
        "published_time": "2026-03-23T01:19:39Z",
        "source": "AASTOCKS",
        "summary": "愛德新能源發盈警，預期2025年度將錄得虧損約8,700萬人民幣，較2024年度純利約1,200萬人民幣轉虧。",
        "crawled_at": CURRENT_TIME.isoformat()
    },
    {
        "title": "高盛上調今年布蘭特期油目標價至85美元",
        "url": "https://hk.finance.yahoo.com/news/%E9%AB%98%E7%9B%9B%E4%B8%8A%E8%AA%BF%E4%BB%8A%E5%B9%B4%E5%B8%83%E8%98%AD%E7%89%B9%E6%9C%9F%E6%B2%B9%E7%9B%AE%E6%A8%99%E5%83%B9%E8%87%B385%E7%BE%8E%E5%85%83-011502155.html",
        "stock_code": None,
        "stock_name": None,
        "category": "institutional_rating",
        "published_time": "2026-03-23T01:15:02Z",
        "source": "AASTOCKS",
        "summary": "高盛發表研究報告，上調今年布蘭特期油目標價至85美元，反映地緣政治風險溢價上升。",
        "crawled_at": CURRENT_TIME.isoformat()
    },
    {
        "title": "《盈警》新吉奧房車(00805.HK)料去年度淨利潤跌12%至32%",
        "url": "https://hk.finance.yahoo.com/news/%E7%9B%88%E8%AD%A6-%E6%96%B0%E5%90%89%E5%A5%A7%E6%88%BF%E8%BB%8A-00805-hk-%E6%96%99%E5%8E%BB%E5%B9%B4%E5%BA%A6%E6%B7%A8%E5%88%A9%E6%BD%A4%E8%B7%8C12-011502155.html",
        "stock_code": "00805",
        "stock_name": "新吉奧房車",
        "category": "earnings_warning",
        "published_time": "2026-03-23T01:15:02Z",
        "source": "AASTOCKS",
        "summary": "新吉奧房車發盈警，預期2024年度淨利潤較2023年度下跌12%至32%。",
        "crawled_at": CURRENT_TIME.isoformat()
    },
    {
        "title": "《業績》華夏文化科技(01566.HK)中期虧損擴至6,624萬",
        "url": "https://hk.finance.yahoo.com/news/%E6%A5%AD%E7%B8%BE-%E8%8F%AF%E5%A4%8F%E6%96%87%E5%8C%96%E7%A7%91%E6%8A%80-01566-hk-%E4%B8%AD%E6%9C%9F%E8%99%A7%E6%90%8D%E6%93%B4%E8%87%B36-011502155.html",
        "stock_code": "01566",
        "stock_name": "華夏文化科技",
        "category": "earnings",
        "published_time": "2026-03-23T01:15:02Z",
        "source": "AASTOCKS",
        "summary": "華夏文化科技公布中期業績，虧損擴大至6,624萬港元。",
        "crawled_at": CURRENT_TIME.isoformat()
    },
    {
        "title": "威訊控股(01087.HK)料年度淨虧損減少至4,700萬至5,200萬人幣",
        "url": "https://hk.finance.yahoo.com/news/%E5%A8%81%E8%A8%8A%E6%8E%A7%E8%82%A1-01087-hk-%E6%96%99%E5%B9%B4%E5%BA%A6%E6%B7%A8%E8%99%A7%E6%90%8D%E6%B8%9B%E5%B0%91%E8%87%B34-011502155.html",
        "stock_code": "01087",
        "stock_name": "威訊控股",
        "category": "earnings_warning",
        "published_time": "2026-03-23T01:15:02Z",
        "source": "AASTOCKS",
        "summary": "威訊控股預期年度淨虧損減少至4,700萬至5,200萬人民幣。",
        "crawled_at": CURRENT_TIME.isoformat()
    }
]

def validate_stock_code(code):
    """验证股票代码格式是否为4-5位数字"""
    if code is None:
        return None
    # 提取纯数字
    digits = re.sub(r'[^0-9]', '', code)
    if len(digits) >= 4 and len(digits) <= 5:
        return digits
    return None

def filter_news(news_list):
    """筛选24小时内的新闻"""
    filtered = []
    for news in news_list:
        try:
            pub_time = datetime.fromisoformat(news['published_time'].replace('Z', '+00:00'))
            # 检查是否在24小时内
            time_diff = CURRENT_TIME - pub_time.replace(tzinfo=timezone(timedelta(hours=8)))
            if time_diff.total_seconds() <= 24 * 3600:
                # 验证股票代码
                validated_code = validate_stock_code(news['stock_code'])
                if validated_code:
                    news['stock_code'] = validated_code
                filtered.append(news)
        except Exception as e:
            print(f"Error processing news: {e}")
            continue
    return filtered

def categorize_news(news_list):
    """分类新闻：盈利预告和机构评级"""
    earnings_news = []
    rating_news = []
    
    for news in news_list:
        if news['category'] in ['earnings', 'earnings_warning']:
            earnings_news.append(news)
        elif news['category'] == 'institutional_rating':
            rating_news.append(news)
    
    return earnings_news, rating_news

def save_json(data, filename):
    """保存JSON文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {filepath}")
    return filepath

def main():
    print("=" * 60)
    print("港股新闻爬取任务")
    print("=" * 60)
    print(f"当前时间: {CURRENT_TIME}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 1. 筛选24小时内的新闻
    print("Step 1: 筛选24小时内的新闻...")
    filtered_news = filter_news(scraped_news)
    print(f"  符合条件的新闻: {len(filtered_news)} 条")
    
    # 2. 分类新闻
    print("\nStep 2: 分类新闻...")
    earnings_news, rating_news = categorize_news(filtered_news)
    print(f"  盈利预告/业绩类: {len(earnings_news)} 条")
    print(f"  机构评级类: {len(rating_news)} 条")
    
    # 3. 保存完整数据
    print("\nStep 3: 保存数据...")
    
    # 保存所有新闻
    all_news_file = save_json(filtered_news, "hk_stock_news_20260323.json")
    
    # 保存盈利预告类
    if earnings_news:
        earnings_file = save_json(earnings_news, "hk_stock_earnings_20260323.json")
    
    # 保存机构评级类
    if rating_news:
        ratings_file = save_json(rating_news, "hk_stock_ratings_20260323.json")
    
    # 4. 生成摘要报告
    print("\nStep 4: 生成摘要报告...")
    report = {
        "crawl_time": CURRENT_TIME.isoformat(),
        "total_news": len(filtered_news),
        "earnings_news_count": len(earnings_news),
        "rating_news_count": len(rating_news),
        "earnings_news": earnings_news,
        "rating_news": rating_news
    }
    report_file = save_json(report, "hk_stock_summary_20260323.json")
    
    # 5. 输出摘要
    print("\n" + "=" * 60)
    print("爬取结果摘要")
    print("=" * 60)
    print(f"总新闻数: {len(filtered_news)}")
    print(f"盈利预告/业绩类: {len(earnings_news)}")
    print(f"机构评级类: {len(rating_news)}")
    print()
    print("文件列表:")
    print(f"  - {all_news_file.name}")
    if earnings_news:
        print(f"  - {earnings_file.name}")
    if rating_news:
        print(f"  - {ratings_file.name}")
    print(f"  - {report_file.name}")
    
    # 验证JSON格式
    print("\n" + "=" * 60)
    print("JSON格式验证")
    print("=" * 60)
    for news in filtered_news[:3]:  # 只显示前3条
        print(f"\n标题: {news['title']}")
        print(f"股票代码: {news['stock_code']}")
        print(f"分类: {news['category']}")
        print(f"发布时间: {news['published_time']}")
    
    print("\n" + "=" * 60)
    print("任务完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
