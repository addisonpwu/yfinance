#!/usr/bin/env python3
"""
Sync processed Hong Kong stock news to local API
"""

import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import re

API_BASE_URL = "http://localhost:8000"

def normalize_hk_code(code: str) -> str:
    """
    Convert HK stock code to standard format (4 digits + .HK)
    
    Rules:
    - If code has more than 4 digits: remove leading zeros to get 4 digits
    - If code has less than 4 digits: pad with leading zeros
    """
    digits = re.sub(r'\D', '', code)
    
    if not digits:
        return code.upper()
    
    if len(digits) > 4:
        normalized = digits.lstrip('0') or '0'
        normalized = normalized[-4:].zfill(4)
    else:
        normalized = digits.zfill(4)
    
    return f"{normalized}.HK"

def check_stock_exists(symbol: str) -> bool:
    """Check if a stock exists in the system"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stocks/{symbol}", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def create_news(news_data: Dict) -> Tuple[bool, str]:
    """Create a news entry via API"""
    try:
        # Convert sentiment to integer
        sentiment = 1 if news_data['sentiment'] == 'positive' else 0
        
        payload = {
            "stock_symbol": news_data['stock_symbol'],
            "title": news_data['title'],
            "content": news_data.get('content', ''),
            "sentiment": sentiment,
            "publish_time": datetime.now().isoformat(),
            "url": news_data.get('url', '')
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/news/",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 201:
            return True, "Success"
        elif response.status_code == 409:
            return False, "Duplicate"
        else:
            return False, f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {str(e)[:50]}"

def main():
    print("=" * 60)
    print("Syncing Hong Kong Stock News to API")
    print("=" * 60)
    
    # Find the most recent processed news file
    temp_dir = Path.home() / "Documents" / "temp"
    news_files = sorted(temp_dir.glob("stock_2026-*.json"), reverse=True)
    
    if not news_files:
        print("No processed news files found!")
        return
    
    latest_file = news_files[0]
    print(f"\nUsing file: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    news_items = data.get('news_items', [])
    print(f"Total news items to sync: {len(news_items)}")
    
    # Deduplicate by URL
    seen_urls = set()
    unique_news = []
    for item in news_items:
        url = item.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_news.append(item)
    
    print(f"Unique news items (by URL): {len(unique_news)}")
    
    # Group by sentiment
    positive_news = [n for n in unique_news if n['sentiment'] == 'positive']
    negative_news = [n for n in unique_news if n['sentiment'] == 'negative']
    
    print(f"Positive news: {len(positive_news)}")
    print(f"Negative news: {len(negative_news)}")
    
    # Process positive news (directly add)
    print("\n" + "=" * 60)
    print("Processing POSITIVE news...")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for item in positive_news:
        success, msg = create_news(item)
        if success:
            success_count += 1
            print(f"✓ {item['stock_symbol']}: {item['title'][:40]}...")
        elif msg == "Duplicate":
            skip_count += 1
        else:
            error_count += 1
            print(f"✗ Error {item['stock_symbol']}: {msg}")
        
        time.sleep(0.05)
    
    print(f"\nPositive: Success={success_count}, Skip={skip_count}, Error={error_count}")
    
    # Process negative news (check stock exists first)
    print("\n" + "=" * 60)
    print("Processing NEGATIVE news (checking stock exists first)...")
    print("=" * 60)
    
    neg_success = 0
    neg_skip = 0
    neg_error = 0
    
    for item in negative_news:
        symbol = item['stock_symbol']
        
        # Check if stock exists
        if not check_stock_exists(symbol):
            neg_skip += 1
            print(f"- Skipped (not in system): {symbol}")
            continue
        
        success, msg = create_news(item)
        if success:
            neg_success += 1
            print(f"✓ {symbol}: {item['title'][:40]}...")
        elif msg == "Duplicate":
            neg_skip += 1
        else:
            neg_error += 1
            print(f"✗ Error {symbol}: {msg}")
        
        time.sleep(0.05)
    
    print(f"\nNegative: Success={neg_success}, Skip={neg_skip}, Error={neg_error}")
    
    print("\n" + "=" * 60)
    print("Sync Complete!")
    print("=" * 60)
    total_success = success_count + neg_success
    total_skip = skip_count + neg_skip
    total_error = error_count + neg_error
    print(f"Total added: {total_success}")
    print(f"Total skipped: {total_skip}")
    print(f"Total errors: {total_error}")

if __name__ == '__main__':
    main()
