#!/usr/bin/env python3
"""
Process Hong Kong stock news from JSON files
Classify as positive/negative and prepare for API sync
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def normalize_hk_code(code: str) -> str:
    """
    Convert HK stock code to standard format (4 digits + .HK)
    
    Rules:
    - If code has more than 4 digits: remove leading zeros to get 4 digits
    - If code has less than 4 digits: pad with leading zeros
    - Examples: 01187.HK → 1187.HK | 01941.HK → 1941.HK | 5.HK → 0005.HK
    """
    # Extract just digits
    digits = re.sub(r'\D', '', code)
    
    if not digits:
        return code.upper()
    
    # Handle the normalization based on number of digits
    if len(digits) > 4:
        # Remove leading zeros until we have 4 digits
        normalized = digits.lstrip('0') or '0'
        # Take the last 4 digits
        normalized = normalized[-4:].zfill(4)
    else:
        # Pad to 4 digits
        normalized = digits.zfill(4)
    
    return f"{normalized}.HK"

def extract_hk_codes(text: str) -> List[str]:
    """Extract Hong Kong stock codes from text"""
    codes = []
    seen = set()
    
    # Match patterns like: (XXXXX.HK), XXXX.HK, etc.
    patterns = [
        r'\((\d{4,5})\.HK\)',  # (1234.HK) or (12345.HK)
        r'(\d{4,5})\.HK',       # 1234.HK or 12345.HK
        r'\((\d{4,5})\)(?!\d)',  # (1234) - common in aastocks format, followed by non-digit
        r'股\w*\((\d{4,5})\)',  # 騰訊(00700)
        r'[\(（](\d{4,5})[\)）]',  # (1234) or （1234）
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            code = normalize_hk_code(match.group(1))
            if code not in seen:
                seen.add(code)
                codes.append(code)
    
    return codes

# Positive news keywords (with weights)
POSITIVE_KEYWORDS = [
    ('盈喜', 3), ('預增', 3), ('業績預告', 2), ('內幕消息', 3), 
    ('正面盈喜', 3), ('業績大增', 3), ('純利增長', 3), ('盈利上升', 3),
    ('買入', 2), ('增持', 2), ('上調評級', 3), ('強烈推薦', 3),
    ('跑贏大市', 2), ('首選', 2), ('目標價', 1), ('看好', 2),
    ('推薦', 2), ('升級', 3), ('首季業績勝預期', 3), ('淨流入', 2),
    ('大漲', 3), ('飆升', 3), ('上漲', 2), ('升逾', 2), ('升幅', 2),
    ('急升', 3), ('漲超', 3), ('維持增持', 2), ('維持買入', 2),
    ('給予增持', 2), ('給予買入', 2), ('重申', 1), ('上調目標價', 2),
    ('勝預期', 2), ('高於預期', 2), ('超預期', 2),
    ('創新高', 2), ('反彈', 2), ('回升', 2), ('突破', 1),
    ('受惠', 2), ('利好', 2), ('正面', 2), ('好過預期', 2)
]

# Negative news keywords (with weights)
NEGATIVE_KEYWORDS = [
    ('盈警', 3), ('虧損', 3), ('下調評級', 3), ('降低評級', 3),
    ('違約', 3), ('訴訟', 2), ('盈利倒退', 3), ('退市風險', 3),
    ('跌', 1), ('挫', 2), ('低開', 2), ('下跌', 2), ('大跌', 3),
    ('倒跌', 3), ('淨流出', 2), ('減持', 2), ('遜預期', 2),
    ('低於預期', 2), ('降級', 3), ('下調目標價', 2), ('預期下調', 2),
    ('弱', 1), ('差過預期', 2), ('蝕', 3), ('派息削減', 3),
    ('被收回', 2), ('被狙擊', 3), ('停牌', 2), ('短暫停牌', 2)
]

def classify_news(title: str, content: str) -> Tuple[str, List[str]]:
    """Classify news as positive or negative based on keywords"""
    text = title + ' ' + content
    
    # Calculate weighted scores
    positive_score = 0
    negative_score = 0
    
    for keyword, weight in POSITIVE_KEYWORDS:
        if keyword in text:
            positive_score += weight
            
    for keyword, weight in NEGATIVE_KEYWORDS:
        if keyword in text:
            negative_score += weight
    
    # Extract stock codes
    stock_codes = extract_hk_codes(text)
    
    # Classify based on scores
    if negative_score > positive_score:
        return 'negative', stock_codes
    elif positive_score > negative_score:
        return 'positive', stock_codes
    else:
        # Default based on specific patterns
        if any(kw in text for kw in ['升', '漲', '買入', '增持', '評級上調', '跑贏']):
            return 'positive', stock_codes
        elif any(kw in text for kw in ['跌', '挫', '減持', '下調', '虧']):
            return 'negative', stock_codes
        # Neutral - classify as positive for news purposes
        return 'positive', stock_codes

def process_news_files():
    """Process all news JSON files and return classified news"""
    news_files = [
        '~/Documents/temp/stock_news.json',
        '~/Documents/temp/aastocks_news.json',
        '~/Documents/temp/eastmoney_hk_news.json',
        '~/Documents/temp/sinafinance_hk_news.json'
    ]
    
    all_processed_news = []
    stock_news_count = {}
    
    for news_file in news_files:
        file_path = Path(news_file).expanduser()
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Handle potential extra data after JSON
                try:
                    news_data = json.loads(content)
                except json.JSONDecodeError:
                    # Try to find the last complete JSON object
                    for i in range(len(content), 0, -1):
                        try:
                            news_data = json.loads(content[:i])
                            print(f"Warning: {file_path.name} had extra data, truncated")
                            break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        for item in news_data:
            title = item.get('title', '')
            content = item.get('content', '')
            url = item.get('url', '')
            source = item.get('source', '')
            time_str = item.get('time', '')
            news_type = item.get('type', '')
            
            # Classify news
            sentiment, stock_codes = classify_news(title, content)
            
            # Filter for HK stocks only
            if stock_codes:
                for code in stock_codes:
                    # Count occurrences
                    if code not in stock_news_count:
                        stock_news_count[code] = {'positive': 0, 'negative': 0}
                    stock_news_count[code][sentiment] += 1
                    
                    news_entry = {
                        'stock_symbol': code,
                        'title': title,
                        'content': content[:500] if content else '',
                        'url': url,
                        'source': source,
                        'time': time_str,
                        'news_type': news_type,
                        'sentiment': sentiment,
                        'original_file': file_path.name
                    }
                    all_processed_news.append(news_entry)
    
    return all_processed_news, stock_news_count

def main():
    print("=" * 60)
    print("Processing Hong Kong Stock News")
    print("=" * 60)
    
    # Test normalization
    test_codes = ['01187.HK', '00700.HK', '01941.HK', '5.HK', '00100.HK', '2883.HK']
    print("\nNormalization test:")
    for code in test_codes:
        print(f"  {code} -> {normalize_hk_code(code)}")
    
    # Process news files
    all_news, stock_counts = process_news_files()
    
    print(f"\nTotal news items processed: {len(all_news)}")
    print(f"Unique stocks found: {len(stock_counts)}")
    
    # Count positive and negative
    positive_count = sum(1 for n in all_news if n['sentiment'] == 'positive')
    negative_count = sum(1 for n in all_news if n['sentiment'] == 'negative')
    print(f"Positive news: {positive_count}")
    print(f"Negative news: {negative_count}")
    
    # Generate output filename with current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f"~/Documents/temp/stock_{timestamp}.json"
    
    # Prepare output data
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'total_news_items': len(all_news),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'unique_stocks': len(stock_counts),
        'news_items': all_news,
        'stock_summary': {
            code: counts for code, counts in stock_counts.items()
        }
    }
    
    # Write output file
    output_path = Path(output_file).expanduser()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nOutput saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Stock Summary (Top 20):")
    print("=" * 60)
    sorted_stocks = sorted(stock_counts.items(), 
                          key=lambda x: x[1]['positive'] + x[1]['negative'], 
                          reverse=True)[:20]
    for code, counts in sorted_stocks:
        print(f"  {code}: Positive={counts['positive']}, Negative={counts['negative']}")
    
    return output_path

if __name__ == '__main__':
    main()
