#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import json
import re
from datetime import datetime, timezone, timedelta

rss_url = "https://hk.finance.yahoo.com/rss/"

hk_tz = timezone(timedelta(hours=8))
now = datetime.now(hk_tz)
cutoff = now - timedelta(hours=24)

print(f"Current time (HKT): {now}")
print(f"Cutoff time (HKT): {cutoff}")

import urllib.request
req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as response:
    rss_content = response.read().decode('utf-8')

root = ET.fromstring(rss_content)
channel = root.find('channel')
items = channel.findall('item')

print(f"Total items: {len(items)}")

earnings_keywords = ['盈喜', '盈警', '預告', '預計', '純利', '虧損', '利潤', '業績', '全年']
rating_keywords = ['目標價', '評級', '上調', '下調', '買入', '增持', '減持', '賣出', '維持', '大行', '高盛', '摩根', '瑞銀', '美銀', '中金', '匯豐', '美銀']

results = []

for item in items:
    title = item.find('title').text if item.find('title') is not None else ""
    link = item.find('link').text if item.find('link') is not None else ""
    pub_date_str = item.find('pubDate').text if item.find('pubDate') is not None else ""
    
    try:
        if '+00:00' in pub_date_str:
            pub_dt = datetime.fromisoformat(pub_date_str.replace('+00:00', '+0000'))
        else:
            pub_dt = datetime.strptime(pub_date_str.strip(), '%a, %d %b %Y %H:%M:%S %z')
        pub_dt_hkt = pub_dt.astimezone(hk_tz)
    except:
        continue
    
    if pub_dt_hkt < cutoff:
        continue
    
    is_earnings = any(kw in title for kw in earnings_keywords)
    is_rating = any(kw in title for kw in rating_keywords)
    
    if not (is_earnings or is_rating):
        continue
    
    # Extract stock codes from title like (01675.HK) or from link
    codes = re.findall(r'\((\d{5})\.HK\)', title)
    codes += re.findall(r'\((\d{4})\.HK\)', title)
    codes += re.findall(r'(\d{5})\.HK', link.upper())
    codes += re.findall(r'(\d{4})\.HK', link.upper())
    codes = list(set(codes))
    codes = [c + '.HK' for c in codes]
    
    source_elem = item.find('source')
    source = source_elem.text if source_elem is not None and source_elem.get('url') is None else "Yahoo Finance"
    
    result = {
        "title": title,
        "link": link,
        "pubDate": pub_dt_hkt.isoformat(),
        "source": source,
        "stockCodes": codes,
        "category": "earnings" if is_earnings else "rating"
    }
    results.append(result)

output = {
    "collectTime": now.isoformat(),
    "timeRange": f"{cutoff.isoformat()} to {now.isoformat()}",
    "count": len(results),
    "news": results
}

output_file = "hk_news_24h.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n=== Saved {len(results)} articles ===")
for r in results:
    cat = "E" if r['category'] == 'earnings' else "R"
    codes_str = f" {r['stockCodes']}" if r['stockCodes'] else ""
    print(f"[{cat}]{codes_str} {r['title'][:70]}")
