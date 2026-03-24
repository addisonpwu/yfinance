import json
import re

with open('hk_news_24h.json', 'r') as f:
    data = json.load(f)

print("=== JSON Validation ===")
print(f"Valid JSON: YES")
print(f"Count: {data['count']}")
print(f"Time range: {data['timeRange']}")
print(f"Collect time: {data['collectTime']}")

earnings = [n for n in data['news'] if n['category'] == 'earnings']
rating = [n for n in data['news'] if n['category'] == 'rating']

print(f"\n=== Category Breakdown ===")
print(f"Earnings news (盈利預告): {len(earnings)}")
print(f"Rating news (機構評級): {len(rating)}")

print(f"\n=== Stock Code Validation ===")
pattern = r'^\d{5}\.HK$'
invalid = []
for n in data['news']:
    codes = n.get('stockCodes', [])
    for c in codes:
        if not re.match(pattern, c):
            invalid.append((n['title'][:30], c))

if invalid:
    print("Invalid codes found:")
    for title, code in invalid:
        print(f"  {title}... -> {code}")
else:
    print("All stock codes are valid (5-digit.HK format)")

print(f"\n=== Sample Data ===")
for n in data['news'][:3]:
    print(f"Title: {n['title'][:50]}...")
    print(f"  Category: {n['category']}")
    print(f"  Codes: {n['stockCodes']}")
    print(f"  Date: {n['pubDate']}")
    print()
