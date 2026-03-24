#!/usr/bin/env python3
"""
Embed stock.json data into stock_dashboard.html to fix CORS issue
"""

import json
import re
from pathlib import Path

# Read the stock.json data
with open('reports/stock.json', 'r', encoding='utf-8') as f:
    stock_data = json.load(f)

# Read the dashboard HTML
with open('reports/stock_dashboard.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Convert stock data to JSON string for embedding
stock_json_str = json.dumps(stock_data, ensure_ascii=False)

# New code with embedded data
new_code = f"""        // Embedded data (to avoid CORS issues when opening file directly)
        const embeddedData = {stock_json_str};

        // Load and process data
        function loadData() {{
            try {{
                // Use embedded data directly
                const data = embeddedData;
                stockData = data.stocks || [];
                filteredData = [...stockData];

                // Update last updated time
                if (data.metadata && data.metadata.mergedAt) {{
                    const date = new Date(data.metadata.mergedAt);
                    document.getElementById('lastUpdated').textContent =
                        date.toLocaleString('zh-HK', {{
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit'
                        }});
                }} else {{
                    // Use current time if no metadata
                    const now = new Date();
                    document.getElementById('lastUpdated').textContent =
                        now.toLocaleString('zh-HK', {{
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit'
                        }});
                }}

                processAndRender();
            }} catch (error) {{
                console.error('Error loading data:', error);
                document.querySelector('.stats-grid').innerHTML =
                    '<div class="loading"><div class="spinner"></div>數據加載失敗 | Data Load Failed</div>';
            }}
        }}"""

# Use regex to replace the loadData function
pattern = r'        // Load and process data\n        async function loadData\(\) \{[^}]+\}\n        \}'
replacement = new_code

# Replace
new_html = re.sub(pattern, replacement, html_content, flags=re.DOTALL)

# Write the updated HTML
with open('reports/stock_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

print("✅ Successfully embedded stock.json data into stock_dashboard.html")
print(f"📊 Total stocks: {len(stock_data.get('stocks', []))}")
print(f"📰 Total news items: {sum(len(s.get('news', [])) for s in stock_data.get('stocks', []))}")
