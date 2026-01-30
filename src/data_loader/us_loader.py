
import requests
import urllib3
import re

def get_us_tickers():
    """從財報狗網站自動獲取所有美股代碼列表"""
    print("正在從財報狗(statementdog.com)獲取可用的美股代碼列表...")
    try:
        # 停用 SSL 警告訊息
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        url = 'https://statementdog.com/us-stock-list'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status() # 如果請求失敗則拋出錯誤

        # 使用精準的正規表示式，只匹配 <span class="us-stock-company-ticker">XXXX</span> 結構中的代碼
        tickers = re.findall(r'<span class="us-stock-company-ticker">([A-Z0-9\.-]+)</span>', response.text)
        
        if not tickers:
            print("無法從頁面內容中解析出股票代碼，將使用備用列表。")
            raise ValueError("No tickers found")

        print(f"成功獲取 {len(tickers)} 支美股代碼。")
        return tickers
    except Exception as e:
        print(f"無法自動獲取美股列表，錯誤: {e}")
        print("將使用預設的少量股票列表進行分析。")
        return [
            'NVDA', 'AAPL', 'MSFT', 'TSLA', 'META', 'GOOGL' # 備用列表
        ]
