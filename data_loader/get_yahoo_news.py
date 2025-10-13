import requests
from bs4 import BeautifulSoup
import re
import json


def get_yahoo_finance_news_html(stock_code):
    """
    獲取雅虎財經特定股票的新聞頁面HTML
    
    Args:
        stock_code (str): 股票代碼，例如 '0471.HK'
        
    Returns:
        str: 頁面HTML內容
    """
    # 構建URL
    url = f"https://hk.finance.yahoo.com/quote/{stock_code}/news/"
    
    # 設置請求頭，模仿真實瀏覽器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # 發送請求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 檢查HTTP錯誤
        
        # 返回HTML內容
        return response.text

    except requests.exceptions.RequestException as e:
        print(f"獲取頁面時發生錯誤: {e}")
        return None


def extract_hk_stock_news(html_content, stock_code):
    """
    從HTML內容中提取港股股票新聞標題和連結
    
    Args:
        html_content (str): HTML內容
        stock_code (str): 股票代碼，例如 '0471.HK'
    
    Returns:
        list: 包含新聞標題和連結的字典列表
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    news_items = []
    
    # 創建股票代碼的靈活匹配模式
    code = stock_code.replace('.HK', '')
    # 匹配各種可能的代碼格式
    pattern = re.compile(rf'({re.escape(stock_code)}|{re.escape(code)}|{code.lstrip("0")})', re.IGNORECASE)
    
    # 方法1: 從JSON數據提取所有包含股票代碼的新聞
    scripts = soup.find_all('script', type='application/json')
    
    for script in scripts:
        try:
            json_data = json.loads(script.get_text())
            
            if 'data' in json_data and 'tickerStream' in json_data['data']:
                stream_data = json_data['data']['tickerStream']
                
                if 'stream' in stream_data:
                    for item in stream_data['stream']:
                        if 'content' in item:
                            content = item['content']
                            title = content.get('title', '')
                            url = content.get('url', '')
                            
                            # 使用靈活的模式匹配股票代碼
                            if pattern.search(title):
                                news_items.append({
                                    'title': title.strip(),
                                    'url': url,
                                    'summary': content.get('summary', '')
                                })
        except (json.JSONDecodeError, TypeError):
            continue
    
    # 方法2: 從HTML中搜尋所有包含股票代碼的連結
    # 搜尋包含股票代碼的文本節點
    text_nodes = soup.find_all(string=pattern)
    
    for text_node in text_nodes:
        # 往上搜尋直到找到包含新聞URL的元素
        parent = text_node.parent
        while parent and parent.name != 'body':
            if parent.name == 'a' and parent.get('href') and 'news' in parent['href']:
                title = text_node.strip()
                url = parent['href']
                
                # 處理相對連結
                if url.startswith('/'):
                    url = 'https://hk.finance.yahoo.com' + url
                
                # 如果新聞標題包含股票代碼且不重複，則添加
                if pattern.search(title) and not any(item['title'] == title for item in news_items):
                    news_items.append({
                        'title': title,
                        'url': url,
                        'summary': ''
                    })
            parent = parent.parent
    
    return news_items


def get_news_title_regex():
    """
    返回通用的港股新聞標題匹配正則表達式
    """
    # 匹配包含港股代碼的中文標題的正則表達式（不限定特定內容）
    return r'(?:[^\n\r>]*?)(?:\d{4,5}\.HK|\d{4,5})[^\n\r<]*?(?=，|。|！|？|；|、|\n|\r|<|$)'


def get_stock_news_pattern(stock_code):
    """
    返回針對特定股票代碼的匹配模式
    
    Args:
        stock_code (str): 股票代碼，例如 '0471.HK'
    
    Returns:
        str: 正則表達式模式
    """
    code = stock_code.replace('.HK', '')
    # 支持多種代碼格式匹配
    return f'(?:[^\n\r>]*?)({re.escape(stock_code)}|{re.escape(code)}|{code.lstrip("0")})[^\n\r<]*?(?=，|。|！|？|；|、|\n|\r|<|$)'


# 使用範例
if __name__ == "__main__":
    # 測試多個股票代碼
    stock_codes = ["0471.HK", "0700.HK"]  # 可以替換為任何你想測試的港股代碼
    
    for stock_code in stock_codes:
        print(f"正在處理股票代碼: {stock_code}")
        html_content = get_yahoo_finance_news_html(stock_code)
        
        if html_content:
            print("成功獲取HTML內容")
            print(f"HTML內容長度: {len(html_content)} 字元")
            
            # 提取新聞標題和連結
            news_items = extract_hk_stock_news(html_content, stock_code)
            
            print(f"\n找到 {len(news_items)} 條相關新聞:")
            print("-" * 50)
            for i, item in enumerate(news_items, 1):
                print(f"{i}. 標題: {item['title']}")
                print(f"   連結: {item['url']}")
                if item['summary']:
                    print(f"   摘要: {item['summary']}")
                print()
            
            # 顯示使用的正則表達式
            print("使用的正則表達式:")
            print("-" * 30)
            print(f"通用新聞標題匹配: {get_news_title_regex()}")
            print(f"特定股票匹配: {get_stock_news_pattern(stock_code)}")
            print("\n" + "="*60 + "\n")
            
        else:
            print("無法獲取HTML內容")