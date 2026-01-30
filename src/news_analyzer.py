# news_analyzer.py

from playwright.sync_api import sync_playwright
import json

def fetch_news_from_yahoo(symbol: str, market: str) -> list[dict]:
    """
    使用 Playwright 從雅虎財經抓取指定股票的最新新聞。
    """
    print(f" - [新聞模塊] 正在為 {symbol} 抓取新聞...")
    news_items = []
    try:
        if market.upper() == 'HK' and not symbol.endswith('.HK'):
            symbol = f"{int(symbol):04d}.HK"

        url = f"https://finance.yahoo.com/quote/{symbol}/news"

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=60000) # 增加超時時間
            
            # 等待新聞列表加載
            page.wait_for_selector('li.news-stream-item', timeout=30000)

            script = """
            () => {
                const items = [];
                const newsList = document.querySelectorAll('li.news-stream-item');
                for (let i = 0; i < Math.min(newsList.length, 5); i++) {
                    const item = newsList[i];
                    const linkElement = item.querySelector('a');
                    const titleElement = item.querySelector('h3');
                    if (linkElement && titleElement) {
                        items.push({
                            title: titleElement.innerText,
                            link: linkElement.href
                        });
                    }
                }
                return items;
            }
            """
            news_items = page.evaluate(script)
            browser.close()
        
        print(f" - [新聞模塊] 成功為 {symbol} 抓取到 {len(news_items)} 條新聞。")
        return news_items

    except Exception as e:
        print(f" - [新聞模塊] 為 {symbol} 抓取新聞時出錯: {e}")
        return []

def analyze_news_sentiment(news_items: list[dict]) -> list[dict]:
    """
    利用大型語言模型分析新聞標題的情感。
    """
    for item in news_items:
        title = item['title'].lower()
        if any(keyword in title for keyword in ['profit warning', 'investigation', 'lawsuit', 'plunges', 'drops', 'falls']):
            item['sentiment'] = '利空'
            item['reason'] = '標題包含負面關鍵詞。'
        elif any(keyword in title for keyword in ['record profit', 'beats estimates', 'upgrades', 'surges', 'booming', 'rises']):
            item['sentiment'] = '利好'
            item['reason'] = '標題包含正面關鍵詞。'
        else:
            item['sentiment'] = '中性'
            item['reason'] = '標題未顯示明顯情感傾向。'
            
    return news_items

def get_and_analyze_news(symbol: str, market: str) -> list[dict]:
    """
    整合抓取和分析兩個步驟的總功能。
    """
    news_items = fetch_news_from_yahoo(symbol, market)
    if not news_items:
        return []
    
    analyzed_news = analyze_news_sentiment(news_items)
    return analyzed_news