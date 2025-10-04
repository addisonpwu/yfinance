
import yfinance as yf
import pandas as pd
import requests
import urllib3
import re

# --- 策略設定 ---

# 1. 股票清單獲取方式
def get_all_us_tickers():
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

# 手動股票清單 (如果不想用全美股，可以註解掉 get_all_us_tickers() 那行，並使用這個)
MANUAL_STOCK_LIST = [
    '2330.TW', '2454.TW', '2603.TW', '1301.TW', # 台股範例
    'NVDA', 'AAPL', 'MSFT', 'TSLA', 'META'      # 美股範例
]

# 2. 篩選條件參數
VOLUME_INCREASE_RATIO = 3.0  # 成交量放大倍數：當日成交量 > 過去20日平均成交量的 N 倍
PRICE_INCREASE_LOWER_BOUND = 2.0   # 當日漲幅百分比下限：收盤價漲幅 > N%
PRICE_INCREASE_UPPER_BOUND = 4.0   # 當日漲幅百分比上限：收盤價漲幅 < N%

def screen_momentum_stocks(stock_list):
    """
    根據「價漲量增」動能突破策略篩選股票
    """
    print(f"開始篩選 {len(stock_list)} 支股票...")
    qualified_stocks = []
    total_stocks = len(stock_list)

    for i, stock_symbol in enumerate(stock_list):
        try:
            # 顯示進度
            print(f"\n({i+1}/{total_stocks}) 正在分析: {stock_symbol}...", end='')
            
            # 下載近6個月的數據，以確保有足夠數據計算100日均線
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(period="6mo", auto_adjust=True) # auto_adjust=True 處理股利和拆分

            if hist.empty or len(hist) < 101: # 確保有足夠數據計算100MA
                # print(f"  - {stock_symbol} 歷史數據不足，跳過。") # 在大量掃描時可關閉此訊息
                continue

            # --- 計算技術指標 ---
            # 1. 計算移動平均線 (MA)
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA100'] = hist['Close'].rolling(window=100).mean()

            # 2. 計算過去20日的平均成交量 (不包含當日)
            hist['Avg_Volume_20'] = hist['Volume'].rolling(window=20).mean().shift(1)

            # 取得最近一天的數據
            latest_data = hist.iloc[-1]
            previous_data = hist.iloc[-2]

            # --- 開始進行條件篩選 ---
            
            # 條件一：中期趨勢向上 (股價高於50日或100日均線)
            is_trend_up = (latest_data['Close'] > latest_data['MA50']) or \
                          (latest_data['Close'] > latest_data['MA100'])
            
            if not is_trend_up:
                # print(f"  - {stock_symbol} 不滿足趨勢向上條件。") # 在大量掃描時可關閉此訊息
                continue

            # 條件二：成交量顯著放大 (當日成交量 > 20日均量的 N 倍)
            is_volume_surge = latest_data['Volume'] > (latest_data['Avg_Volume_20'] * VOLUME_INCREASE_RATIO)

            if not is_volume_surge:
                # print(f"  - {stock_symbol} 不滿足成交量放大條件。") # 在大量掃描時可關閉此訊息
                continue

            # 條件三：價格同步上漲 (紅K棒且漲幅在 2% ~ 4% 之間)
            is_red_candle = latest_data['Close'] > latest_data['Open']
            price_change_percent = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100
            is_moderate_price_up = (price_change_percent > PRICE_INCREASE_LOWER_BOUND) and (price_change_percent < PRICE_INCREASE_UPPER_BOUND)

            if not (is_red_candle and is_moderate_price_up):
                # print(f"  - {stock_symbol} 不滿足價格同步上漲條件。") # 在大量掃描時可關閉此訊息
                continue

            # --- 所有條件均滿足 ---
            print(f"  ✅ {stock_symbol} 符合所有條件！")
            print(f"     - 收盤價: {latest_data['Close']:.2f}")
            print(f"     - 漲幅: {price_change_percent:.2f}%")
            print(f"     - 成交量: {latest_data['Volume']:.0f} (放大倍數: {latest_data['Volume']/latest_data['Avg_Volume_20']:.2f}x)")
            
            # 獲取交易所資訊
            info = ticker.info
            exchange = info.get('exchange', 'UNKNOWN')
            qualified_stocks.append({'symbol': stock_symbol, 'exchange': exchange})

        except Exception as e:
            # 在大量掃描時，可以選擇忽略單一股票的錯誤
            # print(f"  - 分析 {stock_symbol} 時發生錯誤: {e}")
            pass

    return qualified_stocks

if __name__ == '__main__':
    # --- 決定要使用的股票清單 ---
    # 1. 自動獲取全美股列表 (預設，執行時間較長)
    stock_list_to_screen = get_all_us_tickers()
    
    # 2. 使用手動列表 (如果需要，請取消註解下面這行，並註解掉上面那行)
    # stock_list_to_screen = MANUAL_STOCK_LIST

    # 執行選股策略
    final_list = screen_momentum_stocks(stock_list_to_screen)

    print("\n--- 篩選結果 ---")
    if final_list:
        # yfinance 的交易所代碼與常見名稱的對照表
        exchange_map = {
            'NMS': 'NASDAQ',
            'NYQ': 'NYSE',
            'PCX': 'NYSE ARCA',
            'TAI': 'TWSE' # 台灣證券交易所
        }

        formatted_stocks = []
        for stock in final_list:
            # 使用對照表來取得交易所名稱，如果找不到就用原始代碼
            exchange_name = exchange_map.get(stock['exchange'], stock['exchange'])
            formatted_stocks.append(f"{exchange_name}:{stock['symbol']}")
        
        # 組合最終字串
        output_string = ','.join(formatted_stocks)

        # 將結果寫入 us.txt 檔案
        try:
            with open('us.txt', 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"已將 {len(formatted_stocks)} 支符合條件的股票輸出至 us.txt")
        except Exception as e:
            print(f"寫入檔案 us.txt 時發生錯誤: {e}")

    else:
        print("今日在指定的股票清單中，沒有找到符合條件的股票。")

    print("\n--- 操作邏輯提醒 ---")
    print("1. 進場時機：可在符合條件的隔天開盤時考慮進場，或在盤中觀察動能延續時介入。")
    print("2. 停損點：可設定在突破點下方，或放量長紅K棒的低點。")
    print("3. 注意事項：此策略旨在抓住市場動能，但需注意避免追高，並警惕股價高檔爆量後反轉的風險。")
