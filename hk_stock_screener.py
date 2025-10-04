
import yfinance as yf
import pandas as pd
import re
import requests
import io
import urllib3

# --- 港股專用設定 ---

def get_hk_stock_tickers():
    """
    從香港交易所網站下載最新的證券列表Excel，並篩選出以港幣交易的股本證券。
    """
    print("正在從香港交易所(hkex.com.hk)下載證券列表...")
    try:
        url = "https://www.hkex.com.hk/chi/services/trading/securities/securitieslists/ListOfSecurities_c.xlsx"
        
        # 繞過 SSL 驗證下載檔案
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status() # 確保下載成功

        # 將下載的二進位內容轉換為 pandas 可讀取的 file-like object
        excel_file = io.BytesIO(response.content)
        
        # 使用 pandas 從記憶體中的檔案內容讀取 Excel
        # 需要安裝 openpyxl: pip install openpyxl
        df = pd.read_excel(excel_file, sheet_name="ListOfSecurities", header=2)

        print("讀取成功，開始篩選港幣交易的股本證券...")

        # 篩選條件 1: 分類 == "股本"
        df_equity = df[df['分類'] == '股本']
        # 篩選條件 2: 交易貨幣 == "HKD"
        df_hkd = df_equity[df_equity['交易貨幣'] == 'HKD']

        if df_hkd.empty:
            raise ValueError("在Excel中找不到任何以HKD交易的股本證券")

        # 提取股份代號，並轉換為 yfinance 格式 (例如： 700 -> "0700.HK")
        # 先將代號轉為整數，再轉為字串，避免 '2800' 變 '2800.0' 的問題
        # 根據使用者提供規則，將代碼補足4位數 (e.g. 700 -> 0700)
        tickers = df_hkd['股份代號'].astype(int).astype(str).str.zfill(4) + '.HK'
        
        # 轉換為列表並移除任何可能的空值
        ticker_list = tickers.dropna().tolist()

        print(f"成功篩選出 {len(ticker_list)} 支港股。")
        return ticker_list

    except Exception as e:
        print(f"無法獲取港股列表，錯誤: {e}")
        print("將使用預設的少量股票列表進行分析。")
        return ['0700.HK', '9988.HK', '0005.HK'] # 備用列表

# --- 策略設定 ---

# 篩選條件參數 (與美股腳本相同)
VOLUME_INCREASE_RATIO = 2.0  # 成交量放大倍數
PRICE_INCREASE_LOWER_BOUND = 2.0   # 漲幅下限
PRICE_INCREASE_UPPER_BOUND = 4.0   # 漲幅上限

def screen_momentum_stocks(stock_list):
    """
    根據「價漲量增」動能突破策略篩選股票 (邏輯與美股腳本完全相同)
    """
    print(f"\n開始篩選 {len(stock_list)} 支股票...")
    qualified_stocks = []
    total_stocks = len(stock_list)

    for i, stock_symbol in enumerate(stock_list):
        try:
            print(f"\n({i+1}/{total_stocks}) 正在分析: {stock_symbol}...", end='')
            
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(period="6mo", auto_adjust=True)

            if hist.empty or len(hist) < 101:
                continue

            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA100'] = hist['Close'].rolling(window=100).mean()
            hist['Avg_Volume_20'] = hist['Volume'].rolling(window=20).mean().shift(1)

            latest_data = hist.iloc[-1]
            previous_data = hist.iloc[-2]

            is_trend_up = (latest_data['Close'] > latest_data['MA50']) or \
                          (latest_data['Close'] > latest_data['MA100'])
            
            if not is_trend_up:
                continue

            # 檢查 Avg_Volume_20 是否存在且不為 NaN 或 0
            if pd.isna(latest_data['Avg_Volume_20']) or latest_data['Avg_Volume_20'] == 0:
                continue

            is_volume_surge = latest_data['Volume'] > (latest_data['Avg_Volume_20'] * VOLUME_INCREASE_RATIO)

            if not is_volume_surge:
                continue

            is_red_candle = latest_data['Close'] > latest_data['Open']
            price_change_percent = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100
            is_moderate_price_up = (price_change_percent > PRICE_INCREASE_LOWER_BOUND) and (price_change_percent < PRICE_INCREASE_UPPER_BOUND)

            if not (is_red_candle and is_moderate_price_up):
                continue

            print(f"  ✅ {stock_symbol} 符合所有條件！")
            info = ticker.info
            exchange = info.get('exchange', 'HKEX') # 預設為 HKEX
            qualified_stocks.append({'symbol': stock_symbol, 'exchange': exchange})

        except Exception as e:
            pass

    return qualified_stocks

if __name__ == '__main__':
    stock_list_to_screen = get_hk_stock_tickers()

    if stock_list_to_screen:
        final_list = screen_momentum_stocks(stock_list_to_screen)

        print("\n--- 篩選結果 ---")
        if final_list:
            formatted_stocks = []
            for stock in final_list:
                # 移除 .HK 後綴
                code_without_suffix = stock['symbol'].replace('.HK', '')
                # 轉換為整數再轉回字串，以移除前面的補零 (例如 '0700' -> 700 -> '700')
                numeric_code = str(int(code_without_suffix))
                # 組合成最終格式
                formatted_stocks.append(f"HKEX:{numeric_code}")

            output_string = ','.join(formatted_stocks)
            
            try:
                with open('hk.txt', 'w', encoding='utf-8') as f:
                    f.write(output_string)
                print(f"已將 {len(final_list)} 支符合條件的股票輸出至 hk.txt")
            except Exception as e:
                print(f"寫入檔案 hk.txt 時發生錯誤: {e}")
        else:
            print("今日在指定的股票清單中，沒有找到符合條件的股票ảng。")

