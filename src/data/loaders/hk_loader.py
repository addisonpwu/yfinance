
import pandas as pd
import requests
import io
import urllib3

def get_hk_tickers():
    """
    從香港交易所網站下載最新的證券列表Excel，並篩選出以港幣交易的股本證券。
    """
    print("\n正在從香港交易所(hkex.com.hk)下載證券列表...")
    try:
        url = "https://www.hkex.com.hk/chi/services/trading/securities/securitieslists/ListOfSecurities_c.xlsx"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()
        excel_file = io.BytesIO(response.content)
        df = pd.read_excel(excel_file, sheet_name="ListOfSecurities", header=2)

        print("讀取成功，開始篩選港幣交易的股本證券...")
        df_equity = df[df['分類'] == '股本']
        df_hkd = df_equity[df_equity['交易貨幣'] == 'HKD']

        if df_hkd.empty:
            raise ValueError("在Excel中找不到任何以HKD交易的股本證券")

        tickers = df_hkd['股份代號'].astype(int).astype(str).str.zfill(4) + '.HK'
        ticker_list = tickers.dropna().tolist()
        print(f"成功篩選出 {len(ticker_list)} 支港股。")
        return ticker_list

    except Exception as e:
        print(f"無法獲取港股列表，錯誤: {e}")
        return []
