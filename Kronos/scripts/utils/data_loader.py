import pandas as pd
import yfinance as yf
from datetime import date, timedelta

def load_hk_stock_data(ticker: str, lookback: int, interval: str) -> pd.DataFrame | None:
    """
    Downloads and preprocesses a specific amount of historical stock data.

    Args:
        ticker: The stock ticker symbol (e.g., "1810.HK").
        lookback: The number of data points (candlesticks) to retrieve.
        interval: The data interval (e.g., "1d", "1h", "5m").

    Returns:
        A preprocessed pandas DataFrame with exactly `lookback` rows, or None if an error occurs.
    """
    print(f"\n正在從 yfinance 下載 {ticker} 的最新數據 (Interval: {interval})...")

    # Determine the appropriate period to fetch based on interval to ensure enough data
    # yfinance has limitations on intraday data lookback periods
    period_to_fetch = "max" # Default to max and then trim
    if 'm' in interval or 'h' in interval:
        if interval == '1m':
            # 1m data is limited to the last 7 days
            period_to_fetch = "7d"
        elif interval in ['2m', '5m', '15m', '30m']:
            # these intervals are limited to the last 60 days
            period_to_fetch = "60d"
        elif interval in ['60m', '1h']:
            # these intervals are limited to the last 730 days
            period_to_fetch = "730d"

    try:
        tk = yf.Ticker(ticker)
        # Use period to fetch a chunk of data, then we will select the tail.
        data = tk.history(period=period_to_fetch, interval=interval, auto_adjust=False)

        if data.empty:
            print("錯誤：從 yfinance 下載的數據為空。")
            return None

        # Check if we have enough data
        if len(data) < lookback:
            print(f"錯誤：下載的K線數量 ({len(data)}) 少於要求的 LOOKBACK ({lookback})。任務中止。")
            print("提示：對於分鐘線(1m)，yfinance 最多提供7天數據；對於其他分鐘線(<=30m)，最多提供60天數據。請嘗試縮小 LOOKBACK 或更換時間頻率(INTERVAL)。")
            return None

        # Get the most recent `lookback` candles and reset index
        df = data.iloc[-lookback:].copy()
        df.reset_index(inplace=True)

    except Exception as e:
        print(f"下載數據時發生錯誤: {e}")
        return None


    # Data cleaning and preparation
    # The column name for timestamp varies with interval ('Date' for daily, 'Datetime' for intraday)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'timestamps'}, inplace=True)
    elif 'Date' in df.columns:
        df.rename(columns={'Date': 'timestamps'}, inplace=True)
    else:
        print("錯誤：找不到 'Date' 或 'Datetime' 欄位。")
        return None

    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    # Drop columns that are not needed and might be added by yfinance
    for col_to_drop in ['Dividends', 'Stock Splits']:
        if col_to_drop in df.columns:
            df.drop(columns=[col_to_drop], inplace=True)

    df['amount'] = df[['open', 'high', 'low', 'close']].mean(axis=1) * df['volume']

    required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
    df = df[required_cols]

    print(f"數據下載與預處理完成，共獲取 {len(df)} 條K線。")
    return df