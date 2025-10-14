import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor
from config.config_hk import *
from utils.data_loader import load_hk_stock_data

def plot_prediction(kline_df, pred_df, ticker, rise_prob):
    """
    繪製歷史數據與機率性預測結果的圖表，並自動儲存。
    """
    # --- 數據準備 ---
    # 篩選收盤價和成交量的預測列
    # 在 sample_count > 1 的情況下，返回的欄位名會是 close_0, close_1...
    close_pred_cols = [c for c in pred_df.columns if 'close' in str(c)]
    volume_pred_cols = [c for c in pred_df.columns if 'volume' in str(c)]

    # 如果沒有找到帶後綴的欄位，則假定 sample_count=1
    if not close_pred_cols:
        close_pred_cols = ['close']
    if not volume_pred_cols:
        volume_pred_cols = ['volume']

    # 計算預測的均值、最小值、最大值
    mean_close_preds = pred_df[close_pred_cols].mean(axis=1)
    min_close_preds = pred_df[close_pred_cols].min(axis=1)
    max_close_preds = pred_df[close_pred_cols].max(axis=1)
    mean_volume_preds = pred_df[volume_pred_cols].mean(axis=1)

    # 獲取最後的歷史日期和預測時間範圍
    last_hist_date = kline_df['timestamps'].iloc[-1]
    # 修正預測時間的生成邏輯，使其更穩健
    time_diff = kline_df['timestamps'].diff().mean()
    pred_time = pd.date_range(start=last_hist_date + time_diff, periods=len(pred_df), freq=time_diff)


    # --- 開始繪圖 ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(18, 10), 
        sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(f'{ticker} Probabilistic Price & Volume Forecast (Next {PRED_LEN} candles)', fontsize=18)

    # --- 繪製價格圖 (ax1) ---
    # 歷史價格
    ax1.plot(kline_df['timestamps'], kline_df['close'], color='royalblue', label='Historical Price')
    # 預測平均價格
    ax1.plot(pred_time, mean_close_preds, color='darkorange', linestyle='--', label='Mean Forecast')
    # 預測範圍
    ax1.fill_between(pred_time, min_close_preds, max_close_preds, color='sandybrown', alpha=0.3, label='Forecast Range (Min-Max)')
    
    # --- 新增：標註上升機率 ---
    prob_text = f"Rise Probability: {rise_prob:.1f}%"
    ax1.text(0.02, 0.95, prob_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='mediumseagreen', alpha=0.5))

    ax1.set_ylabel('Price (HKD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- 繪製成交量圖 (ax2) ---
    # 歷史成交量
    ax2.bar(kline_df['timestamps'], kline_df['volume'], width=0.8, color='skyblue', label='Historical Volume')
    # 預測平均成交量
    ax2.bar(pred_time, mean_volume_preds, width=0.8, color='sandybrown', alpha=0.7, label='Mean Forecasted Volume')

    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 在兩個子圖上都畫出垂直分割線
    for ax in [ax1, ax2]:
        ax.axvline(x=last_hist_date, color='red', linestyle='--')

    # 整合圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')


    # 自動格式化日期顯示
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # --- 新增：儲存圖表 ---
    # 獲取專案根目錄
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'figures', 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    # 產生唯一檔案名稱
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 清理 ticker 中的非法字元
    safe_ticker = ticker.replace('.', '_')
    filename = f"{safe_ticker}_{timestamp_str}.png"
    filepath = os.path.join(save_dir, filename)
    
    # 儲存圖檔
    plt.savefig(filepath, dpi=120)
    print(f"\n圖表已儲存至: {filepath}")

    # 顯示圖表
    plt.show()


def main():
    print(f"--- Kronos: 預測 {TICKER} 未來 {PRED_LEN} 根K線走勢 ---")

    # 1. 下載並準備歷史數據
    # The new loader function ensures we get exactly LOOKBACK candles, or None if data is insufficient.
    df = load_hk_stock_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL)

    if df is None:
        print("獲取數據失敗，腳本終止。")
        return

    # 2. 準備預測模型的輸入
    # The dataframe `df` now contains exactly the `LOOKBACK` number of rows required.
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    x_df = df[required_cols]
    x_timestamp = df['timestamps']
    print(f"使用 {LOOKBACK} 條 ({INTERVAL} 頻率) K線作為模型輸入。")

    # 3. 產生未來 K 線的時間戳
    last_timestamp = df['timestamps'].iloc[-1]
    # Use the interval from the data to ensure correct future timestamps
    time_diff = df['timestamps'].diff().mean()
    y_timestamp = pd.date_range(start=last_timestamp + time_diff, periods=PRED_LEN, freq=time_diff)
    y_timestamp = pd.Series(y_timestamp)

    # 4. 載入模型並開始預測
    print(f"載入模型: {MODEL_ID}...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=CONTEXT_LENGTH)

    sample_count_for_prob = 1
    print(f"開始進行 {PRED_LEN} 根K線的機率性預測 ({sample_count_for_prob}次採樣)...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.95,
        sample_count=sample_count_for_prob,
        verbose=True,
        return_samples=True
    )

    # 5. 計算並顯示上升/下跌機率
    print("\n計算上升/下跌機率...")
    last_hist_close = x_df['close'].iloc[-1]
    close_pred_cols = [c for c in pred_df.columns if 'close' in str(c)]
    if not close_pred_cols:
        close_pred_cols = ['close']
    
    last_day_preds = pred_df[close_pred_cols].iloc[-1]
    num_samples = len(last_day_preds)
    num_rise = (last_day_preds > last_hist_close).sum()
    rise_prob = (num_rise / num_samples) * 100

    print(f"基於 {num_samples} 次採樣:")
    print(f"預測期結束時，價格上升機率: {rise_prob:.2f}%")
    print(f"預測期結束時，價格下跌機率: {100 - rise_prob:.2f}%")

    # 6. 視覺化結果
    print("\n預測完成，正在生成圖表...")
    # For plotting, we use the historical data we just downloaded.
    kline_df = df.iloc[-252:]
    plot_prediction(kline_df, pred_df, TICKER, rise_prob)


if __name__ == "__main__":
    main()