import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import numpy as np
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor
from config.config_hk_iterative import *
from utils.data_loader import load_hk_stock_data

def plot_iterative_prediction(hist_df, all_preds_df, actual_df, ticker):
    """
    繪製歷史數據、迭代預測結果以及未來真實數據的圖表。
    """
    # --- 數據準備 ---
    # 計算預測的均值、最小值、最大值
    mean_close_preds = all_preds_df[[c for c in all_preds_df.columns if 'close' in str(c)]].mean(axis=1)
    min_close_preds = all_preds_df[[c for c in all_preds_df.columns if 'close' in str(c)]].min(axis=1)
    max_close_preds = all_preds_df[[c for c in all_preds_df.columns if 'close' in str(c)]].max(axis=1)
    mean_volume_preds = all_preds_df[[c for c in all_preds_df.columns if 'volume' in str(c)]].mean(axis=1)

    # --- 開始繪圖 ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(f'{ticker} Iterative Forecast (Predict {TOTAL_PRED_LEN} candles in {TOTAL_PRED_LEN // SHORT_STEP} steps)', fontsize=18)

    # --- 繪製價格圖 (ax1) ---
    # 歷史價格
    ax1.plot(hist_df['timestamps'], hist_df['close'], color='royalblue', label='Historical Price')
    # 預測平均價格
    ax1.plot(all_preds_df.index, mean_close_preds, color='darkorange', linestyle='--', label='Mean Iterative Forecast')
    # 預測範圍
    ax1.fill_between(all_preds_df.index, min_close_preds, max_close_preds, color='sandybrown', alpha=0.3, label='Forecast Range (Min-Max)')
    # 未來真實價格
    ax1.plot(actual_df['timestamps'], actual_df['close'], color='green', linestyle=':', label='Actual Future Price')

    ax1.set_ylabel('Price (HKD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 繪製成交量圖 (ax2) ---
    # 歷史成交量
    ax2.bar(hist_df['timestamps'], hist_df['volume'], width=0.8, color='skyblue', label='Historical Volume')
    # 預測平均成交量
    ax2.bar(all_preds_df.index, mean_volume_preds, width=0.8, color='sandybrown', alpha=0.7, label='Mean Forecasted Volume')
    # 未來真實成交量
    ax2.bar(actual_df['timestamps'], actual_df['volume'], width=0.8, color='lightgreen', alpha=0.7, label='Actual Future Volume')

    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 在兩個子圖上都畫出垂直分割線
    last_hist_date = hist_df['timestamps'].iloc[-1]
    for ax in [ax1, ax2]:
        ax.axvline(x=last_hist_date, color='red', linestyle='--')

    # 整合圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # 自動格式化日期顯示
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # --- 儲存圖表 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'figures', 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace('.', '_')
    filename = f"{safe_ticker}_iterative_{timestamp_str}.png"
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath, dpi=120)
    print(f"\n圖表已儲存至: {filepath}")
    plt.show()


def main():
    print(f"--- Kronos Iterative Prediction: Forecast {TICKER} for {TOTAL_PRED_LEN} candles ---")

    # 1. 下載足夠的歷史數據 (LOOKBACK + TOTAL_PRED_LEN)
    total_required_data = LOOKBACK + TOTAL_PRED_LEN
    full_df = load_hk_stock_data(ticker=TICKER, lookback=total_required_data, interval=INTERVAL)

    if full_df is None or len(full_df) < total_required_data:
        print(f"獲取數據失敗或數據不足，需要 {total_required_data} 條數據，實際獲取 {len(full_df) if full_df is not None else 0} 條。腳本終止。")
        return

    # 2. 劃分初始歷史數據和未來真實數據
    initial_hist_df = full_df.iloc[:LOOKBACK]
    actual_future_df = full_df.iloc[LOOKBACK:]

    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    current_x_df = initial_hist_df[required_cols].copy()
    
    print(f"使用最新的 {LOOKBACK} 條 ({INTERVAL} 頻率) K線作為初始模型輸入。")
    print(f"將預測未來 {TOTAL_PRED_LEN} 條K線，並與真實數據進行比較。")

    # 3. 載入模型
    print(f"載入模型: {MODEL_ID}...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=CONTEXT_LENGTH)

    # 4. 迭代預測
    # 使用向上取整的除法，確保能覆蓋整個預測長度
    num_iterations = (TOTAL_PRED_LEN + SHORT_STEP - 1) // SHORT_STEP
    all_predictions_list = []

    print(f"\n開始進行迭代預測，共 {num_iterations} 步，每步預測 {SHORT_STEP} 根K線...")
    
    progress_bar = trange(num_iterations, desc="Iterative Prediction")
    for i in progress_bar:
        # 準備當前迭代的輸入
        x_timestamp = full_df.iloc[i*SHORT_STEP : LOOKBACK + i*SHORT_STEP]['timestamps']
        
        # 產生未來 K 線的時間戳
        last_timestamp = x_timestamp.iloc[-1]
        time_diff = x_timestamp.diff().mean()
        y_timestamp = pd.date_range(start=last_timestamp + time_diff, periods=SHORT_STEP, freq=time_diff)
        y_timestamp = pd.Series(y_timestamp)

        # 進行短步長預測
        step_pred_df = predictor.predict(
            df=current_x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=SHORT_STEP,
            T=1.0,
            top_p=0.95,
            sample_count=SAMPLE_COUNT,
            verbose=False, # 在循環中關閉詳細輸出
            return_samples=True
        )
        all_predictions_list.append(step_pred_df)

        # 模擬數據校正：將下一步的真實數據加入到歷史數據中
        # 從 actual_future_df 中獲取新的真實數據
        start_idx = i * SHORT_STEP
        end_idx = (i + 1) * SHORT_STEP
        new_real_data_df = actual_future_df.iloc[start_idx:end_idx][required_cols]

        # 更新 current_x_df
        current_x_df = pd.concat([current_x_df, new_real_data_df]).iloc[SHORT_STEP:]
        
        progress_bar.set_description(f"Step {i+1}/{num_iterations} Done")

    # 5. 整理並合併所有預測結果
    final_pred_df = pd.concat(all_predictions_list)
    final_pred_df.sort_index(inplace=True)
    print("\n迭代預測完成。")

    # 6. 計算最終的上升/下跌機率
    last_hist_close = initial_hist_df['close'].iloc[-1]
    close_pred_cols = [c for c in final_pred_df.columns if 'close' in str(c)]
    
    last_day_preds = final_pred_df[close_pred_cols].iloc[-1]
    num_samples = len(last_day_preds)
    num_rise = (last_day_preds > last_hist_close).sum()
    rise_prob = (num_rise / num_samples) * 100

    print(f"\n基於 {num_samples} 次採樣:")
    print(f"預測期結束時，價格相對於初始點的上升機率: {rise_prob:.2f}%")

    # 7. 視覺化結果
    print("正在生成圖表...")
    plot_iterative_prediction(initial_hist_df.iloc[-252:], final_pred_df, actual_future_df, TICKER)


if __name__ == "__main__":
    main()
