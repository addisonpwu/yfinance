
import argparse
import os
from datetime import datetime
from src.core.services.analysis_service import run_analysis

def main():
    parser = argparse.ArgumentParser(description="靈活的股票篩選器，支援多種策略")
    parser.add_argument('--market', type=str, required=True, choices=['US', 'HK'], help="要分析的市場 (US 或 HK)")
    parser.add_argument('--no-cache-update', action='store_true', help="跳過緩存更新，直接使用現有緩存數據")
    parser.add_argument('--skip-strategies', action='store_true', help="跳過策略篩選，所有股票都進行AI分析")
    parser.add_argument('--symbol', type=str, help="指定分析單一股票代碼（例如：0017.HK）")
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '1h', '1m'], help="數據時段類型：1d（日線，默認）、1h（小時線）、1m（分鐘線）")
    parser.add_argument('--model', type=str, default='deepseek-v3.2', choices=['iflow-rome-30ba3b', 'qwen3-max', 'tstars2.0', 'deepseek-v3.2', 'qwen3-coder-plus', 'all'], help="AI分析模型：iflow-rome-30ba3b/qwen3-max/tstars2.0/deepseek-v3.2/qwen3-coder-plus/all")
    args = parser.parse_args()

    print(f"--- 開始對 {args.market.upper()} 市場進行分析 ---")
    if args.no_cache_update:
        print(f"--- 已啟用快速模式：跳過緩存更新 ---")
    if args.skip_strategies:
        print(f"--- 已啟用跳過策略模式：所有股票都進行AI分析 ---")
    if args.symbol:
        print(f"--- 分析指定股票: {args.symbol} ---")
    print(f"--- 數據時段類型: {args.interval} ---")
    print(f"--- AI分析模型: {args.model} ---")

    # 生成报告文件名
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_filename = f"{args.market.lower()}_stocks_{today_str}.txt"

    # 初始化报告文件（写入标题）
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("--- 最終篩選結果 (詳細) ---\n")
        print(f"--- 報告文件已創建: {output_filename} ---")
    except Exception as e:
        print(f"創建報告文件時發生錯誤: {e}")
        output_filename = None

    final_list = run_analysis(
        args.market,
        force_fast_mode=args.no_cache_update,
        skip_strategies=args.skip_strategies,
        symbol_filter=args.symbol,
        interval=args.interval,
        model=args.model,
        output_filename=output_filename
    )

    print("\n--- 最終篩選結果 ---")
    if final_list and output_filename and os.path.exists(output_filename):
        # 读取已写入的内容
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            print(existing_content)
        except Exception as e:
            print(f"讀取報告文件時發生錯誤: {e}")

        # 追加摘要列表（实时报告只包含详细分析）
        exchange_map = {
            'NMS': 'NASDAQ',
            'NGM': 'NASDAQ',
            'NCM': 'NASDAQ',
            'NYQ': 'NYSE',
            'PCX': 'NYSE ARCA',
            'TAI': 'TWSE',
            'HKG': 'HKEX'
        }

        formatted_stocks = []
        for stock in final_list:
            info = stock.get('info', {})
            long_name = info.get('longName', stock['symbol'])
            exchange_name = exchange_map.get(stock['exchange'], stock['exchange'])
            symbol = stock['symbol']

            if args.market.upper() == 'HK':
                symbol = str(int(symbol.replace('.HK', '')))

            formatted_stocks.append(f"{exchange_name}:{symbol} ({long_name})")

        # 追加摘要列表到文件
        summary_lines = [
            "\n" + "="*50,
            "--- 摘要列表 (便於複製到交易軟體) ---",
            ", ".join(formatted_stocks)
        ]
        with open(output_filename, 'a', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))

        print("\n" + "="*50)
        print("--- 摘要列表 (便於複製到交易軟體) ---")
        print(", ".join(formatted_stocks))
        print(f"\n--- 完整報告已儲存至 {output_filename} ---")

    elif final_list:
        # 如果没有输出文件，只打印结果
        for stock in final_list:
            info = stock.get('info', {})
            print(f"✅ {info.get('longName', stock['symbol'])} ({stock['symbol']}) - {stock['strategies']}")
    else:
        print("在指定的市場中，沒有找到符合任何策略的股票。")

if __name__ == '__main__':
    main()
