
import argparse
from datetime import datetime
from analysis import analyzer

def main():
    parser = argparse.ArgumentParser(description="靈活的股票篩選器，支援多種策略")
    parser.add_argument('--market', type=str, required=True, choices=['US', 'HK'], help="要分析的市場 (US 或 HK)")
    args = parser.parse_args()

    print(f"--- 開始對 {args.market.upper()} 市場進行分析 ---")
    final_list = analyzer.run_analysis(args.market)

    print("\n--- 最終篩選結果 ---")
    if final_list:
        today_str = datetime.now().strftime('%Y-%m-%d')

        # --- 產生並儲存詳細報告 ---
        detailed_output_filename = f"{args.market.lower()}_stocks_{today_str}_details.txt"
        detailed_output_lines = ["--- 最終篩選結果 (詳細) ---"]

        for stock in final_list:
            info = stock.get('info', {})
            market_cap_str = f"{info.get('marketCap', 0) / 1e8:.2f} 億" if info.get('marketCap') else "N/A"

            detailed_output_lines.append(f"\n✅ {info.get('longName', stock['symbol'])} ({stock['symbol']})")
            detailed_output_lines.append(f"   - 符合策略: {stock['strategies']}")
            detailed_output_lines.append(f"   - 產業: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}")
            detailed_output_lines.append(f"   - 市值: {market_cap_str}")
            detailed_output_lines.append(f"   - 市盈率 (PE): {info.get('trailingPE', 'N/A'):.2f}")
            detailed_output_lines.append(f"   - 網站: {info.get('website', 'N/A')}")
        
        detailed_output_string = "\n".join(detailed_output_lines)
        
        print(detailed_output_string) # 仍在控制台打印詳細報告

        try:
            with open(detailed_output_filename, 'w', encoding='utf-8') as f:
                f.write(detailed_output_string)
            print(f"\n--- 詳細報告已儲存至 {detailed_output_filename} ---")
        except Exception as e:
            print(f"寫入詳細報告 {detailed_output_filename} 時發生錯誤: {e}")

        # --- 產生並儲存原有的簡潔列表 ---
        output_filename = f"{args.market.lower()}_stocks_{today_str}.txt"
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
            exchange_name = exchange_map.get(stock['exchange'], stock['exchange'])
            symbol = stock['symbol']
            
            if args.market.upper() == 'HK':
                symbol = str(int(symbol.replace('.HK', '')))

            formatted_stocks.append(f"{exchange_name}:{symbol}")
        
        output_string = ",".join(formatted_stocks)

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"\n--- 摘要列表 (已儲存至 {output_filename}) ---")
            print(output_string)
        except Exception as e:
            print(f"寫入檔案 {output_filename} 時發生錯誤: {e}")

    else:
        print("在指定的市場中，沒有找到符合任何策略的股票。")

if __name__ == '__main__':
    main()
