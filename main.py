
import argparse
from analysis import analyzer

def main():
    parser = argparse.ArgumentParser(description="靈活的股票篩選器，支援多種策略")
    parser.add_argument('--market', type=str, required=True, choices=['US', 'HK'], help="要分析的市場 (US 或 HK)")
    args = parser.parse_args()

    print(f"--- 開始對 {args.market.upper()} 市場進行分析 ---")
    final_list = analyzer.run_analysis(args.market)

    print("\n--- 最終篩選結果 ---")
    if final_list:
        output_filename = f"{args.market.lower()}_stocks.txt"
        
        # 交易所代碼對照表
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
            
            # 格式化港股代碼 (移除 .HK 並補零)
            if args.market.upper() == 'HK':
                symbol = str(int(symbol.replace('.HK', '')))

            formatted_stocks.append(f"{exchange_name}:{symbol}")
        
        output_string = ",".join(formatted_stocks)

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"已將 {len(final_list)} 支符合條件的股票輸出至 {output_filename}")
            print("\n--- 符合條件的股票列表 ---")
            print(output_string)
        except Exception as e:
            print(f"寫入檔案 {output_filename} 時發生錯誤: {e}")

    else:
        print("在指定的市場中，沒有找到符合任何策略的股票。")

if __name__ == '__main__':
    main()
