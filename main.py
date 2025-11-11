
import argparse
from datetime import datetime
from analysis import analyzer
from news_analyzer import get_and_analyze_news

def main():
    parser = argparse.ArgumentParser(description="靈活的股票篩選器，支援多種策略")
    parser.add_argument('--market', type=str, required=True, choices=['US', 'HK'], help="要分析的市場 (US 或 HK)")
    args = parser.parse_args()

    print(f"--- 開始對 {args.market.upper()} 市場進行分析 ---")
    final_list = analyzer.run_analysis(args.market)

    print("\n--- 開始進行新聞情感分析 ---")
    for stock in final_list:
        # 為每支股票獲取並分析新聞
        stock['analyzed_news'] = get_and_analyze_news(stock['symbol'], args.market)

    print("\n--- 最終篩選結果 (已包含新聞分析) ---")
    if final_list:
        today_str = datetime.now().strftime('%Y-%m-%d')

        # --- 產生並儲存詳細報告 (已移除新聞) ---
        detailed_output_filename = f"{args.market.lower()}_stocks_{today_str}_details.txt"
        detailed_output_lines = ["--- 最終篩選結果 (詳細) ---"]

        for stock in final_list:
            info = stock.get('info', {})
            kronos_prediction = stock.get('kronos_prediction', 'N/A') # <-- 獲取預測值
            
            # 安全地格式化市值和PE
            market_cap = info.get('marketCap')
            market_cap_str = f"{market_cap / 1e8:.2f} 億" if isinstance(market_cap, (int, float)) else "N/A"

            pe_ratio = info.get('trailingPE')
            pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

            float_shares = info.get('floatShares')
            float_shares_str = f"{float_shares:,.0f}" if isinstance(float_shares, (int, float)) else "N/A"

            volume = info.get('volume')
            volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "N/A"

            detailed_output_lines.append(f"\n✅ {info.get('longName', stock['symbol'])} ({stock['symbol']})")
            detailed_output_lines.append(f"   - 符合策略: {stock['strategies']}")
            detailed_output_lines.append(f"   - Kronos 預測: {kronos_prediction}") # <-- 新增此行以顯示預測
            detailed_output_lines.append(f"   - 產業: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}")
            detailed_output_lines.append(f"   - 市值: {market_cap_str}")
            detailed_output_lines.append(f"   - 流通股本: {float_shares_str}")
            detailed_output_lines.append(f"   - 成交量: {volume_str}")
            detailed_output_lines.append(f"   - 市盈率 (PE): {pe_ratio_str}")
            detailed_output_lines.append(f"   - 網站: {info.get('website', 'N/A')}")

            # --- 新增新聞分析結果的輸出 ---
            if stock.get('analyzed_news'):
                detailed_output_lines.append("   --- 最新新聞分析 ---")
                for news in stock['analyzed_news']:
                    sentiment_icon = {'利好': '🟢', '利空': '🔴', '中性': '⚪️'}.get(news['sentiment'], '⚪️')
                    detailed_output_lines.append(f"     {sentiment_icon} [{news['sentiment']}] {news['title']}")
                    # detailed_output_lines.append(f"        理由: {news['reason']}") # 可以選擇性加入理由
                    detailed_output_lines.append(f"        連結: {news['link']}")
            else:
                detailed_output_lines.append("   --- 未找到相關新聞 ---")
        
        detailed_output_string = "\n".join(detailed_output_lines)
        print(detailed_output_string)

        try:
            with open(detailed_output_filename, 'w', encoding='utf-8') as f:
                f.write(detailed_output_string)
            print(f"\n--- 詳細報告已儲存至 {detailed_output_filename} ---")
        except Exception as e:
            print(f"寫入詳細報告 {detailed_output_filename} 時發生錯誤: {e}")

        # --- 產生並儲存新的摘要列表 (包含股票名稱) ---
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
            info = stock.get('info', {})
            long_name = info.get('longName', stock['symbol'])
            exchange_name = exchange_map.get(stock['exchange'], stock['exchange'])
            symbol = stock['symbol']
            
            if args.market.upper() == 'HK':
                symbol = str(int(symbol.replace('.HK', '')))

            formatted_stocks.append(f"{exchange_name}:{symbol} ({long_name})")
        
        output_string = ", ".join(formatted_stocks) # 使用 ", " 增加可讀性

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
