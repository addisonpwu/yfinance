"""測試成交量異常放大策略"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
from src.strategies.volume_anomaly_strategy import VolumeAnomalyStrategy
from src.core.strategies.strategy import StrategyContext

def test_volume_anomaly_strategy():
    print("測試成交量異常放大策略...")
    
    # 獲取測試股票數據
    symbol = "AAPL"  # 可以更換為其他股票
    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")  # 獲取6個月數據
    
    if len(hist) < 20:
        print(f"數據不足，僅有 {len(hist)} 天數據")
        return
    
    print(f"獲取到 {len(hist)} 天 {symbol} 的歷史數據")
    
    # 獲取股票基本信息
    info = stock.info
    
    # 創建策略上下文
    context = StrategyContext(
        hist=hist,
        info=info,
        market_return=0.01,  # 假設市場回報率
        is_market_healthy=True
    )
    
    # 創建策略實例
    strategy = VolumeAnomalyStrategy()
    
    # 運行策略
    result = strategy.execute(context)
    
    print(f"策略名稱: {strategy.name}")
    print(f"策略結果: {'符合' if result.passed else '不符合'}")
    print(f"策略置信度: {result.confidence}")
    
    # 顯示一些關鍵指標
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
    
    avg_volume_20 = hist['Volume'].rolling(window=20).mean().iloc[-2] if len(hist) > 20 else hist['Volume'].mean()
    volume_multiplier = latest['Volume'] / avg_volume_20 if avg_volume_20 > 0 else 0
    
    true_range_pct = (latest['High'] - latest['Low']) / prev['Close'] if prev['Close'] > 0 else 0
    close_change_pct = abs((latest['Close'] - prev['Close']) / prev['Close']) if prev['Close'] > 0 else 0
    
    print(f"當日成交量: {latest['Volume']:,}")
    print(f"過去20日平均成交量: {avg_volume_20:,.0f}")
    print(f"成交量放大倍數: {volume_multiplier:.2f}")
    print(f"當日振幅: {true_range_pct:.2%}")
    print(f"收盤價波動: {close_change_pct:.2%}")
    
    # 測試配置參數的功能
    print("\n測試配置參數功能...")
    custom_strategy = VolumeAnomalyStrategy({
        'volume_multiplier_threshold': 2.0,  # 降低閾值測試
        'volatility_threshold': 0.06,        # 提高閾值測試
        'rsi_lower': 30,
        'rsi_upper': 65
    })
    
    custom_result = custom_strategy.execute(context)
    print(f"使用自定義參數的策略結果: {'符合' if custom_result.passed else '不符合'}")
    print(f"自定義策略置信度: {custom_result.confidence}")
    print(f"自定義策略名稱: {custom_strategy.name}")

if __name__ == "__main__":
    test_volume_anomaly_strategy()
