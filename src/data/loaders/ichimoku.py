"""
Ichimoku 雲圖計算模組

計算 Ichimoku Cloud 指標：
- Tenkan-sen (轉換線) = (9日高 + 9日低) / 2
- Kijun-sen (基準線) = (26日高 + 26日低) / 2
- Senkou Span A (先行上線) = (Tenkan + Kijun) / 2，向前投射26日
- Senkou Span B (先行下線) = (52日高 + 52日低) / 2，向前投射26日
- Chikou Span (延遲線) = 收盤價，向後投射26日

作者: iFlow CLI Team
日期: 2026-03-18
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# Ichimoku 參數
ICHIMOKU_TENKAN_PERIOD = 9   # 轉換線週期
ICHIMOKU_KIJUN_PERIOD = 26   # 基準線週期
ICHIMOKU_SPAN_B_PERIOD = 52  # 遲行帶週期
ICHIMOKU_DISPLACEMENT = 26   # 投射週期


def calculate_ichimoku(hist: pd.DataFrame) -> pd.DataFrame:
    """
    計算 Ichimoku 雲圖指標
    
    Args:
        hist: 包含 OHLC 數據的 DataFrame
        
    Returns:
        添加了 Ichimoku 指標的 DataFrame
    """
    if hist is None or hist.empty:
        return hist
    
    result = hist.copy()
    
    # Tenkan-sen (轉換線)
    tenkan_high = result['High'].rolling(window=ICHIMOKU_TENKAN_PERIOD, min_periods=1).max()
    tenkan_low = result['Low'].rolling(window=ICHIMOKU_TENKAN_PERIOD, min_periods=1).min()
    result['ICHIMOKU_Tenkan'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (基準線)
    kijun_high = result['High'].rolling(window=ICHIMOKU_KIJUN_PERIOD, min_periods=1).max()
    kijun_low = result['Low'].rolling(window=ICHIMOKU_KIJUN_PERIOD, min_periods=1).min()
    result['ICHIMOKU_Kijun'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (先行上線)
    # 投射到未來 ICHIMOKU_DISPLACEMENT 天
    span_a = ((result['ICHIMOKU_Tenkan'] + result['ICHIMOKU_Kijun']) / 2).shift(ICHIMOKU_DISPLACEMENT)
    result['ICHIMOKU_SenkouA'] = span_a
    
    # Senkou Span B (先行下線)
    span_b_high = result['High'].rolling(window=ICHIMOKU_SPAN_B_PERIOD, min_periods=1).max()
    span_b_low = result['Low'].rolling(window=ICHIMOKU_SPAN_B_PERIOD, min_periods=1).min()
    span_b = ((span_b_high + span_b_low) / 2).shift(ICHIMOKU_DISPLACEMENT)
    result['ICHIMOKU_SenkouB'] = span_b
    
    # Chikou Span (遲行線)
    # 向後投射 ICHIMOKU_DISPLACEMENT 天
    result['ICHIMOKU_Chikou'] = result['Close'].shift(-ICHIMOKU_DISPLACEMENT)
    
    # 雲帶顏色（用於可視化）
    # Span A > Span B 為綠色（多頭），否則為紅色（空頭）
    result['ICHIMOKU_CloudGreen'] = result['ICHIMOKU_SenkouA'] > result['ICHIMOKU_SenkouB']
    
    return result


def get_ichimoku_signals(hist: pd.DataFrame) -> Dict:
    """
    獲取 Ichimoku 雲圖信號
    
    Args:
        hist: 包含 Ichimoku 指標的 DataFrame
        
    Returns:
        Dict: 包含 Ichimoku 信號分析
    """
    if hist is None or hist.empty:
        return {'error': '數據不足'}
    
    # 確保有 Ichimoku 指標
    required_cols = ['ICHIMOKU_Tenkan', 'ICHIMOKU_Kijun', 'ICHIMOKU_SenkouA', 'ICHIMOKU_SenkouB']
    if not all(col in hist.columns for col in required_cols):
        hist = calculate_ichimoku(hist)
    
    # 獲取最新數據
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest
    
    current_price = latest['Close']
    tenkan = latest['ICHIMOKU_Tenkan']
    kijun = latest['ICHIMOKU_Kijun']
    senkou_a = latest['ICHIMOKU_SenkouA']
    senkou_b = latest['ICHIMOKU_SenkouB']
    
    signals = []
    
    # 1. Tenkan-Kijun 交叉信號
    if pd.notna(tenkan) and pd.notna(kijun):
        if prev['ICHIMOKU_Tenkan'] <= prev['ICHIMOKU_Kijun'] and tenkan > kijun:
            signals.append({
                'type': 'tk_bullish_cross',
                'name': '轉換線-基準線金叉',
                'description': 'Tenkan上穿Kijun，看漲信號',
                'strength': 'strong',
                'signal': 'bullish'
            })
        elif prev['ICHIMOKU_Tenkan'] >= prev['ICHIMOKU_Kijun'] and tenkan < kijun:
            signals.append({
                'type': 'tk_bearish_cross',
                'name': '轉換線-基準線死叉',
                'description': 'Tenkan下穿Kijun，看跌信號',
                'strength': 'strong',
                'signal': 'bearish'
            })
    
    # 2. 價格與雲帶關係
    if pd.notna(senkou_a) and pd.notna(senkou_b):
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if current_price > cloud_top:
            signals.append({
                'type': 'price_above_cloud',
                'name': '價格在雲圖上方',
                'description': '多頭趨勢，信號強勢',
                'strength': 'strong',
                'signal': 'bullish'
            })
        elif current_price < cloud_bottom:
            signals.append({
                'type': 'price_below_cloud',
                'name': '價格在雲圖下方',
                'description': '空頭趨勢，信號弱勢',
                'strength': 'strong',
                'signal': 'bearish'
            })
        else:
            signals.append({
                'type': 'price_in_cloud',
                'name': '價格在雲帶內',
                'description': '震盪行情，建議觀望',
                'strength': 'neutral',
                'signal': 'neutral'
            })
    
    # 3. 雲帶變化信號
    if pd.notna(senkou_a) and pd.notna(senkou_b):
        if senkou_a > senkou_b:
            signals.append({
                'type': 'cloud_green',
                'name': '雲帶為綠色',
                'description': '多頭雲帶（Span A > Span B）',
                'strength': 'medium',
                'signal': 'bullish'
            })
        else:
            signals.append({
                'type': 'cloud_red',
                'name': '雲帶為紅色',
                'description': '空頭雲帶（Span A < Span B）',
                'strength': 'medium',
                'signal': 'bearish'
            })
    
    # 4. Chikou 交叉（需歷史數據）
    if len(hist) > ICHIMOKU_DISPLACEMENT:
        chikou = latest.get('ICHIMOKU_Chikou')
        if pd.notna(chikou):
            # 比較26天前的價格
            hist_idx = len(hist) - 1 - ICHIMOKU_DISPLACEMENT
            if hist_idx >= 0:
                price_26d_ago = hist.iloc[hist_idx]['Close']
                if chikou > price_26d_ago:
                    signals.append({
                        'type': 'chikou_above',
                        'name': '遲行線在價格上方',
                        'description': '多頭確認信號',
                        'strength': 'medium',
                        'signal': 'bullish'
                    })
                else:
                    signals.append({
                        'type': 'chikou_below',
                        'name': '遲行線在價格下方',
                        'description': '空頭確認信號',
                        'strength': 'medium',
                        'signal': 'bearish'
                    })
    
    # 計算整體信號強度
    bullish_count = sum(1 for s in signals if s.get('signal') == 'bullish')
    bearish_count = sum(1 for s in signals if s.get('signal') == 'bearish')
    
    if bullish_count > bearish_count:
        overall = 'bullish'
    elif bearish_count > bullish_count:
        overall = 'bearish'
    else:
        overall = 'neutral'
    
    return {
        'signals': signals,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'overall': overall,
        'tenkan': round(tenkan, 2) if pd.notna(tenkan) else None,
        'kijun': round(kijun, 2) if pd.notna(kijun) else None,
        'senkou_a': round(senkou_a, 2) if pd.notna(senkou_a) else None,
        'senkou_b': round(senkou_b, 2) if pd.notna(senkou_b) else None,
        'cloud_top': round(max(senkou_a, senkou_b), 2) if pd.notna(senkou_a) and pd.notna(senkou_b) else None,
        'cloud_bottom': round(min(senkou_a, senkou_b), 2) if pd.notna(senkou_a) and pd.notna(senkou_b) else None,
        'price_position': 'above_cloud' if current_price > max(senkou_a, senkou_b) else ('below_cloud' if current_price < min(senkou_a, senkou_b) else 'in_cloud') if pd.notna(senkou_a) and pd.notna(senkou_b) else 'unknown',
        'summary': f"{bullish_count}個看漲信號 / {bearish_count}個看跌信號"
    }


def format_ichimoku_analysis(ichimoku_data: Dict) -> str:
    """
    格式化 Ichimoku 分析結果
    
    Args:
        ichimoku_data: get_ichimoku_signals 返回的數據
        
    Returns:
        str: 格式化的分析結果
    """
    if 'error' in ichimoku_data:
        return f"Ichimoku分析: {ichimoku_data['error']}"
    
    lines = [
        "【Ichimoku 雲圖分析】",
        f"整體信號: {'看漲' if ichimoku_data['overall'] == 'bullish' else '看跌' if ichimoku_data['overall'] == 'bearish' else '中性'}",
        f"",
        "關鍵數值:",
        f"  Tenkan-sen (轉換線): {ichimoku_data.get('tenkan', 'N/A')}",
        f"  Kijun-sen (基準線): {ichimoku_data.get('kijun', 'N/A')}",
        f"  Senkou Span A (先行上線): {ichimoku_data.get('senkou_a', 'N/A')}",
        f"  Senkou Span B (先行下線): {ichimoku_data.get('senkou_b', 'N/A')}",
        f"  雲帶頂部: {ichimoku_data.get('cloud_top', 'N/A')}",
        f"  雲帶底部: {ichimoku_data.get('cloud_bottom', 'N/A')}",
        f"  價格位置: {ichimoku_data.get('price_position', 'N/A')}",
        f"",
        "信號列表:"
    ]
    
    signals = ichimoku_data.get('signals', [])
    if signals:
        for s in signals:
            emoji = "🟢" if s['signal'] == 'bullish' else "🔴" if s['signal'] == 'bearish' else "⚪"
            lines.append(f"  {emoji} {s['name']}: {s['description']}")
    else:
        lines.append("  無明顯信號")
    
    lines.append(f"")
    lines.append(f"{ichimoku_data['summary']}")
    
    return "\n".join(lines)


# 測試函數
if __name__ == "__main__":
    import numpy as np
    
    # 創建模擬數據
    dates = pd.date_range('2024-01-01', periods=100)
    
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.random.rand(100) * 5
    low = close - np.random.rand(100) * 5
    
    hist = pd.DataFrame({
        'Close': close,
        'High': high,
        'Low': low,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    hist = calculate_ichimoku(hist)
    signals = get_ichimoku_signals(hist)
    
    print(format_ichimoku_analysis(signals))
