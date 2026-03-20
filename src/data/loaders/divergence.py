"""
背離信號檢測模組

自動識別 RSI、MACD、價格之間的頂/底背離：
- RSI 頂背離：價格創新高但 RSI 未能創新高
- RSI 底背離：價格創新低但 RSI 未創新低
- MACD 頂背離：價格新高但 MACD 峰值降低
- MACD 底背離：價格新低但 MACD 谷值抬高
- 量價背離：價格上漲但成交量萎縮，或價格下跌但成交量放大

作者: iFlow CLI Team
日期: 2026-03-18
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DivergenceSignal:
    """背離信號"""
    type: str  # 'rsi_bearish', 'rsi_bullish', 'macd_bearish', 'macd_bullish', 'volume_bearish', 'volume_bullish'
    strength: float  # 信號強度 0-1
    price_high: float  # 價格高點
    price_low: float  # 價格低點
    indicator_high: float  # 指標高點
    indicator_low: float  # 指標低點
    price_date: str  # 價格日期
    description: str  # 描述


def detect_rsi_divergence(
    hist: pd.DataFrame,
    lookback_period: int = 20,
    min_price_change: float = 0.03
) -> Dict:
    """
    檢測 RSI 背離
    
    Args:
        hist: 包含 RSI 指標的歷史數據
        lookback_period: 回看天數
        min_price_change: 最小價格變化（默認3%）
        
    Returns:
        Dict: 包含背離檢測結果
    """
    if hist is None or hist.empty:
        return {'error': '數據不足'}
    
    if 'RSI_14' not in hist.columns:
        return {'error': '缺少 RSI_14 指標'}
    
    # 獲取最近 N 天的數據
    data = hist.tail(lookback_period)
    
    # 找局部高點和低點
    close_prices = data['Close']
    rsi_values = data['RSI_14']
    
    # 簡單的局部極值識別
    def find_local_extrema(series: pd.Series, window: int = 5) -> Tuple[List, List]:
        """找到局部極值點"""
        highs = []
        lows = []
        
        for i in range(window, len(series) - window):
            is_high = True
            is_low = True
            
            for j in range(1, window + 1):
                if series.iloc[i] <= series.iloc[i - j] or series.iloc[i] <= series.iloc[i + j]:
                    is_high = False
                if series.iloc[i] >= series.iloc[i - j] or series.iloc[i] >= series.iloc[i + j]:
                    is_low = False
            
            if is_high:
                highs.append((i, series.iloc[i]))
            if is_low:
                lows.append((i, series.iloc[i]))
        
        return highs, lows
    
    price_highs, price_lows = find_local_extrema(close_prices)
    rsi_highs, rsi_lows = find_local_extrema(rsi_values)
    
    if not price_highs or not rsi_highs:
        return {'divergences': [], 'summary': '未找到足夠的極值點'}
    
    divergences = []
    
    # 檢測頂背離：價格創新高但 RSI 未創新高
    if price_highs and rsi_highs:
        latest_price_high_idx = price_highs[-1][0]
        latest_price_high = price_highs[-1][1]
        
        # 找前一個高點
        prev_price_high = price_highs[-2][1] if len(price_highs) > 1 else price_highs[-1][1]
        
        # 檢查 RSI 是否更低
        if len(rsi_highs) >= 2:
            latest_rsi_high = rsi_highs[-1][1]
            prev_rsi_high = rsi_highs[-2][1]
            
            if latest_price_high > prev_price_high and latest_rsi_high < prev_rsi_high:
                # 頂背離確認
                strength = min(1.0, (prev_rsi_high - latest_rsi_high) / prev_rsi_high)
                divergences.append({
                    'type': 'rsi_bearish',
                    'name': 'RSI 頂背離',
                    'strength': round(strength, 2),
                    'description': f'價格創新高({latest_price_high:.2f})但RSI下降({latest_rsi_high:.1f} vs {prev_rsi_high:.1f})',
                    'signal': 'bearish',
                    'action': '注意風險'
                })
    
    # 檢測底背離：價格創新低但 RSI 未創新低
    if price_lows and rsi_lows:
        latest_price_low_idx = price_lows[-1][0]
        latest_price_low = price_lows[-1][1]
        
        prev_price_low = price_lows[-2][1] if len(price_lows) > 1 else price_lows[-1][1]
        
        if len(rsi_lows) >= 2:
            latest_rsi_low = rsi_lows[-1][1]
            prev_rsi_low = rsi_lows[-2][1]
            
            if latest_price_low < prev_price_low and latest_rsi_low > prev_rsi_low:
                # 底背離確認
                strength = min(1.0, (latest_rsi_low - prev_rsi_low) / prev_rsi_low if prev_rsi_low > 0 else 0)
                divergences.append({
                    'type': 'rsi_bullish',
                    'name': 'RSI 底背離',
                    'strength': round(strength, 2),
                    'description': f'價格創新低({latest_price_low:.2f})但RSI上升({latest_rsi_low:.1f} vs {prev_rsi_low:.1f})',
                    'signal': 'bullish',
                    'action': '關注買入機會'
                })
    
    return {
        'divergences': divergences,
        'has_bearish': any(d['type'] == 'rsi_bearish' for d in divergences),
        'has_bullish': any(d['type'] == 'rsi_bullish' for d in divergences),
        'summary': f"檢測到 {len(divergences)} 個 RSI 背離信號"
    }


def detect_macd_divergence(
    hist: pd.DataFrame,
    lookback_period: int = 20
) -> Dict:
    """
    檢測 MACD 背離
    
    Args:
        hist: 包含 MACD 指標的歷史數據
        lookback_period: 回看天數
        
    Returns:
        Dict: 包含背離檢測結果
    """
    if hist is None or hist.empty:
        return {'error': '數據不足'}
    
    if 'MACD' not in hist.columns:
        return {'error': '缺少 MACD 指標'}
    
    data = hist.tail(lookback_period)
    
    close_prices = data['Close']
    macd_values = data['MACD']
    
    # 找局部極值
    def find_local_extrema(series: pd.Series, window: int = 5) -> Tuple[List, List]:
        highs = []
        lows = []
        
        for i in range(window, len(series) - window):
            is_high = True
            is_low = True
            
            for j in range(1, window + 1):
                if series.iloc[i] <= series.iloc[i - j] or series.iloc[i] <= series.iloc[i + j]:
                    is_high = False
                if series.iloc[i] >= series.iloc[i - j] or series.iloc[i] >= series.iloc[i + j]:
                    is_low = False
            
            if is_high:
                highs.append((i, series.iloc[i]))
            if is_low:
                lows.append((i, series.iloc[i]))
        
        return highs, lows
    
    price_highs, price_lows = find_local_extrema(close_prices)
    macd_highs, macd_lows = find_local_extrema(macd_values)
    
    divergences = []
    
    # 頂背離
    if price_highs and macd_highs and len(macd_highs) >= 2:
        latest_price_high = price_highs[-1][1]
        prev_price_high = price_highs[-2][1] if len(price_highs) > 1 else price_highs[-1][1]
        
        latest_macd_high = macd_highs[-1][1]
        prev_macd_high = macd_highs[-2][1]
        
        if latest_price_high > prev_price_high and latest_macd_high < prev_macd_high:
            strength = min(1.0, abs(prev_macd_high - latest_macd_high) / abs(prev_macd_high))
            divergences.append({
                'type': 'macd_bearish',
                'name': 'MACD 頂背離',
                'strength': round(strength, 2),
                'description': f'價格新高但MACD峰值降低',
                'signal': 'bearish',
                'action': '注意風險'
            })
    
    # 底背離
    if price_lows and macd_lows and len(macd_lows) >= 2:
        latest_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1] if len(price_lows) > 1 else price_lows[-1][1]
        
        latest_macd_low = macd_lows[-1][1]
        prev_macd_low = macd_lows[-2][1]
        
        if latest_price_low < prev_price_low and latest_macd_low > prev_macd_low:
            strength = min(1.0, abs(latest_macd_low - prev_macd_low) / abs(prev_macd_low))
            divergences.append({
                'type': 'macd_bullish',
                'name': 'MACD 底背離',
                'strength': round(strength, 2),
                'description': f'價格新低但MACD谷值擡高',
                'signal': 'bullish',
                'action': '關注買入機會'
            })
    
    return {
        'divergences': divergences,
        'has_bearish': any(d['type'] == 'macd_bearish' for d in divergences),
        'has_bullish': any(d['type'] == 'macd_bullish' for d in divergences),
        'summary': f"檢測到 {len(divergences)} 個 MACD 背離信號"
    }


def detect_volume_price_divergence(
    hist: pd.DataFrame,
    lookback_period: int = 10
) -> Dict:
    """
    檢測量價背離
    
    Args:
        hist: 歷史數據
        lookback_period: 回看天數
        
    Returns:
        Dict: 包含背離檢測結果
    """
    if hist is None or hist.empty or len(hist) < lookback_period:
        return {'error': '數據不足'}
    
    data = hist.tail(lookback_period + 1)
    
    # 計算價格變化和成交量變化
    price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-lookback_period]) / data['Close'].iloc[-lookback_period]
    volume_change = (data['Volume'].iloc[-1] - data['Volume'].iloc[-lookback_period]) / data['Volume'].iloc[-lookback_period]
    
    divergences = []
    
    # 價格上漲但成交量萎縮
    if price_change > 0.02 and volume_change < -0.2:
        divergences.append({
            'type': 'volume_bearish',
            'name': '量價背離（價漲量縮）',
            'strength': round(min(1.0, abs(volume_change)), 2),
            'description': f'價格上漲{price_change*100:.1f}%，但成交量下降{abs(volume_change)*100:.1f}%',
            'signal': 'bearish',
            'action': '量能不足，注意風險'
        })
    
    # 價格下跌但成交量放大
    if price_change < -0.02 and volume_change > 0.2:
        divergences.append({
            'type': 'volume_bullish',
            'name': '量價背離（價跌量增）',
            'strength': round(min(1.0, volume_change), 2),
            'description': f'價格下跌{abs(price_change)*100:.1f}%，但成交量增加{volume_change*100:.1f}%',
            'signal': 'bullish',
            'action': '低位承接，關注機會'
        })
    
    return {
        'divergences': divergences,
        'has_bearish': any(d['type'] == 'volume_bearish' for d in divergences),
        'has_bullish': any(d['type'] == 'volume_bullish' for d in divergences),
        'price_change_pct': round(price_change * 100, 2),
        'volume_change_pct': round(volume_change * 100, 2),
        'summary': f"檢測到 {len(divergences)} 個量價背離信號"
    }


def detect_all_divergences(
    hist: pd.DataFrame,
    lookback_period: int = 20
) -> Dict:
    """
    檢測所有類型的背離
    
    Args:
        hist: 歷史數據
        lookback_period: 回看天數
        
    Returns:
        Dict: 完整的背離分析結果
    """
    results = {
        'rsi_divergence': detect_rsi_divergence(hist, lookback_period),
        'macd_divergence': detect_macd_divergence(hist, lookback_period),
        'volume_divergence': detect_volume_price_divergence(hist, min(10, lookback_period))
    }
    
    # 匯總所有背離信號
    all_divergences = []
    bullish_count = 0
    bearish_count = 0
    
    for source, result in results.items():
        if 'divergences' in result:
            for d in result['divergences']:
                d['source'] = source
                all_divergences.append(d)
                if d.get('signal') == 'bullish':
                    bullish_count += 1
                elif d.get('signal') == 'bearish':
                    bearish_count += 1
    
    # 計算總體信號強度
    total_strength = sum(d.get('strength', 0) for d in all_divergences)
    avg_strength = total_strength / len(all_divergences) if all_divergences else 0
    
    # 判斷總體方向
    if bullish_count > bearish_count:
        overall_signal = 'bullish'
    elif bearish_count > bullish_count:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    return {
        'divergences': all_divergences,
        'total_count': len(all_divergences),
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'overall_signal': overall_signal,
        'avg_strength': round(avg_strength, 2),
        'rsi_divergence': results['rsi_divergence'],
        'macd_divergence': results['macd_divergence'],
        'volume_divergence': results['volume_divergence'],
        'summary': f"共{len(all_divergences)}個背離信號(看漲{bullish_count}個/看跌{bearish_count}個)"
    }


def format_divergence_analysis(divergence_data: Dict) -> str:
    """
    格式化背離分析結果
    
    Args:
        divergence_data: detect_all_divergences 返回的數據
        
    Returns:
        str: 格式化的分析結果
    """
    if 'error' in divergence_data:
        return f"背離分析: {divergence_data['error']}"
    
    lines = [
        "【背離信號分析】",
        f"總結: {divergence_data['summary']}",
        f"整體信號: {'看漲' if divergence_data['overall_signal'] == 'bullish' else '看跌' if divergence_data['overall_signal'] == 'bearish' else '中性'}",
        f"平均強度: {divergence_data['avg_strength']:.0%}",
        ""
    ]
    
    # RSI 背離
    rsi = divergence_data.get('rsi_divergence', {})
    if 'divergences' in rsi and rsi['divergences']:
        lines.append("RSI 背離:")
        for d in rsi['divergences']:
            lines.append(f"  • {d['name']}: {d['description']}")
            lines.append(f"    信號: {d['action']}")
        lines.append("")
    
    # MACD 背離
    macd = divergence_data.get('macd_divergence', {})
    if 'divergences' in macd and macd['divergences']:
        lines.append("MACD 背離:")
        for d in macd['divergences']:
            lines.append(f"  • {d['name']}: {d['description']}")
            lines.append(f"    信號: {d['action']}")
        lines.append("")
    
    # 量價背離
    vol = divergence_data.get('volume_divergence', {})
    if 'divergences' in vol and vol['divergences']:
        lines.append("量價背離:")
        for d in vol['divergences']:
            lines.append(f"  • {d['name']}: {d['description']}")
            lines.append(f"    信號: {d['action']}")
    
    return "\n".join(lines)


# 測試函數
if __name__ == "__main__":
    import numpy as np
    
    # 模擬數據測試
    dates = pd.date_range('2024-01-01', periods=60)
    
    # 創建頂背離數據：價格新高但 RSI 降低
    close = np.linspace(100, 120, 30).tolist() + np.linspace(120, 130, 15).tolist() + np.linspace(130, 125, 15).tolist()
    rsi = np.linspace(70, 80, 30).tolist() + np.linspace(80, 70, 15).tolist() + np.linspace(70, 65, 15).tolist()
    
    hist = pd.DataFrame({
        'Close': close,
        'RSI_14': rsi,
        'MACD': np.random.randn(60) * 0.5,
        'Volume': np.random.randint(1000000, 5000000, 60)
    }, index=dates)
    
    result = detect_all_divergences(hist)
    print(format_divergence_analysis(result))
