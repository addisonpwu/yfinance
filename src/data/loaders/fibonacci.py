"""
斐波那契回調位計算模組

計算指定周期內的斐波那契回調位：
- 23.6%, 38.2%, 50%, 61.8%, 78.6%
- 支持擴展位：138.6%, 161.8%, 200%

作者: iFlow CLI Team
日期: 2026-03-18
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# 斐波那契回調位比例
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIBONACCI_EXTENSIONS = [1.236, 1.382, 1.618, 2.0]


def calculate_fibonacci_retracements(
    high: float,
    low: float,
    include_extensions: bool = False
) -> Dict[str, float]:
    """
    計算斐波那契回調位和擴展位
    
    Args:
        high: 周期內最高價
        low: 周期內最低價
        include_extensions: 是否包含擴展位
        
    Returns:
        Dict: 包含各個斐波那契位及價格
            {
                'retracements': {0.236: price, 0.382: price, ...},
                'extensions': {1.236: price, ...},
                'range': high - low
            }
    """
    diff = high - low
    
    retracements = {}
    for level in FIBONACCI_LEVELS:
        # 回調位 = 高點 - (高點 - 低點) × 比例
        price = high - (diff * level)
        retracements[level] = round(price, 2)
    
    extensions = {}
    if include_extensions:
        for level in FIBONACCI_EXTENSIONS:
            # 擴展位 = 低點 + (高點 - 低點) × 比例
            price = low + (diff * level)
            extensions[level] = round(price, 2)
    
    return {
        'retracements': retracements,
        'extensions': extensions,
        'range': round(diff, 2),
        'high': round(high, 2),
        'low': round(low, 2)
    }


def find_swing_points(
    hist: pd.DataFrame,
    lookback_period: int = 60,
    min_swing_strength: float = 0.03
) -> Dict:
    """
    識別波段高點和低點
    
    Args:
        hist: 歷史數據 DataFrame
        lookback_period: 回看天數
        min_swing_strength: 最小波動幅度（默認3%）
        
    Returns:
        Dict: 包含波段高點、低點及相關信息
    """
    if hist is None or hist.empty or len(hist) < 10:
        return {'error': '數據不足'}
    
    # 獲取指定周期的數據
    data = hist.tail(min(lookback_period, len(hist)))
    
    # 找最高點和最低點
    high_idx = data['High'].idxmax()
    low_idx = data['Low'].idxmin()
    
    swing_high = data.loc[high_idx, 'High']
    swing_low = data.loc[low_idx, 'Low']
    
    # 計算波動幅度
    swing_range = swing_high - swing_low
    swing_pct = swing_range / swing_low if swing_low > 0 else 0
    
    # 確定趨勢方向（高點在高點的右側為上升趨勢）
    is_uptrend = high_idx > low_idx
    
    return {
        'swing_high': round(swing_high, 2),
        'swing_high_date': str(high_idx.date()) if hasattr(high_idx, 'date') else str(high_idx),
        'swing_low': round(swing_low, 2),
        'swing_low_date': str(low_idx.date()) if hasattr(low_idx, 'date') else str(low_idx),
        'swing_range': round(swing_range, 2),
        'swing_pct': round(swing_pct * 100, 2),
        'trend': 'uptrend' if is_uptrend else 'downtrend',
        'is_significant': swing_pct >= min_swing_strength
    }


def get_fibonacci_levels(
    hist: pd.DataFrame,
    period: int = 60,
    include_extensions: bool = False
) -> Dict:
    """
    獲取股票的斐波那契回調位（完整分析）
    
    Args:
        hist: 歷史數據 DataFrame
        period: 分析周期（天數）
        include_extensions: 是否包含擴展位
        
    Returns:
        Dict: 完整的斐波那契分析結果
    """
    if hist is None or hist.empty:
        return {'error': '數據不足'}
    
    # 識別波段點
    swing = find_swing_points(hist, lookback_period=period)
    
    if 'error' in swing:
        return swing
    
    # 根據趨勢選擇高點和低點
    if swing['trend'] == 'uptrend':
        # 上升趨勢：從低點漲到高點
        reference_low = swing['swing_low']
        reference_high = swing['swing_high']
    else:
        # 下降趨勢：從高點跌到低點
        reference_low = swing['swing_low']
        reference_high = swing['swing_high']
    
    # 計算斐波那契位
    fib_levels = calculate_fibonacci_retracements(
        reference_high,
        reference_low,
        include_extensions=include_extensions
    )
    
    # 獲取當前價格
    current_price = hist['Close'].iloc[-1]
    
    # 計算價格與各斐波那契位的距離
    distances = {}
    for level, price in fib_levels['retracements'].items():
        distance = ((current_price - price) / price) * 100
        distances[f'{level:.3f}'] = round(distance, 2)
    
    # 識別最近的支撐位和阻力位
    supports = []
    resistances = []
    
    if swing['trend'] == 'uptrend':
        # 上升趨勢：低點方向是支撐，高點方向是阻力
        for level, price in fib_levels['retracements'].items():
            if price < current_price:
                supports.append({'level': level, 'price': price, 'distance_pct': distances[f'{level:.3f}']})
            else:
                resistances.append({'level': level, 'price': price, 'distance_pct': distances[f'{level:.3f}']})
    else:
        # 下降趨勢：相反
        for level, price in fib_levels['retracements'].items():
            if price > current_price:
                supports.append({'level': level, 'price': price, 'distance_pct': distances[f'{level:.3f}']})
            else:
                resistances.append({'level': level, 'price': price, 'distance_pct': distances[f'{level:.3f}']})
    
    # 按距離排序
    supports.sort(key=lambda x: abs(x['distance_pct']))
    resistances.sort(key=lambda x: abs(x['distance_pct']))
    
    return {
        'current_price': round(current_price, 2),
        'trend': swing['trend'],
        'swing_high': fib_levels['high'],
        'swing_low': fib_levels['low'],
        'range': fib_levels['range'],
        'retracements': fib_levels['retracements'],
        'extensions': fib_levels.get('extensions', {}),
        'nearest_supports': supports[:3],
        'nearest_resistances': resistances[:3],
        'distances_to_levels': distances,
        'swing_info': swing
    }


def format_fibonacci_analysis(fib_data: Dict) -> str:
    """
    格式化斐波那契分析結果為字符串
    
    Args:
        fib_data: get_fibonacci_levels 返回的數據
        
    Returns:
        str: 格式化的分析結果
    """
    if 'error' in fib_data:
        return f"斐波那契分析: {fib_data['error']}"
    
    lines = [
        "【斐波那契回調分析】",
        f"當前價格: {fib_data['current_price']}",
        f"趨勢判斷: {'上升趨勢' if fib_data['trend'] == 'uptrend' else '下降趨勢'}",
        f"波段高點: {fib_data['swing_high']}",
        f"波段低點: {fib_data['swing_low']}",
        f"波動幅度: {fib_data['range']} ({fib_data['swing_info']['swing_pct']}%)",
        "",
        "回調位:",
    ]
    
    for level, price in fib_data['retracements'].items():
        level_pct = int(level * 100)
        dist = fib_data['distances_to_levels'].get(f'{level:.3f}', 0)
        direction = "↑" if dist > 0 else "↓"
        lines.append(f"  {level_pct}%: {price} ({direction}{abs(dist):.1f}%)")
    
    if fib_data.get('extensions'):
        lines.append("")
        lines.append("擴展位:")
        for level, price in fib_data['extensions'].items():
            level_pct = int((level - 1) * 100) + 100
            lines.append(f"  {level_pct}%: {price}")
    
    lines.append("")
    lines.append("支撐位 (最近3個):")
    for s in fib_data['nearest_supports']:
        lines.append(f"  {int(s['level']*100)}%: {s['price']}")
    
    lines.append("阻力位 (最近3個):")
    for r in fib_data['nearest_resistances']:
        lines.append(f"  {int(r['level']*100)}%: {r['price']}")
    
    return "\n".join(lines)


# 測試函數
if __name__ == "__main__":
    # 簡單測試
    result = calculate_fibonacci_retracements(100, 80)
    print("斐波那契回調位測試:")
    print(f"高點: 100, 低點: 80")
    for level, price in result['retracements'].items():
        print(f"  {int(level*100)}%: {price}")