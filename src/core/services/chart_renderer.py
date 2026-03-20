"""
K線圖渲染模組

使用 Lightweight Charts 渲染專業級 K線圖：
- K線圖（陰陽燭）
- 技術指標疊加（MA、BB、MACD、RSI）
- 成交量柱狀圖
- 響應式設計

作者: iFlow CLI Team
日期: 2026-03-18
"""

def generate_candlestick_html(
    symbol: str,
    hist_data: list,
    title: str = "",
    width: int = 800,
    height: int = 500,
    show_ma: bool = True,
    show_bb: bool = True,
    show_volume: bool = True,
    interval: str = "1d"
) -> str:
    """
    生成 K線圖 HTML 代碼
    
    Args:
        symbol: 股票代碼
        hist_data: K線數據 [{time, open, high, low, close, volume, ...}]
        title: 圖表標題
        width: 寬度
        height: 高度
        show_ma: 顯示移動平均線
        show_bb: 顯示布林帶
        show_volume: 顯示成交量
        interval: 數據週期
        
    Returns:
        str: HTML 代碼
    """
    
    # 處理 K線數據
    candle_data = []
    volume_data = []
    ma5_data = []
    ma10_data = []
    ma20_data = []
    ma50_data = []
    bb_upper = []
    bb_middle = []
    bb_lower = []
    
    for i, bar in enumerate(hist_data):
        # 時間戳
        if isinstance(bar.get('Date'), str):
            time_str = bar['Date']
        else:
            time_str = str(i)
        
        candle_data.append({
            'time': time_str,
            'open': float(bar.get('Open', bar.get('close', 0))),
            'high': float(bar.get('High', bar.get('close', 0))),
            'low': float(bar.get('Low', bar.get('close', 0))),
            'close': float(bar.get('Close', bar.get('close', 0)))
        })
        
        if show_volume and 'Volume' in bar:
            volume_data.append({
                'time': time_str,
                'value': float(bar.get('Volume', 0)),
                'color': '#26a69a' if float(bar.get('Close', 0)) >= float(bar.get('Open', 0)) else '#ef5350'
            })
        
        # MA 數據
        if show_ma:
            for ma_period, ma_key in [(5, 'MA_5'), (10, 'MA_10'), (20, 'MA_20'), (50, 'MA_50')]:
                ma_val = bar.get(ma_key)
                if ma_val is not None:
                    if ma_period == 5:
                        ma5_data.append({'time': time_str, 'value': float(ma_val)})
                    elif ma_period == 10:
                        ma10_data.append({'time': time_str, 'value': float(ma_val)})
                    elif ma_period == 20:
                        ma20_data.append({'time': time_str, 'value': float(ma_val)})
                    elif ma_period == 50:
                        ma50_data.append({'time': time_str, 'value': float(ma_val)})
        
        # BB 數據
        if show_bb:
            bb_up = bar.get('BB_Upper')
            bb_mid = bar.get('BB_Middle')
            bb_low = bar.get('BB_Lower')
            if bb_up is not None:
                bb_upper.append({'time': time_str, 'value': float(bb_up)})
            if bb_mid is not None:
                bb_middle.append({'time': time_str, 'value': float(bb_mid)})
            if bb_low is not None:
                bb_lower.append({'time': time_str, 'value': float(bb_low)})
    
    # 轉換為 JSON
    import json
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} K線圖</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            padding: 20px;
        }}
        .chart-container {{
            max-width: {width}px;
            margin: 0 auto;
        }}
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 15px 20px;
            background: #16213e;
            border-radius: 8px;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
        }}
        .chart-symbol {{
            font-size: 14px;
            color: #888;
        }}
        .chart-controls {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: #2196F3;
            color: white;
        }}
        .btn-primary:hover {{
            background: #1976D2;
        }}
        .btn-secondary {{
            background: #424242;
            color: #aaa;
        }}
        .btn-secondary:hover {{
            background: #616161;
        }}
        .btn-active {{
            background: #4CAF50;
            color: white;
        }}
        #chart {{
            width: 100%;
            height: {height}px;
        }}
        #volume-chart {{
            width: 100%;
            height: 100px;
        }}
        .chart-wrapper {{
            background: #16213e;
            border-radius: 8px;
            padding: 10px;
        }}
        .legend {{
            position: absolute;
            left: 12px;
            top: 12px;
            z-index: 1;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.5);
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 4px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <div class="chart-header">
            <div>
                <div class="chart-title">{title or symbol}</div>
                <div class="chart-symbol">{symbol} - {interval}</div>
            </div>
            <div class="chart-controls">
                <button class="btn btn-secondary" onclick="setInterval('1d')">日線</button>
                <button class="btn btn-secondary" onclick="setInterval('1h')">小時</button>
                <button class="btn btn-secondary" onclick="resetTimeframe()">重置</button>
            </div>
        </div>
        
        <div class="chart-wrapper">
            <div id="chart">
                <div class="legend">
                    <span class="legend-item"><span class="legend-color" style="background: #2196F3"></span>K線</span>
                    <span class="legend-item"><span class="legend-color" style="background: #FF9800"></span>MA5</span>
                    <span class="legend-item"><span class="legend-color" style="background: #9C27B0"></span>MA20</span>
                    <span class="legend-item"><span class="legend-color" style="background: #00BCD4"></span>BB</span>
                </div>
            </div>
            {"<div id='volume-chart'></div>" if show_volume else ""}
        </div>
    </div>

    <script>
        // 創建圖表
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
            width: document.getElementById('chart').clientWidth,
            height: {height - (20 if show_volume else 0)},
            layout: {{
                background: {{ type: 'solid', color: '#16213e' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2B2B43' }},
                horzLines: {{ color: '#2B2B43' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            timeScale: {{
                borderColor: '#2B2B43',
                timeVisible: true,
            }},
            rightPriceScale: {{
                borderColor: '#2B2B43',
            }},
        }});

        // K線系列
        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderDownColor: '#ef5350',
            borderUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            wickUpColor: '#26a69a',
        }});

        // 設置K線數據
        candleSeries.setData({json.dumps(candle_data[:200])});

        // MA 系列
        {"const ma5Series = chart.addLineSeries({ color: '#FF9800', lineWidth: 1, priceLineVisible: false }); ma5Series.setData(" + json.dumps(ma5_data[-200:]) + ");" if show_ma and ma5_data else ""}
        {"const ma10Series = chart.addLineSeries({ color: '#E91E63', lineWidth: 1, priceLineVisible: false }); ma10Series.setData(" + json.dumps(ma10_data[-200:]) + ");" if show_ma and ma10_data else ""}
        {"const ma20Series = chart.addLineSeries({ color: '#9C27B0', lineWidth: 1, priceLineVisible: false }); ma20Series.setData(" + json.dumps(ma20_data[-200:]) + ");" if show_ma and ma20_data else ""}
        {"const ma50Series = chart.addLineSeries({ color: '#00BCD4', lineWidth: 1, priceLineVisible: false }); ma50Series.setData(" + json.dumps(ma50_data[-200:]) + ");" if show_ma and ma50_data else ""}

        // 布林帶系列
        {"const bbUpperSeries = chart.addLineSeries({ color: 'rgba(0, 188, 212, 0.3)', lineWidth: 1, priceLineVisible: false }); bbUpperSeries.setData(" + json.dumps(bb_upper[-200:]) + ");" if show_bb and bb_upper else ""}
        {"const bbMiddleSeries = chart.addLineSeries({ color: '#00BCD4', lineWidth: 1, priceLineVisible: false }); bbMiddleSeries.setData(" + json.dumps(bb_middle[-200:]) + ");" if show_bb and bb_middle else ""}
        {"const bbLowerSeries = chart.addLineSeries({ color: 'rgba(0, 188, 212, 0.3)', lineWidth: 1, priceLineVisible: false }); bbLowerSeries.setData(" + json.dumps(bb_lower[-200:]) + ");" if show_bb and bb_lower else ""}

        // 成交量圖表
        {"const volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), { width: document.getElementById('volume-chart').clientWidth, height: 100, layout: { background: { type: 'solid', color: '#16213e' }, textColor: '#d1d4dc' }, grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } }, timeScale: { borderColor: '#2B2B43' } });" if show_volume else ""}
        {"const volumeSeries = volumeChart.addHistogramSeries({ color: '#26a69a', priceFormat: { type: 'volume' }, priceScaleId: '', }); volumeSeries.setData(" + json.dumps(volume_data[-200:]) + "); volumeChart.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 }, });" if show_volume else ""}

        // 響應式調整
        window.addEventListener('resize', () => {{
            chart.applyOptions({{ width: document.getElementById('chart').clientWidth }});
            {"volumeChart.applyOptions({ width: document.getElementById('volume-chart').clientWidth });" if show_volume else ""}
        }});

        // 週期切換
        function setInterval(interval) {{
            console.log('切換到 ' + interval);
            // 這裡可以添加數據重新加載邏輯
        }}

        function resetTimeframe() {{
            chart.timeScale().resetTimeScale();
        }}
    </script>
</body>
</html>"""
    
    return html


def generate_dashboard_html(
    results: list,
    title: str = "股票篩選結果儀表板"
) -> str:
    """
    生成摘要儀表板 HTML
    
    Args:
        results: 股票分析結果列表
        title: 儀表板標題
        
    Returns:
        str: HTML 代碼
    """
    
    # 統計數據
    total = len(results)
    bullish = sum(1 for r in results if 'bullish' in r.get('ai_direction', '').lower())
    bearish = sum(1 for r in results if 'bearish' in r.get('ai_direction', '').lower())
    neutral = total - bullish - bearish
    
    # 策略統計
    strategy_stats = {}
    for r in results:
        for s in r.get('strategies', []):
            strategy_stats[s] = strategy_stats.get(s, 0) + 1
    
    # 排序策略
    sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[1], reverse=True)
    
    import json
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .dashboard-header {{
            margin-bottom: 20px;
        }}
        .dashboard-title {{
            font-size: 24px;
            font-weight: 600;
            color: #333;
        }}
        .dashboard-subtitle {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 600;
            color: #333;
        }}
        .stat-change {{
            font-size: 14px;
            margin-top: 5px;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #F44336;
        }}
        .neutral {{
            color: #9E9E9E;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .chart-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }}
        .stock-list {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stock-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        .stock-item:last-child {{
            border-bottom: none;
        }}
        .stock-symbol {{
            font-weight: 600;
            color: #2196F3;
        }}
        .stock-name {{
            font-size: 12px;
            color: #888;
        }}
        .stock-direction {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        .direction-bullish {{
            background: #E8F5E9;
            color: #4CAF50;
        }}
        .direction-bearish {{
            background: #FFEBEE;
            color: #F44336;
        }}
        .direction-neutral {{
            background: #F5F5F5;
            color: #9E9E9E;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="dashboard-header">
            <div class="dashboard-title">{title}</div>
            <div class="dashboard-subtitle">數據更新: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">篩選數量</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">看漲</div>
                <div class="stat-value positive">{bullish}</div>
                <div class="stat-change positive">{bullish/total*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">看跌</div>
                <div class="stat-value negative">{bearish}</div>
                <div class="stat-change negative">{bearish/total*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">中性</div>
                <div class="stat-value neutral">{neutral}</div>
                <div class="stat-change neutral">{neutral/total*100:.1f}%</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-title">市場方向分佈</div>
                <canvas id="directionChart"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">策略命中統計</div>
                <canvas id="strategyChart"></canvas>
            </div>
        </div>

        <div class="stock-list">
            <div class="chart-title">符合條件的股票 ({total})</div>
            {"".join([f'''
            <div class="stock-item">
                <div>
                    <div class="stock-symbol">{r.get('symbol', 'N/A')}</div>
                    <div class="stock-name">{r.get('name', '')}</div>
                </div>
                <div class="stock-direction direction-{'bullish' if 'bullish' in r.get('ai_direction', '').lower() else 'bearish' if 'bearish' in r.get('ai_direction', '').lower() else 'neutral'}">
                    {r.get('ai_direction', 'N/A')}
                </div>
            </div>''' for r in results[:10]])}
        </div>
    </div>

    <script>
        // 市場方向圖表
        const directionCtx = document.getElementById('directionChart').getContext('2d');
        new Chart(directionCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['看漲', '看跌', '中性'],
                datasets: [{{
                    data: [{bullish}, {bearish}, {neutral}],
                    backgroundColor: ['#4CAF50', '#F44336', '#9E9E9E'],
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                    }}
                }}
            }}
        }});

        // 策略命中圖表
        const strategyCtx = document.getElementById('strategyChart').getContext('2d');
        new Chart(strategyCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([s[0] for s in sorted_strategies[:5]])},
                datasets: [{{
                    label: '命中次數',
                    data: {json.dumps([s[1] for s in sorted_strategies[:5]])},
                    backgroundColor: '#2196F3',
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false,
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    return html


# 測試函數
if __name__ == "__main__":
    # 簡單測試
    import pandas as pd
    import numpy as np
    
    # 模擬 K線數據
    dates = pd.date_range('2024-01-01', periods=50)
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(50))
    data = []
    
    for i, (date, c) in enumerate(zip(dates, close)):
        open_price = c + np.random.randn() * 2
        high_price = max(c, open_price) + abs(np.random.randn() * 2)
        low_price = min(c, open_price) - abs(np.random.randn() * 2)
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': c,
            'Volume': int(np.random.randint(1000000, 5000000)),
            'MA_5': pd.Series(close[:i+1]).rolling(5).mean().iloc[-1] if i >= 4 else None,
            'MA_20': pd.Series(close[:i+1]).rolling(20).mean().iloc[-1] if i >= 19 else None,
        })
    
    html = generate_candlestick_html('TEST', data, '測試股票')
    print(f"生成的HTML長度: {len(html)} 字節")
