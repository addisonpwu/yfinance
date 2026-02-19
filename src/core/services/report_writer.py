"""
报告写入器

负责报告文件的实时输出和格式化，支持 TXT 和 HTML 格式
"""
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json


class ReportWriter:
    """报告写入器，支持 TXT 和 HTML 格式"""
    
    # 交易所映射
    EXCHANGE_MAP = {
        'NMS': 'NASDAQ',
        'NGM': 'NASDAQ',
        'NCM': 'NASDAQ',
        'NYQ': 'NYSE',
        'PCX': 'NYSE ARCA',
        'TAI': 'TWSE',
        'HKG': 'HKEX'
    }
    
    # 策略名称中文映射
    STRATEGY_NAMES = {
        'momentum_breakout': '动量突破',
        'volatility_squeeze': '波动率压缩',
        'accumulation_acceleration': '主力吸筹加速',
        'signal_scorer': '信号评分器',
        'market_regime': '市场状态'
    }
    
    def __init__(self, filename: str = None, market: str = 'HK', output_format: str = 'html'):
        """
        初始化报告写入器
        
        Args:
            filename: 输出文件名（可含或不含扩展名）
            market: 市场代码
            output_format: 输出格式
        """
        raw_filename = filename or self._generate_basename(market)
        # 移除已有的 .txt 扩展名，避免重复
        if raw_filename.endswith('.txt'):
            raw_filename = raw_filename[:-4]
        self.base_filename = raw_filename
        self.market = market
        self.output_format = output_format
        self._lock = threading.Lock()
        self._initialized = False
        self._results: List[Dict[str, Any]] = []
        self._start_time = datetime.now()
    
    def _generate_basename(self, market: str) -> str:
        """生成默认文件名（不含扩展名）"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        return f"{market.lower()}_stocks_{today_str}"
    
    def initialize(self) -> bool:
        """
        初始化报告文件
        
        Returns:
            是否成功创建
        """
        try:
            # 初始化 TXT 文件
            if self.output_format in ('txt', 'both'):
                txt_filename = f"{self.base_filename}.txt"
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(f"=== 股票筛选报告 ===\n")
                    f.write(f"生成时间: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"市场: {self.market}\n")
                    f.write("=" * 50 + "\n\n")
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"创建报告文件时发生错误: {e}")
            return False
    
    def write_stock_result(self, result: Dict[str, Any]) -> None:
        """
        写入单只股票的分析结果
        
        Args:
            result: 股票分析结果字典
        """
        if not self._initialized:
            return
        
        with self._lock:
            self._results.append(result)
            
            # 写入 TXT 格式
            if self.output_format in ('txt', 'both'):
                self._write_txt_stock(result)
    
    def _write_txt_stock(self, result: Dict[str, Any]) -> None:
        """写入 TXT 格式的股票信息"""
        info = result.get('info', {})
        symbol = result.get('symbol', '')
        strategies = result.get('strategies', [])
        ai_analysis = result.get('ai_analysis')
        
        # 格式化字段
        market_cap = info.get('marketCap')
        market_cap_str = f"{market_cap / 1e8:.2f} 亿" if isinstance(market_cap, (int, float)) else "N/A"
        
        pe_ratio = info.get('trailingPE')
        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        
        float_shares = info.get('floatShares')
        float_shares_str = f"{float_shares:,.0f}" if isinstance(float_shares, (int, float)) else "N/A"
        
        volume = info.get('volume')
        volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "N/A"
        
        txt_filename = f"{self.base_filename}.txt"
        try:
            with open(txt_filename, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"✅ {info.get('longName', symbol)} ({symbol})\n")
                f.write(f"{'='*50}\n")
                f.write(f"符合策略: {', '.join(strategies)}\n")
                f.write(f"产业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}\n")
                f.write(f"市值: {market_cap_str}\n")
                f.write(f"流通股本: {float_shares_str}\n")
                f.write(f"成交量: {volume_str}\n")
                f.write(f"市盈率: {pe_ratio_str}\n")
                f.write(f"网站: {info.get('website', 'N/A')}\n")
                
                if ai_analysis:
                    f.write(f"\n--- AI 综合分析 ---\n")
                    f.write(f"{ai_analysis.get('summary', 'N/A')}\n")
                    f.write(f"模型: {ai_analysis.get('model_used', 'N/A')}\n")
                else:
                    f.write(f"\n--- AI 分析未完成 ---\n")
        except Exception as e:
            print(f"写入 TXT 报告时出错: {e}")
    
    def write_summary(self, results: List[Dict[str, Any]], market: str) -> None:
        """
        写入摘要列表并生成最终报告
        
        Args:
            results: 股票分析结果列表
            market: 市场代码
        """
        with self._lock:
            # 写入 TXT 摘要
            if self.output_format in ('txt', 'both'):
                self._write_txt_summary(results, market)
            
            # 生成 HTML 报告
            if self.output_format in ('html', 'both'):
                self._generate_html_report(results, market)
    
    def _write_txt_summary(self, results: List[Dict[str, Any]], market: str) -> None:
        """写入 TXT 格式的摘要"""
        formatted_stocks = []
        
        for stock in results:
            info = stock.get('info', {})
            long_name = info.get('longName', stock['symbol'])
            exchange_name = self.EXCHANGE_MAP.get(stock.get('exchange'), stock.get('exchange', 'UNKNOWN'))
            symbol = stock['symbol']
            
            if market.upper() == 'HK':
                try:
                    symbol = str(int(symbol.replace('.HK', '')))
                except (ValueError, AttributeError):
                    pass
            
            formatted_stocks.append(f"{exchange_name}:{symbol}")
        
        txt_filename = f"{self.base_filename}.txt"
        try:
            with open(txt_filename, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write("--- 摘要列表 ---\n")
                f.write(f"共筛选出 {len(results)} 只股票\n")
                f.write(", ".join(formatted_stocks))
                f.write(f"\n\n报告生成完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"写入 TXT 摘要时出错: {e}")
    
    def _generate_html_report(self, results: List[Dict[str, Any]], market: str) -> None:
        """生成 HTML 格式的完整报告"""
        html_filename = f"{self.base_filename}.html"
        
        html_content = self._build_html(results, market)
        
        try:
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\n📊 HTML 报告已生成: {html_filename}")
            
            # 尝试生成 PDF
            self._try_generate_pdf(html_content)
        except Exception as e:
            print(f"生成 HTML 报告时出错: {e}")
    
    def _try_generate_pdf(self, html_content: str) -> None:
        """尝试生成 PDF 报告"""
        pdf_filename = f"{self.base_filename}.pdf"
        
        # 尝试使用 weasyprint
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(pdf_filename)
            print(f"📄 PDF 报告已生成: {pdf_filename}")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"PDF 生成失败 (weasyprint): {e}")
        
        # 尝试使用 pdfkit (需要 wkhtmltopdf)
        try:
            import pdfkit
            pdfkit.from_string(html_content, pdf_filename, options={
                'encoding': 'UTF-8',
                'quiet': ''
            })
            print(f"📄 PDF 报告已生成: {pdf_filename}")
            return
        except ImportError:
            pass
        except Exception:
            pass
    
    def _build_html(self, results: List[Dict[str, Any]], market: str) -> str:
        """构建完整的 HTML 报告"""
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()
        
        # 按策略分组统计
        strategy_stats = {}
        for r in results:
            for s in r.get('strategies', []):
                strategy_stats[s] = strategy_stats.get(s, 0) + 1
        
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票筛选报告 - {market} - {self._start_time.strftime('%Y-%m-%d')}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 40px 30px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
        }}
        
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .header .meta {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .header .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            opacity: 0.9;
        }}
        
        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: var(--card);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border);
        }}
        
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .stat-card .label {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        
        /* Strategy Tags */
        .strategy-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 30px;
        }}
        
        .strategy-tag {{
            background: var(--card);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .strategy-tag .count {{
            background: var(--primary);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
        }}
        
        /* Stock Cards */
        .stocks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 20px;
        }}
        
        .stock-card {{
            background: var(--card);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stock-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px -8px rgba(0, 0, 0, 0.15);
        }}
        
        .stock-header {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }}
        
        .stock-header h3 {{
            font-size: 1.1rem;
            color: var(--text);
            margin-bottom: 8px;
        }}
        
        .stock-header .symbol {{
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            background: var(--primary);
            color: white;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .stock-strategies {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }}
        
        .stock-strategies .badge {{
            background: var(--success);
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .stock-body {{
            padding: 20px;
        }}
        
        .stock-info {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 15px;
        }}
        
        .stock-info-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: var(--bg);
            border-radius: 8px;
        }}
        
        .stock-info-item .label {{
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        
        .stock-info-item .value {{
            font-weight: 600;
            color: var(--text);
        }}
        
        .stock-info-item .value.positive {{
            color: var(--success);
        }}
        
        .stock-info-item .value.negative {{
            color: var(--danger);
        }}
        
        /* News Section */
        .news-section {{
            background: #f0f9ff;
            border-radius: 12px;
            padding: 16px;
            margin-top: 15px;
            border-left: 4px solid #0ea5e9;
        }}
        
        .news-section h4 {{
            color: #0369a1;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .news-list {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .news-item {{
            padding: 8px 12px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0f2fe;
        }}
        
        .news-title {{
            color: #0c4a6e;
            text-decoration: none;
            font-size: 0.85rem;
            line-height: 1.4;
            display: block;
        }}
        
        .news-title:hover {{
            color: var(--primary);
            text-decoration: underline;
        }}
        
        .news-meta {{
            display: flex;
            gap: 12px;
            margin-top: 4px;
            font-size: 0.75rem;
            color: #64748b;
        }}
        
        .news-date {{
            color: #64748b;
        }}
        
        .news-source {{
            color: #0ea5e9;
        }}
        
        /* AI Analysis */
        .ai-analysis {{
            background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
            border-radius: 12px;
            padding: 16px;
            margin-top: 15px;
            border-left: 4px solid var(--warning);
        }}
        
        .ai-analysis h4 {{
            color: #92400e;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .ai-analysis .content {{
            font-size: 0.9rem;
            color: #78350f;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .ai-analysis .model {{
            margin-top: 10px;
            font-size: 0.8rem;
            color: #a16207;
            text-align: right;
        }}
        
        /* Score Display */
        .score-display {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}
        
        .score-item {{
            text-align: center;
        }}
        
        .score-item .score {{
            font-size: 1.5rem;
            font-weight: 700;
        }}
        
        .score-item .label {{
            font-size: 0.7rem;
            color: var(--text-muted);
        }}
        
        .score-high {{ color: var(--success); }}
        .score-medium {{ color: var(--warning); }}
        .score-low {{ color: var(--danger); }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            margin-top: 40px;
            border-top: 1px solid var(--border);
        }}
        
        /* Print Styles */
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                max-width: 100%;
            }}
            .stock-card {{
                break-inside: avoid;
                page-break-inside: avoid;
            }}
            .header {{
                box-shadow: none;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .stocks-grid {{
                grid-template-columns: 1fr;
            }}
            .header {{
                padding: 25px 20px;
            }}
            .header h1 {{
                font-size: 1.5rem;
            }}
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>
                <span>📈</span>
                股票筛选报告
            </h1>
            <div class="meta">
                <div class="meta-item">
                    <span>🌏</span>
                    <span>市场: {market}</span>
                </div>
                <div class="meta-item">
                    <span>📅</span>
                    <span>{self._start_time.strftime('%Y-%m-%d %H:%M')}</span>
                </div>
                <div class="meta-item">
                    <span>⏱️</span>
                    <span>耗时: {duration:.1f} 秒</span>
                </div>
                <div class="meta-item">
                    <span>✅</span>
                    <span>筛选结果: {len(results)} 只</span>
                </div>
            </div>
        </div>
        
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{len(results)}</div>
                <div class="label">符合条件股票</div>
            </div>
            <div class="stat-card">
                <div class="value">{len(strategy_stats)}</div>
                <div class="label">命中策略数</div>
            </div>
            <div class="stat-card">
                <div class="value">{max(strategy_stats.values()) if strategy_stats else 0}</div>
                <div class="label">最高命中次数</div>
            </div>
            <div class="stat-card">
                <div class="value">{duration:.0f}s</div>
                <div class="label">分析耗时</div>
            </div>
        </div>
        
        <!-- Strategy Tags -->
        <div class="strategy-tags">
            {''.join([f'<div class="strategy-tag"><span>{self.STRATEGY_NAMES.get(s, s)}</span><span class="count">{c}</span></div>' for s, c in sorted(strategy_stats.items(), key=lambda x: -x[1])])}
        </div>
        
        <!-- Stock Cards -->
        <div class="stocks-grid">
            {''.join([self._build_stock_card(r, market) for r in results])}
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>报告生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 5px; font-size: 0.8rem;">本报告仅供参考，不构成投资建议</p>
        </div>
    </div>
</body>
</html>'''
    
    def _build_stock_card(self, result: Dict[str, Any], market: str) -> str:
        """构建单只股票的 HTML 卡片"""
        info = result.get('info', {})
        symbol = result.get('symbol', '')
        strategies = result.get('strategies', [])
        ai_analysis = result.get('ai_analysis')
        news = result.get('news', [])  # 获取新闻数据
        
        # 格式化数据
        market_cap = info.get('marketCap')
        market_cap_str = f"{market_cap / 1e8:.2f} 亿" if isinstance(market_cap, (int, float)) else "N/A"
        
        pe_ratio = info.get('trailingPE')
        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        
        volume = info.get('volume')
        volume_str = f"{volume / 1e4:.1f} 万" if isinstance(volume, (int, float)) else "N/A"
        
        change_52w = info.get('52WeekChange')
        change_class = ''
        if isinstance(change_52w, (int, float)):
            if change_52w > 0:
                change_class = 'positive'
            elif change_52w < 0:
                change_class = 'negative'
        change_str = f"{change_52w * 100:.1f}%" if isinstance(change_52w, (int, float)) else "N/A"
        
        # 提取评分
        tech_score = 0
        fund_score = 0
        total_score = 0
        if ai_analysis:
            summary = ai_analysis.get('summary', '')
            import re
            tech_match = re.search(r'技术面\s*(\d+)/10', summary)
            fund_match = re.search(r'基本面\s*(\d+)/10', summary)
            total_match = re.search(r'综合\s*(\d+(?:\.\d+)?)/10', summary)
            if tech_match:
                tech_score = int(tech_match.group(1))
            if fund_match:
                fund_score = int(fund_match.group(1))
            if total_match:
                total_score = float(total_match.group(1))
        
        def get_score_class(score):
            if score >= 7:
                return 'score-high'
            elif score >= 5:
                return 'score-medium'
            return 'score-low'
        
        return f'''
        <div class="stock-card">
            <div class="stock-header">
                <h3>{info.get('longName', symbol)}</h3>
                <span class="symbol">{symbol.replace('.HK', '') if market.upper() == 'HK' else symbol}</span>
                <div class="stock-strategies">
                    {''.join([f'<span class="badge">{self.STRATEGY_NAMES.get(s, s)}</span>' for s in strategies])}
                </div>
            </div>
            <div class="stock-body">
                <div class="stock-info">
                    <div class="stock-info-item">
                        <span class="label">行业</span>
                        <span class="value">{info.get('sector', 'N/A')}</span>
                    </div>
                    <div class="stock-info-item">
                        <span class="label">市值</span>
                        <span class="value">{market_cap_str}</span>
                    </div>
                    <div class="stock-info-item">
                        <span class="label">成交量</span>
                        <span class="value">{volume_str}</span>
                    </div>
                    <div class="stock-info-item">
                        <span class="label">市盈率</span>
                        <span class="value">{pe_ratio_str}</span>
                    </div>
                    <div class="stock-info-item">
                        <span class="label">52周涨跌</span>
                        <span class="value {change_class}">{change_str}</span>
                    </div>
                    <div class="stock-info-item">
                        <span class="label">网站</span>
                        <span class="value" style="font-size: 0.8rem;">
                            {f'<a href="{info.get("website", "#")}" target="_blank" style="color: var(--primary);">访问</a>' if info.get('website') else 'N/A'}
                        </span>
                    </div>
                </div>
                
                {self._build_news_section(news) if news else ''}
                
                {self._build_score_section(tech_score, fund_score, total_score) if ai_analysis else ''}
                
                {self._build_ai_section(ai_analysis) if ai_analysis else '<div class="ai-analysis"><p>AI 分析未完成</p></div>'}
            </div>
        </div>'''
    
    def _build_score_section(self, tech_score: int, fund_score: int, total_score: float) -> str:
        """构建评分显示区域"""
        def get_score_class(score):
            if score >= 7:
                return 'score-high'
            elif score >= 5:
                return 'score-medium'
            return 'score-low'
        
        if tech_score == 0 and fund_score == 0:
            return ''
        
        return f'''
        <div class="score-display">
            <div class="score-item">
                <div class="score {get_score_class(tech_score)}">{tech_score}/10</div>
                <div class="label">技术面</div>
            </div>
            <div class="score-item">
                <div class="score {get_score_class(fund_score)}">{fund_score}/10</div>
                <div class="label">基本面</div>
            </div>
            <div class="score-item">
                <div class="score {get_score_class(total_score)}">{total_score}/10</div>
                <div class="label">综合</div>
            </div>
        </div>'''
    
    def _build_news_section(self, news: list) -> str:
        """构建新闻显示区域"""
        if not news:
            return ''
        
        news_items = []
        for item in news[:5]:  # 最多显示5条
            title = item.get('title', 'N/A')
            link = item.get('link', '#')
            published = item.get('published', '')
            publisher = item.get('publisher', '')
            
            news_items.append(f'''
            <div class="news-item">
                <a href="{link}" target="_blank" class="news-title">{title[:80]}{'...' if len(title) > 80 else ''}</a>
                <div class="news-meta">
                    <span class="news-date">{published}</span>
                    {f'<span class="news-source">{publisher}</span>' if publisher else ''}
                </div>
            </div>''')
        
        return f'''
        <div class="news-section">
            <h4>📰 近期新闻</h4>
            <div class="news-list">
                {''.join(news_items)}
            </div>
        </div>'''
    
    def _build_ai_section(self, ai_analysis: Dict[str, Any]) -> str:
        """构建 AI 分析区域"""
        if not ai_analysis:
            return ''
        
        summary = ai_analysis.get('summary', 'N/A')
        model = ai_analysis.get('model_used', 'N/A')
        
        # 截取前 500 字符
        if len(summary) > 800:
            summary = summary[:800] + '...'
        
        return f'''
        <div class="ai-analysis">
            <h4>🤖 AI 综合分析</h4>
            <div class="content">{summary}</div>
            <div class="model">模型: {model}</div>
        </div>'''
    
    def get_filename(self) -> str:
        """获取输出文件名"""
        return f"{self.base_filename}.txt"
    
    def get_html_filename(self) -> str:
        """获取 HTML 文件名"""
        return f"{self.base_filename}.html"
    
    def exists(self) -> bool:
        """检查文件是否存在"""
        return os.path.exists(f"{self.base_filename}.txt")
    
    def read_content(self) -> str:
        """读取 TXT 文件内容"""
        try:
            with open(f"{self.base_filename}.txt", 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取报告文件时发生错误: {e}")
            return ""