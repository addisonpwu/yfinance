"""
报告写入器

负责报告文件的实时输出和格式化，支持 TXT 和 HTML 格式
采用现代化深色终端主题风格
"""
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import re
import logging


class ReportWriter:
    """报告写入器，支持 TXT 和 HTML 格式"""
    
    # 报告输出目录
    REPORT_DIR = "reports"
    
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
        'market_regime': '市场状态',
        'obv_boll_divergence': 'OBV底背离+BOLL超卖',
        'launch_capture': '启动捕捉策略'
    }
    
    # 策略颜色映射
    STRATEGY_COLORS = {
        'momentum_breakout': '#10b981',      # 绿色
        'volatility_squeeze': '#8b5cf6',     # 紫色
        'accumulation_acceleration': '#f59e0b',  # 橙色
        'signal_scorer': '#3b82f6',          # 蓝色
        'market_regime': '#6b7280',          # 灰色
        'obv_boll_divergence': '#3b82f6',    # 蓝色
        'launch_capture': '#f59e0b'          # 橙色
    }
    
    # AI 提供商颜色映射
    PROVIDER_COLORS = {
        'iflow': '#3b82f6',      # 蓝色
        'nvidia': '#10b981',     # 绿色
        'gemini': '#8b5cf6',     # 紫色
        'IFLOW': '#3b82f6',
        'NVIDIA': '#10b981',
        'GEMINI': '#8b5cf6'
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
        
        # 确保 reports 目录存在
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        
        # 报告文件路径（在 reports 目录下）
        self.base_filename = os.path.join(self.REPORT_DIR, raw_filename)
        self.market = market
        self.output_format = output_format
        self._lock = threading.Lock()
        self._initialized = False
        self._results: List[Dict[str, Any]] = []
        self._start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
    
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
        """写入 TXT 格式的股票信息 - 结构化展示"""
        info = result.get('info', {})
        symbol = result.get('symbol', '')
        strategies = result.get('strategies', [])
        ai_analysis = result.get('ai_analysis')
        news = result.get('news', [])
        
        # 格式化字段
        market_cap = info.get('marketCap')
        market_cap_str = f"{market_cap / 1e8:.2f} 亿" if isinstance(market_cap, (int, float)) else "N/A"
        
        pe_ratio = info.get('trailingPE')
        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        
        float_shares = info.get('floatShares')
        float_shares_str = f"{float_shares / 1e8:.2f} 亿股" if isinstance(float_shares, (int, float)) else "N/A"
        
        volume = info.get('volume')
        volume_str = f"{volume / 1e4:.1f} 万" if isinstance(volume, (int, float)) else "N/A"
        
        # 52周涨跌
        change_52w = info.get('52WeekChange')
        change_symbol = "📈" if isinstance(change_52w, (int, float)) and change_52w > 0 else "📉" if isinstance(change_52w, (int, float)) else "➖"
        change_str = f"{change_52w * 100:+.1f}%" if isinstance(change_52w, (int, float)) else "N/A"
        
        # 提取 AI 分析关键信息
        ai_summary = ai_analysis.get('summary', '') if ai_analysis else ''
        direction, confidence, tech_score, fund_score, total_score = self._parse_ai_summary(ai_summary)
        
        txt_filename = f"{self.base_filename}.txt"
        try:
            with open(txt_filename, 'a', encoding='utf-8') as f:
                # 标题行
                f.write(f"\n{'═' * 60}\n")
                f.write(f"  ✅ {info.get('longName', symbol)}\n")
                f.write(f"     📌 {symbol} | {info.get('sector', 'N/A')}\n")
                f.write(f"{'═' * 60}\n")
                
                # 策略标签
                strategy_names = [self.STRATEGY_NAMES.get(s, s) for s in strategies]
                f.write(f"  🎯 命中策略: {' | '.join(strategy_names)}\n")
                f.write(f"{'─' * 60}\n")
                
                # 关键数据 - 两列布局
                f.write(f"  📊 基本数据\n")
                f.write(f"  ├─ 市值: {market_cap_str:>12}  │  市盈率: {pe_ratio_str:>10}\n")
                f.write(f"  ├─ 成交量: {volume_str:>10}  │  流通股: {float_shares_str:>10}\n")
                f.write(f"  └─ 52周涨跌: {change_str:>8} {change_symbol}\n")
                
                # AI 分析摘要
                if ai_analysis:
                    f.write(f"{'─' * 60}\n")
                    f.write(f"  🤖 AI 分析摘要\n")
                    if direction:
                        direction_icon = "🟢" if direction == "看涨" else "🔴" if direction == "看跌" else "🟡"
                        f.write(f"  ├─ 方向: {direction_icon} {direction} (置信度: {confidence})\n")
                    if total_score:
                        f.write(f"  ├─ 综合评分: {total_score}/10")
                        if tech_score:
                            f.write(f" (技术: {tech_score}/10, 基本: {fund_score}/10)")
                        f.write("\n")
                    f.write(f"  └─ 模型: {ai_analysis.get('model_used', 'N/A')}\n")
                
                # 近期新闻
                if news:
                    f.write(f"{'─' * 60}\n")
                    f.write(f"  📰 近期新闻 (最近 {len(news[:3])} 条)\n")
                    for i, item in enumerate(news[:3], 1):
                        title = item.get('title', 'N/A')[:50]
                        date = item.get('published', '')
                        f.write(f"     {i}. [{date}] {title}{'...' if len(item.get('title', '')) > 50 else ''}\n")
                
                f.write(f"{'═' * 60}\n")
                
        except Exception as e:
            print(f"写入 TXT 报告时出错: {e}")
    
    def _parse_ai_summary(self, summary: str) -> tuple:
        """解析 AI 分析摘要，提取关键信息"""
        direction = ''
        confidence = ''
        tech_score = 0
        fund_score = 0
        total_score = 0
        
        if not summary:
            return direction, confidence, tech_score, fund_score, total_score
        
        # 提取方向
        dir_match = re.search(r'方向:\s*(看涨|看跌|中性)', summary)
        if dir_match:
            direction = dir_match.group(1)
        
        # 提取置信度
        conf_match = re.search(r'置信度:\s*(\d+%)', summary)
        if conf_match:
            confidence = conf_match.group(1)
        
        # 提取评分
        tech_match = re.search(r'技术面[评分]*[:：]\s*(\d+(?:\.\d+)?)/10', summary)
        if tech_match:
            tech_score = float(tech_match.group(1))
        
        fund_match = re.search(r'基本面[评分]*[:：]\s*(\d+(?:\.\d+)?)/10', summary)
        if fund_match:
            fund_score = float(fund_match.group(1))
        
        total_match = re.search(r'综合[评分]*[:：]\s*(\d+(?:\.\d+)?)/10', summary)
        if total_match:
            total_score = float(total_match.group(1))
        
        return direction, confidence, tech_score, fund_score, total_score
    
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
        """写入 TXT 格式的摘要 - 结构化展示"""
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
            
            formatted_stocks.append({
                'display': f"{exchange_name}:{symbol}",
                'name': long_name,
                'strategies': stock.get('strategies', [])
            })
        
        txt_filename = f"{self.base_filename}.txt"
        try:
            with open(txt_filename, 'a', encoding='utf-8') as f:
                f.write(f"\n{'═' * 60}\n")
                f.write(f"  📋 筛选摘要\n")
                f.write(f"{'═' * 60}\n")
                f.write(f"  📊 统计: 共筛选出 {len(results)} 只股票\n")
                f.write(f"  📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'─' * 60}\n")
                f.write(f"  📝 股票列表 (便于复制到交易软件):\n")
                f.write(f"     {', '.join([s['display'] for s in formatted_stocks])}\n")
                f.write(f"{'─' * 60}\n")
                f.write(f"  📈 详细列表:\n")
                for i, s in enumerate(formatted_stocks, 1):
                    strategy_names = [self.STRATEGY_NAMES.get(st, st) for st in s['strategies']]
                    f.write(f"     {i:2d}. {s['display']:15} - {s['name'][:20]} [{', '.join(strategy_names)}]\n")
                f.write(f"{'═' * 60}\n")
                f.write(f"  ⚠️  本报告仅供参考，不构成投资建议\n")
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
        except Exception as e:
            self.logger.debug(f"pdfkit PDF生成失败: {e}")
    
    def _build_html(self, results: List[Dict[str, Any]], market: str) -> str:
        """构建完整的 HTML 报告 - 现代化深色终端主题"""
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()
        
        # 按策略分组统计
        strategy_stats = {}
        for r in results:
            for s in r.get('strategies', []):
                strategy_stats[s] = strategy_stats.get(s, 0) + 1
        
        # 计算看涨共识
        bullish_count = 0
        total_direction_count = 0
        for r in results:
            ai_analysis = r.get('ai_analysis')
            if ai_analysis:
                summary = ai_analysis.get('summary', '')
                if '看涨' in summary:
                    bullish_count += 1
                total_direction_count += 1
        
        bullish_consensus = int(bullish_count / total_direction_count * 100) if total_direction_count > 0 else 0
        
        # 获取策略名称列表
        strategy_names_list = [self.STRATEGY_NAMES.get(s, s) for s in strategy_stats.keys()]
        
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>專業股票篩選報告 | {market} Market {self._start_time.strftime('%Y-%m-%d')}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    }},
                    colors: {{
                        terminal: {{
                            bg: '#0a0e1a',
                            card: '#111827',
                            border: '#1f2937',
                            accent: '#3b82f6',
                            success: '#10b981',
                            warning: '#f59e0b',
                            danger: '#ef4444',
                            text: '#f3f4f6',
                            muted: '#9ca3af',
                            code: '#1e1e2e'
                        }}
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{
            background-color: #0a0e1a;
            color: #f3f4f6;
        }}
        .gradient-border {{
            position: relative;
            background: #111827;
            border-radius: 12px;
        }}
        .gradient-border::before {{
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 12px;
            padding: 1px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
        }}
        .ai-provider-section {{
            border-left: 3px solid;
            background: #0f172a;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
        }}
        .ai-provider-header {{
            padding: 12px 16px;
            font-weight: 600;
            font-size: 14px;
            border-radius: 0 8px 0 0;
        }}
        .ai-provider-content {{
            padding: 16px;
            font-size: 13px;
            line-height: 1.7;
            color: #d1d5db;
            white-space: pre-wrap;
            font-family: 'JetBrains Mono', monospace;
            background: #1e1e2e;
            border-radius: 0 0 8px 0;
            max-height: 400px;
            overflow-y: auto;
        }}
        .provider-iflow {{ border-color: #3b82f6; }}
        .provider-iflow .ai-provider-header {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
        .provider-nvidia {{ border-color: #10b981; }}
        .provider-nvidia .ai-provider-header {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
        .provider-gemini {{ border-color: #8b5cf6; }}
        .provider-gemini .ai-provider-header {{ background: rgba(139, 92, 246, 0.15); color: #a78bfa; }}
        
        .toggle-btn {{
            transition: all 0.3s ease;
        }}
        .toggle-btn:hover {{
            background: rgba(59, 130, 246, 0.2);
        }}
        .hidden-content {{
            display: none;
        }}
        .stock-card {{
            background: #111827;
            border: 1px solid #1f2937;
            transition: all 0.3s ease;
        }}
        .stock-card:hover {{
            border-color: #374151;
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5);
        }}
        .score-ring {{
            transform: rotate(-90deg);
        }}
        .metric-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        }}
        .badge-bullish {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-bearish {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-neutral {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: #0f172a; }}
        ::-webkit-scrollbar-thumb {{ background: #374151; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #4b5563; }}
        
        .section-title {{
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}
    </style>
</head>
<body class="antialiased min-h-screen">

    <!-- Header -->
    <header class="sticky top-0 z-50 bg-terminal-bg/95 backdrop-blur-md border-b border-terminal-border">
        <div class="max-w-7xl mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-white tracking-tight">股票篩選報告</h1>
                        <p class="text-sm text-terminal-muted">{market} Market • {self._start_time.strftime('%Y-%m-%d')} • {', '.join(strategy_names_list[:2])}</p>
                    </div>
                </div>
                <div class="flex items-center gap-6 text-sm">
                    <div class="text-right">
                        <div class="text-terminal-muted text-xs uppercase tracking-wider">篩選耗時</div>
                        <div class="font-mono text-terminal-accent font-semibold">{duration:.1f}s</div>
                    </div>
                    <div class="text-right">
                        <div class="text-terminal-muted text-xs uppercase tracking-wider">符合條件</div>
                        <div class="font-mono text-terminal-success font-semibold">{len(results)} 只股票</div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8">

        <!-- Summary Stats -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div class="gradient-border p-5">
                <div class="text-terminal-muted text-xs uppercase tracking-wider mb-2">命中策略數</div>
                <div class="text-2xl font-bold text-white">{len(strategy_stats)}</div>
                <div class="text-xs text-terminal-muted mt-1">{', '.join(strategy_names_list[:2])}</div>
            </div>
            <div class="gradient-border p-5">
                <div class="text-terminal-muted text-xs uppercase tracking-wider mb-2">最高命中次數</div>
                <div class="text-2xl font-bold text-terminal-accent">{max(strategy_stats.values()) if strategy_stats else 0}</div>
                <div class="text-xs text-terminal-muted mt-1">策略覆蓋度</div>
            </div>
            <div class="gradient-border p-5">
                <div class="text-terminal-muted text-xs uppercase tracking-wider mb-2">看漲共識</div>
                <div class="text-2xl font-bold text-terminal-success">{bullish_consensus}%</div>
                <div class="w-full bg-gray-800 rounded-full h-1.5 mt-2">
                    <div class="bg-terminal-success h-1.5 rounded-full" style="width: {bullish_consensus}%"></div>
                </div>
            </div>
            <div class="gradient-border p-5">
                <div class="text-terminal-muted text-xs uppercase tracking-wider mb-2">篩選耗時</div>
                <div class="text-2xl font-bold text-terminal-warning">{duration:.1f}s</div>
                <div class="text-xs text-terminal-muted mt-1">分析完成</div>
            </div>
        </div>

        <!-- Strategy Tags -->
        <div class="mb-6 flex items-center gap-3 flex-wrap">
            {''.join([f'<div class="px-4 py-2 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 text-sm font-medium"><span class="inline-block w-2 h-2 rounded-full bg-blue-500 mr-2 animate-pulse"></span>{self.STRATEGY_NAMES.get(s, s)}<span class="ml-2 px-2 py-0.5 rounded bg-blue-500/20 text-xs">{c} 命中</span></div>' for s, c in sorted(strategy_stats.items(), key=lambda x: -x[1])])}
        </div>

        <!-- Stock Cards Container -->
        <div class="space-y-6" id="stocks-container">
            {''.join([self._build_stock_card(r, market, i) for i, r in enumerate(results)])}
        </div>

        <!-- Footer -->
        <div class="mt-12 text-center text-terminal-muted text-sm">
            <p>報告生成時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="mt-2 text-xs">⚠️ 本報告僅供參考，不構成投資建議</p>
        </div>
    </main>

    <script>
        function toggleAI(id) {{
            const content = document.getElementById(id);
            const icon = document.getElementById('icon-' + id);
            if (content && icon) {{
                content.classList.toggle('hidden-content');
                icon.classList.toggle('rotate-180');
            }}
        }}
    </script>
</body>
</html>'''
    
    def _build_stock_card(self, result: Dict[str, Any], market: str, index: int = 0) -> str:
        """构建单只股票的 HTML 卡片 - 深色主题风格"""
        info = result.get('info', {})
        symbol = result.get('symbol', '')
        strategies = result.get('strategies', [])
        ai_analysis = result.get('ai_analysis')
        news = result.get('news', [])
        technical_indicators = result.get('technical_indicators', {})
        
        # 格式化数据
        market_cap = info.get('marketCap')
        market_cap_str = f"{market_cap / 1e8:.2f}億" if isinstance(market_cap, (int, float)) else "N/A"
        
        pe_ratio = info.get('trailingPE')
        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        
        volume = info.get('volume')
        volume_str = f"{volume / 1e4:.1f}萬" if isinstance(volume, (int, float)) else "N/A"
        
        change_52w = info.get('52WeekChange')
        change_class = 'text-green-400' if isinstance(change_52w, (int, float)) and change_52w > 0 else 'text-red-400' if isinstance(change_52w, (int, float)) else 'text-gray-400'
        change_str = f"{change_52w * 100:+.1f}%" if isinstance(change_52w, (int, float)) else "N/A"
        
        # 解析 AI 分析
        ai_summary = ai_analysis.get('summary', '') if ai_analysis else ''
        direction, confidence, tech_score, fund_score, total_score = self._parse_ai_summary(ai_summary)
        
        # 确定方向样式
        direction_class = 'badge-neutral'
        direction_icon = '➖'
        if direction == '看涨':
            direction_class = 'badge-bullish'
            direction_icon = '📈'
        elif direction == '看跌':
            direction_class = 'badge-bearish'
            direction_icon = '📉'
        
        # 评分颜色
        def get_score_color(score):
            if score >= 7:
                return '#10b981'  # green
            elif score >= 5:
                return '#f59e0b'  # amber
            return '#ef4444'  # red
        
        # 评分条计算
        tech_color = get_score_color(tech_score) if tech_score else '#374151'
        fund_color = get_score_color(fund_score) if fund_score else '#374151'
        total_color = get_score_color(total_score) if total_score else '#374151'
        
        # SVG 环形进度
        def calc_stroke_offset(score):
            # 圆周长 = 2 * π * 35 ≈ 220
            return 220 - (score / 10 * 220) if score else 220
        
        tech_offset = calc_stroke_offset(tech_score)
        fund_offset = calc_stroke_offset(fund_score)
        total_offset = calc_stroke_offset(total_score)
        
        # 策略标签
        strategy_badges = ''.join([f'<span class="px-2 py-1 rounded text-xs font-medium" style="background: {self.STRATEGY_COLORS.get(s, "#3b82f6")}20; color: {self.STRATEGY_COLORS.get(s, "#3b82f6")}; border: 1px solid {self.STRATEGY_COLORS.get(s, "#3b82f6")}40;">{self.STRATEGY_NAMES.get(s, s)}</span>' for s in strategies])
        
        # 卡片头部渐变
        header_gradient = 'from-green-500/5' if direction == '看涨' else 'from-red-500/5' if direction == '看跌' else 'from-amber-500/5'
        
        # 生成唯一ID
        card_id = f"stock-{index}"
        ai_id = f"ai-{index}"
        
        # 构建技术指标 HTML
        tech_html = self._build_technical_indicators_html_dark(technical_indicators) if technical_indicators else ''
        
        # 构建 AI 分析 HTML
        ai_html = self._build_ai_section_html_dark(ai_analysis, ai_id) if ai_analysis else ''
        
        # 构建新闻 HTML
        news_html = self._build_news_section_html_dark(news) if news else ''
        
        return f'''
            <article class="stock-card rounded-2xl overflow-hidden" data-symbol="{symbol}">
                <!-- Card Header -->
                <div class="p-6 border-b border-terminal-border bg-gradient-to-r {header_gradient} to-transparent">
                    <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                        <div class="flex items-start gap-4">
                            <div class="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center flex-shrink-0">
                                <span class="font-mono font-bold text-blue-400 text-lg">{symbol.replace('.HK', '') if market.upper() == 'HK' else symbol}</span>
                            </div>
                            <div>
                                <h2 class="text-xl font-bold text-white">{info.get('longName', symbol)}</h2>
                                <div class="flex items-center gap-3 mt-1 text-sm text-terminal-muted">
                                    <span>{info.get('sector', 'N/A')}</span>
                                    <span class="w-1 h-1 rounded-full bg-gray-600"></span>
                                    <span>市值 {market_cap_str}</span>
                                </div>
                            </div>
                        </div>
                        <div class="flex items-center gap-3">
                            <span class="{direction_class} px-4 py-2 rounded-lg text-sm">
                                {direction_icon} {direction} {confidence}
                            </span>
                        </div>
                    </div>
                    <div class="flex flex-wrap gap-2 mt-3">
                        {strategy_badges}
                    </div>
                </div>

                <div class="p-6">
                    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <!-- Left: Scores & Metrics -->
                        <div class="lg:col-span-1 space-y-4">
                            <!-- Score Rings -->
                            <div class="grid grid-cols-3 gap-3">
                                <div class="text-center">
                                    <div class="relative w-20 h-20 mx-auto mb-2">
                                        <svg class="w-20 h-20 score-ring">
                                            <circle cx="40" cy="40" r="35" stroke="#1f2937" stroke-width="6" fill="none"/>
                                            <circle cx="40" cy="40" r="35" stroke="{tech_color}" stroke-width="6" fill="none" 
                                                stroke-dasharray="220" stroke-dashoffset="{tech_offset}" stroke-linecap="round"/>
                                        </svg>
                                        <span class="absolute inset-0 flex items-center justify-center text-lg font-bold" style="color: {tech_color}">{tech_score:.1f}</span>
                                    </div>
                                    <div class="text-xs text-terminal-muted">技術面</div>
                                </div>
                                <div class="text-center">
                                    <div class="relative w-20 h-20 mx-auto mb-2">
                                        <svg class="w-20 h-20 score-ring">
                                            <circle cx="40" cy="40" r="35" stroke="#1f2937" stroke-width="6" fill="none"/>
                                            <circle cx="40" cy="40" r="35" stroke="{fund_color}" stroke-width="6" fill="none" 
                                                stroke-dasharray="220" stroke-dashoffset="{fund_offset}" stroke-linecap="round"/>
                                        </svg>
                                        <span class="absolute inset-0 flex items-center justify-center text-lg font-bold" style="color: {fund_color}">{fund_score:.1f}</span>
                                    </div>
                                    <div class="text-xs text-terminal-muted">基本面</div>
                                </div>
                                <div class="text-center">
                                    <div class="relative w-20 h-20 mx-auto mb-2">
                                        <svg class="w-20 h-20 score-ring">
                                            <circle cx="40" cy="40" r="35" stroke="#1f2937" stroke-width="6" fill="none"/>
                                            <circle cx="40" cy="40" r="35" stroke="{total_color}" stroke-width="6" fill="none" 
                                                stroke-dasharray="220" stroke-dashoffset="{total_offset}" stroke-linecap="round"/>
                                        </svg>
                                        <span class="absolute inset-0 flex items-center justify-center text-lg font-bold" style="color: {total_color}">{total_score:.1f}</span>
                                    </div>
                                    <div class="text-xs text-terminal-muted">綜合</div>
                                </div>
                            </div>

                            <!-- Quick Metrics -->
                            <div class="space-y-2 pt-4 border-t border-terminal-border">
                                <div class="flex justify-between items-center py-2">
                                    <span class="text-sm text-terminal-muted">市盈率</span>
                                    <span class="font-mono font-medium">{pe_ratio_str}</span>
                                </div>
                                <div class="flex justify-between items-center py-2">
                                    <span class="text-sm text-terminal-muted">52周漲跌</span>
                                    <span class="font-mono font-medium {change_class}">{change_str}</span>
                                </div>
                                <div class="flex justify-between items-center py-2">
                                    <span class="text-sm text-terminal-muted">成交量</span>
                                    <span class="font-mono font-medium">{volume_str}</span>
                                </div>
                            </div>

                            {tech_html}
                        </div>

                        <!-- Right: AI Analysis -->
                        <div class="lg:col-span-2">
                            {ai_html}
                            {news_html}
                        </div>
                    </div>
                </div>
            </article>'''
    
    def _build_technical_indicators_html_dark(self, indicators: Dict[str, Any]) -> str:
        """构建技术指标显示区域 - 深色主题"""
        if not indicators:
            return ''
        
        items = []
        
        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            status = indicators.get('rsi_status', 'normal')
            status_text = '超買' if status == 'overbought' else '超賣' if status == 'oversold' else '正常'
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">RSI(14)</span>
                <span class="font-mono">{rsi:.1f} {status_text}</span>
            </div>''')
        
        # MACD
        if 'macd' in indicators:
            macd = indicators['macd']
            macd_status = indicators.get('macd_status', 'death_cross')
            status_text = '金叉' if macd_status == 'golden_cross' else '死叉'
            status_class = 'text-green-400' if macd_status == 'golden_cross' else 'text-red-400'
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">MACD</span>
                <span class="font-mono {status_class}">{macd:.4f} {status_text}</span>
            </div>''')
        
        # 布林带位置
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            bb_status = indicators.get('bb_status', 'in_band')
            status_text = '上軌外' if bb_status == 'above_upper' else '下軌外' if bb_status == 'below_lower' else '帶內'
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">布林帶</span>
                <span class="font-mono">{bb_pos:.2f} {status_text}</span>
            </div>''')
        
        # 均线位置
        if 'ma_score' in indicators:
            ma_score = indicators['ma_score']
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">均線位置</span>
                <span class="font-mono">{ma_score}/5</span>
            </div>''')
        
        # ATR
        if 'atr_pct' in indicators:
            atr_pct = indicators['atr_pct']
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">ATR%</span>
                <span class="font-mono">{atr_pct:.2f}%</span>
            </div>''')
        
        if not items:
            return ''
        
        return f'''
                            <div class="pt-4 border-t border-terminal-border">
                                <h4 class="text-xs font-semibold text-terminal-muted uppercase tracking-wider mb-3">技術指標</h4>
                                <div class="space-y-2 text-sm">
                                    {''.join(items)}
                                </div>
                            </div>'''
    
    def _build_news_section_html_dark(self, news: list) -> str:
        """构建新闻显示区域 - 深色主题"""
        if not news:
            return ''
        
        news_items = []
        for item in news[:3]:  # 显示3条新闻
            title = item.get('title', 'N/A')
            link = item.get('link', '#')
            published = item.get('published', '')
            
            news_items.append(f'''
                                <div class="py-2 border-b border-terminal-border/50 last:border-0">
                                    <a href="{link}" target="_blank" class="text-sm text-terminal-text hover:text-terminal-accent transition-colors">{title[:60]}{'...' if len(title) > 60 else ''}</a>
                                    <div class="text-xs text-terminal-muted mt-1">{published}</div>
                                </div>''')
        
        return f'''
                            <div class="mt-4 p-4 rounded-xl bg-blue-500/5 border border-blue-500/20">
                                <h4 class="text-sm font-semibold text-blue-400 mb-3 flex items-center gap-2">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"/>
                                    </svg>
                                    近期新聞 ({len(news)} 條)
                                </h4>
                                <div class="space-y-1">
                                    {''.join(news_items)}
                                </div>
                            </div>'''
    
    def _build_ai_section_html_dark(self, ai_analysis: Dict[str, Any], ai_id: str) -> str:
        """构建可折叠的 AI 分析区域 - 深色主题，支持多提供商"""
        if not ai_analysis:
            return ''
        
        summary = ai_analysis.get('summary', 'N/A')
        model = ai_analysis.get('model_used', 'N/A')
        
        # 检测是否为多提供商分析
        is_multi_provider = '---' in summary and ('IFLOW' in summary.upper() or 'NVIDIA' in summary.upper() or 'GEMINI' in summary.upper())
        
        if is_multi_provider:
            # 解析多提供商分析
            provider_sections = self._parse_multi_provider_summary(summary)
            sections_html = ''
            
            for provider, content in provider_sections:
                color = self.PROVIDER_COLORS.get(provider.lower(), '#3b82f6')
                sections_html += f'''
                                <div class="ai-provider-section provider-{provider.lower()}">
                                    <div class="ai-provider-header flex items-center justify-between">
                                        <span>{provider.upper()}</span>
                                    </div>
                                    <div class="ai-provider-content">{content}</div>
                                </div>'''
            
            return f'''
                            <button onclick="toggleAI('{ai_id}')" class="toggle-btn w-full flex items-center justify-between p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:border-blue-500/40 mb-4">
                                <div class="flex items-center gap-3">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                                    </svg>
                                    <span class="font-semibold">AI 分析詳情 (多提供商)</span>
                                </div>
                                <svg id="icon-{ai_id}" class="w-5 h-5 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                                </svg>
                            </button>

                            <div id="{ai_id}" class="hidden-content space-y-3">
                                {sections_html}
                            </div>'''
        else:
            # 单提供商分析
            return f'''
                            <button onclick="toggleAI('{ai_id}')" class="toggle-btn w-full flex items-center justify-between p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:border-blue-500/40 mb-4">
                                <div class="flex items-center gap-3">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                                    </svg>
                                    <span class="font-semibold">AI 分析詳情</span>
                                    <span class="px-2 py-0.5 rounded bg-blue-500/20 text-xs">{model}</span>
                                </div>
                                <svg id="icon-{ai_id}" class="w-5 h-5 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                                </svg>
                            </button>

                            <div id="{ai_id}" class="hidden-content">
                                <div class="ai-provider-section provider-iflow">
                                    <div class="ai-provider-content">{summary}</div>
                                </div>
                            </div>'''
    
    def _parse_multi_provider_summary(self, summary: str) -> List[tuple]:
        """解析多提供商分析摘要，返回 [(provider, content), ...]"""
        import re
        
        # 匹配格式: --- PROVIDER 分析 --- 或 【PROVIDER 多模型共识分析】
        pattern = r'(?:---\s*(IFLOW|NVIDIA|GEMINI)\s*分析\s*---)|(?:【(IFLOW|NVIDIA|GEMINI)\s*(?:多模型)?[共识分析]+】)'
        parts = re.split(pattern, summary, flags=re.IGNORECASE)
        
        results = []
        
        # 处理分割后的内容
        i = 1
        while i < len(parts):
            # 检查两种捕获组
            provider = parts[i] if parts[i] else (parts[i+1] if i+1 < len(parts) and parts[i+1] else None)
            if provider:
                provider = provider.strip().upper()
                # 找到下一个提供商标记之前的内容
                content_start = i + 2 if parts[i] else i + 3
                content = parts[content_start] if content_start < len(parts) else ''
                
                # 截取到下一个提供商标记
                next_match = re.search(pattern, content, flags=re.IGNORECASE)
                if next_match:
                    content = content[:next_match.start()]
                
                content = content.strip()
                if provider and content:
                    results.append((provider, content))
            i += 3
        
        # 如果解析失败，尝试简单分割
        if not results:
            for provider in ['IFLOW', 'NVIDIA', 'GEMINI']:
                marker = f'--- {provider} 分析 ---'
                if marker in summary:
                    parts = summary.split(marker)
                    if len(parts) > 1:
                        content = parts[1].split('---')[0].strip()
                        results.append((provider, content))
        
        # 如果还是失败，返回原始内容
        if not results:
            results.append(('AI', summary))
        
        return results
    
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
