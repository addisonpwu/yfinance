"""
报告写入器

负责报告文件的实时输出和格式化，支持 JSON 和 HTML 格式
- JSON: 输出筛选的股票代码列表
- HTML: 输出详细分析报告
- 增量保存: 每只股票分析完成后实时更新 HTML
"""
import threading
import time
import tempfile
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import re
import logging


class ReportWriter:
    """报告写入器，支持 JSON 和 HTML 格式"""
    
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
        'opencode': '#f97316',   # 橙色
        'IFLOW': '#3b82f6',
        'NVIDIA': '#10b981',
        'GEMINI': '#8b5cf6',
        'OPENCODE': '#f97316'
    }
    
    def __init__(self, filename: str = None, market: str = 'HK'):
        """
        初始化报告写入器
        
        Args:
            filename: 输出文件名（可含或不含扩展名）
            market: 市场代码
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
        self._lock = threading.RLock()  # 使用 RLock 支持可重入锁
        self._initialized = False
        self._results: List[Dict[str, Any]] = []
        self._start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        
        # 增量保存相关属性
        self._last_update_time: float = 0.0  # 上次更新时间戳
        self._min_update_interval: float = 0.3  # 最小更新间隔（秒），防抖
        self._is_completed: bool = False  # 是否已完成最终报告
    
    def _generate_basename(self, market: str) -> str:
        """生成默认文件名（不含扩展名）"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        return f"{market.lower()}_stocks_{today_str}"
    
    def initialize(self) -> bool:
        """
        初始化报告写入器
        
        Returns:
            是否成功初始化
        """
        self._initialized = True
        return True
    
    def write_stock_result(self, result: Dict[str, Any]) -> None:
        """
        收集单只股票的分析结果并增量更新 HTML
        
        Args:
            result: 股票分析结果字典
        """
        if not self._initialized:
            return
        
        with self._lock:
            self._results.append(result)
            # 增量更新 HTML（未加锁版本，因为已经在锁内）
            self._incremental_update_html_unlocked()
    
    def _incremental_update_html_unlocked(self) -> None:
        """
        增量更新 HTML 文件（内部方法，调用前必须持有锁）
        
        使用防抖机制避免高频文件 I/O
        """
        # 如果已完成最终报告，不再增量更新
        if self._is_completed:
            return
        
        # 防抖检查
        current_time = time.time()
        time_since_last_update = current_time - self._last_update_time
        
        if time_since_last_update < self._min_update_interval:
            # 间隔太短，跳过本次更新（但保留最新数据）
            return
        
        # 更新时间戳
        self._last_update_time = current_time
        
        try:
            # 重建 HTML
            html_content = self._build_html(self._results, self.market, is_in_progress=True)
            
            # 原子写入
            self._write_html_atomic(html_content)
            
        except Exception as e:
            self.logger.error(f"增量更新 HTML 失败: {e}")
    
    def _write_html_atomic(self, content: str) -> None:
        """
        原子写入 HTML 文件
        
        使用临时文件 + os.replace() 确保 HTML 文件始终完整可读
        
        Args:
            content: HTML 内容
        """
        html_filename = f"{self.base_filename}.html"
        
        try:
            # 创建临时文件
            fd, temp_path = tempfile.mkstemp(
                suffix='.html',
                prefix='report_',
                dir=self.REPORT_DIR
            )
            
            try:
                # 写入内容
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # 原子替换（rename 在同文件系统上是原子的）
                os.replace(temp_path, html_filename)
                
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
                
        except Exception as e:
            self.logger.error(f"原子写入 HTML 失败: {e}")
            # 降级为普通写入
            try:
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as fallback_e:
                self.logger.error(f"降级写入 HTML 也失败: {fallback_e}")
    
    def _parse_ai_summary(self, summary: str) -> tuple:
        """解析 AI 分析摘要，提取关键信息"""
        direction = ''
        confidence = ''
        tech_score = 0
        fund_score = 0
        total_score = 0
        
        if not summary:
            return direction, confidence, tech_score, fund_score, total_score, 0.0, 0.0, 0.0  # P0-1: 添加入场价、止损价、止盈价
        
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
        
        return direction, confidence, tech_score, fund_score, total_score, 0.0, 0.0, 0.0  # P0-1: 添加入场价、止损价、止盈价
    
    def write_summary(self, results: List[Dict[str, Any]], market: str) -> None:
        """
        写入摘要列表并生成最终报告
        
        这是分析完成后的最终确认，生成最终版 HTML（无"分析进行中"提示）
        
        Args:
            results: 股票分析结果列表
            market: 市场代码
        """
        with self._lock:
            # 标记为已完成，停止增量更新
            self._is_completed = True
            
            # 生成最终版 HTML 报告（is_in_progress=False）
            self._generate_html_report(results, market, is_final=True)
    
    def _generate_html_report(self, results: List[Dict[str, Any]], market: str, is_final: bool = False) -> None:
        """生成 HTML 格式的完整报告
        
        Args:
            results: 股票分析结果列表
            market: 市场代码
            is_final: 是否为最终版报告（非增量更新）
        """
        html_filename = f"{self.base_filename}.html"
        
        # 最终版不显示"分析进行中"状态
        html_content = self._build_html(results, market, is_in_progress=not is_final)
        
        try:
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            if is_final:
                print(f"\n📊 HTML 报告已生成: {html_filename}")
                # 尝试生成 PDF（仅在最终版时）
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
    
    def _build_html(self, results: List[Dict[str, Any]], market: str, is_in_progress: bool = False) -> str:
        """构建完整的 HTML 报告 - 现代化深色终端主题
        
        Args:
            results: 股票分析结果列表
            market: 市场代码
            is_in_progress: 是否正在分析中（增量更新模式）
        """
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
        
        # 状态标题和提示
        status_title = "股票篩選報告 (分析中...)" if is_in_progress else "股票篩選報告"
        status_badge = '''
                <div class="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/20 border border-amber-500/30">
                    <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
                    <span class="text-amber-400 text-sm font-medium">分析進行中</span>
                </div>''' if is_in_progress else ''
        
        # 分析中提示条
        progress_banner = '''
        <!-- Progress Banner -->
        <div class="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center gap-4">
            <div class="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
                <svg class="w-5 h-5 text-amber-400 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </div>
            <div>
                <p class="text-amber-400 font-semibold">分析進行中，報告實時更新...</p>
                <p class="text-terminal-muted text-sm">已找到 <span class="text-amber-400 font-bold">''' + str(len(results)) + '''</span> 只符合條件的股票</p>
            </div>
        </div>''' if is_in_progress else ''
        
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
        .provider-opencode {{ border-color: #f97316; }}
        .provider-opencode .ai-provider-header {{ background: rgba(249, 115, 22, 0.15); color: #fb923c; }}
        
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
                        <h1 class="text-xl font-bold text-white tracking-tight">{status_title}</h1>
                        <p class="text-sm text-terminal-muted">{market} Market • {self._start_time.strftime('%Y-%m-%d')} • {', '.join(strategy_names_list[:2])}</p>
                    </div>
                </div>
                <div class="flex items-center gap-6 text-sm">
                    {status_badge}
                    <div class="text-right">
                        <div class="text-terminal-muted text-xs uppercase tracking-wider">篩選耗時</div>
                        <div class="font-mono text-terminal-accent font-semibold">{duration:.1f}s</div>
                    </div>
                    <div class="text-right">
                        <div class="text-terminal-muted text-xs uppercase tracking-wider">符合條件</div>
                        <div class="font-mono text-terminal-success font-semibold">{len(results)} 只股票</div>
                    </div>
                    <!-- P1-4: 全局展开/折叠按钮 -->
                    <div class="flex items-center gap-2">
                        <button onclick="expandAllAI()" class="global-toggle-btn bg-blue-500/20 text-blue-400 border border-blue-500/30">展開全部</button>
                        <button onclick="collapseAllAI()" class="global-toggle-btn bg-gray-500/20 text-gray-400 border border-gray-500/30">折疊全部</button>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8">
        {progress_banner}

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
        
        // P1-4: 全局展开所有 AI 分析
        function expandAllAI() {{
            document.querySelectorAll('[id^="ai-"]').forEach(el => {{
                el.classList.remove('hidden-content');
                const icon = document.getElementById('icon-' + el.id);
                if (icon) icon.classList.add('rotate-180');
            }});
        }}
        
        // P1-4: 全局折叠所有 AI 分析
        function collapseAllAI() {{
            document.querySelectorAll('[id^="ai-"]').forEach(el => {{
                el.classList.add('hidden-content');
                const icon = document.getElementById('icon-' + el.id);
                if (icon) icon.classList.remove('rotate-180');
            }});
        }}
        
        // P2-7: 回到顶部按钮逻辑
        window.addEventListener('scroll', () => {{
            const btn = document.getElementById('back-to-top');
            if (btn) {{
                if (window.scrollY > 300) {{
                    btn.classList.add('visible');
                }} else {{
                    btn.classList.remove('visible');
                }}
            }}
        }});
        
        function scrollToTop() {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
    </script>
    
    <!-- P2-7: 回到顶部按钮 -->
    <div id="back-to-top" class="back-to-top" onclick="scrollToTop()">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"/>
        </svg>
    </div>
</body>
</html>'''
    
    def _build_stock_card(self, result: Dict[str, Any], market: str, index: int = 0) -> str:
        """构建单只股票的 HTML 卡片 - 深色主题风格"""
        info = result.get('info', {})
        symbol = result.get('symbol', '')
        strategies = result.get('strategies', [])
        ai_analysis = result.get('ai_analysis')
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
        direction, confidence_parsed, tech_score, fund_score, total_score, entry_price, stop_loss, take_profit = self._parse_ai_summary(ai_summary)
        
        # 优先使用 ai_analysis 中的 confidence 字段，否则从 summary 解析
        if ai_analysis and 'confidence' in ai_analysis:
            confidence = f"{ai_analysis['confidence']:.0%}"
        else:
            confidence = confidence_parsed
        
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
            return '#ff6b6b'  # P0-3: 更亮的红色，提高低分可读性
        
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
        
        # 构建策略评分明细 HTML
        strategy_details = result.get('strategy_details', [])
        scores_html = self._build_strategy_scores_html(strategy_details) if strategy_details else ''
        
        # 构建新闻区域 HTML
        news_list = result.get('news', [])
        news_html = self._build_news_section_html(news_list) if news_list else ''
        
        # 构建 AI 分析 HTML
        ai_html = self._build_ai_section_html_dark(ai_analysis, ai_id) if ai_analysis else ''
        
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
                            {scores_html}
                        </div>

                        <!-- Right: AI Analysis -->
                        <div class="lg:col-span-2">
                            {ai_html}
                        </div>
                    </div>
                    {news_html}
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
                <span class="font-mono">{rsi:.1f} {status_text} →</span>
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
                <span class="font-mono">{bb_pos:.2f} {status_text} →</span>
            </div>''')
        
        # 均线位置
        if 'ma_score' in indicators:
            ma_score = indicators['ma_score']
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">均線位置</span>
                <span class="font-mono">{ma_score}/5 →</span>
            </div>''')
        
        # ATR
        if 'atr_pct' in indicators:
            atr_pct = indicators['atr_pct']
            items.append(f'''
            <div class="flex justify-between">
                <span class="text-terminal-muted">ATR%</span>
                <span class="font-mono">{atr_pct:.2f}% →</span>
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
    
    def _build_news_section_html(self, news_list: List[Dict]) -> str:
        """构建新闻显示区域 - 深色主题
        
        Args:
            news_list: 新闻列表，每条新闻包含：
                - title: 标题
                - publishTime: 发布时间
                - url: 链接
                - type: 类型 (profit=盈利预告, rating=机构评级)
                - agency: 机构
                - rating: 评级
                - profit: 盈利信息
        
        Returns:
            新闻区域 HTML
        """
        if not news_list:
            return ''
        
        news_items = []
        
        for news in news_list[:5]:  # 最多显示5条
            title = news.get('title', '')
            publish_time = news.get('publishTime', '') or news.get('published', '')
            url = news.get('url', '') or news.get('link', '')
            news_type = news.get('type', '')
            agency = news.get('agency', '')
            rating = news.get('rating', '')
            profit = news.get('profit', '')
            
            # 格式化发布时间
            time_display = publish_time
            if 'T' in publish_time:
                try:
                    # 解析 ISO 格式时间
                    from datetime import datetime
                    dt = datetime.fromisoformat(publish_time.replace('+08:00', '+08:00'))
                    time_display = dt.strftime('%m-%d %H:%M')
                except:
                    time_display = publish_time[:16].replace('T', ' ')
            
            # 构建类型标签
            type_badge = ''
            if news_type == 'profit':
                type_badge = '<span class="px-2 py-0.5 rounded text-xs bg-green-500/20 text-green-400 border border-green-500/30">✅ 盈利預告</span>'
            elif news_type == 'rating':
                type_badge = '<span class="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30">✅ 機構評級</span>'
            
            # 构建额外信息
            extra_info = []
            if agency:
                extra_info.append(f'<span class="text-blue-400">{agency}</span>')
            if rating:
                extra_info.append(f'<span class="text-amber-400">{rating}</span>')
            if profit:
                extra_info.append(f'<span class="text-green-400">{profit}</span>')
            extra_info_str = ' | '.join(extra_info) if extra_info else ''
            
            # 构建新闻条目
            if url:
                news_items.append(f'''
                <a href="{url}" target="_blank" class="block p-3 rounded-lg bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 hover:border-gray-600 transition-all">
                    <div class="flex items-start justify-between gap-3">
                        <div class="flex-1">
                            <div class="flex items-center gap-2 mb-1">
                                {type_badge}
                                <span class="text-xs text-gray-500">{time_display}</span>
                            </div>
                            <p class="text-sm text-gray-300 hover:text-white transition-colors">{title}</p>
                            {f'<p class="text-xs text-gray-500 mt-1">{extra_info_str}</p>' if extra_info_str else ''}
                        </div>
                        <svg class="w-4 h-4 text-gray-500 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                        </svg>
                    </div>
                </a>''')
            else:
                news_items.append(f'''
                <div class="block p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
                    <div class="flex items-start justify-between gap-3">
                        <div class="flex-1">
                            <div class="flex items-center gap-2 mb-1">
                                {type_badge}
                                <span class="text-xs text-gray-500">{time_display}</span>
                            </div>
                            <p class="text-sm text-gray-300">{title}</p>
                            {f'<p class="text-xs text-gray-500 mt-1">{extra_info_str}</p>' if extra_info_str else ''}
                        </div>
                    </div>
                </div>''')
        
        if not news_items:
            return ''
        
        return f'''
                <!-- News Section -->
                <div class="mt-6 pt-6 border-t border-terminal-border">
                    <h4 class="text-sm font-semibold text-terminal-muted uppercase tracking-wider mb-4 flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"/>
                        </svg>
                        近期新聞 ({len(news_items)} 條)
                    </h4>
                    <div class="space-y-2">
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
        is_multi_provider = '---' in summary and ('IFLOW' in summary.upper() or 'NVIDIA' in summary.upper() or 'GEMINI' in summary.upper() or 'OPENCODE' in summary.upper())
        
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
        
        # 只匹配 "--- PROVIDER 分析 ---" 格式（不匹配 "【PROVIDER ...】" 格式）
        # 这样可以避免错误分割 NVIDIA 多模型共识分析的内容
        pattern = r'---\s*(IFLOW|NVIDIA|GEMINI|OPENCODE)\s*分析\s*---'
        
        results = []
        
        # 找到所有匹配项
        matches = list(re.finditer(pattern, summary, flags=re.IGNORECASE))
        
        for i, match in enumerate(matches):
            provider = match.group(1).upper()
            start = match.end()
            
            # 内容结束位置：下一个匹配的开始，或者字符串末尾
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(summary)
            
            content = summary[start:end].strip()
            
            if provider and content:
                results.append((provider, content))
        
        # 如果解析失败，返回原始内容
        if not results:
            results.append(('AI', summary))
        
        return results
    

    def _build_strategy_scores_html(self, strategy_details: List[Dict]) -> str:
        """构建策略评分明细 HTML - 深色主题"""
        if not strategy_details:
            return ''
        
        # 根据市场选择货币符号
        currency_symbol = '¥' if self.market.upper() == 'HK' else '$'
        
        html_parts = []
        
        for sd in strategy_details:
            strategy_name = sd.get('strategy_name', '未知策略')
            details = sd.get('details', {})
            score_breakdown = details.get('score_breakdown', {})
            risk_management = details.get('risk_management')
            is_strong = details.get('is_strong_signal', False)
            total_score = details.get('total_score', 0)
            
            # 策略标题颜色
            strategy_key = strategy_name.lower().replace('策略', '').replace(' ', '_')
            strategy_color = self.STRATEGY_COLORS.get(strategy_key, '#3b82f6')
            
            # 构建评分明细区域
            if score_breakdown:
                # 评分项名称映射
                score_labels = {
                    'obv_divergence': ('OBV底背離', 30),
                    'boll_oversold': ('布林帶超賣', 25),
                    'volume_ratio': ('量比評分', 15),
                    'money_flow': ('資金流評分', 15),
                    'trend': ('趨勢評分', 15)
                }
                
                score_items = []
                for key, (label, max_score) in score_labels.items():
                    if key in score_breakdown:
                        score = score_breakdown[key]
                        # 计算进度条百分比
                        pct = min(score / max_score * 100, 100) if max_score > 0 else 0
                        # 根据得分比例确定颜色
                        if pct >= 80:
                            bar_color = '#10b981'  # green
                        elif pct >= 50:
                            bar_color = '#f59e0b'  # amber
                        else:
                            bar_color = '#6b7280'  # gray
                        
                        score_items.append("""
                                <div class="flex items-center gap-3">
                                    <span class="text-xs text-gray-400 w-20">{label}</span>
                                    <div class="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                                        <div class="h-full rounded-full transition-all" style="width: {pct}%; background: {bar_color}"></div>
                                    </div>
                                    <span class="text-xs font-mono w-12 text-right" style="color: {bar_color}">{score:.0f}/{max_score}分</span>
                                </div>""".format(label=label, pct=pct, bar_color=bar_color, score=score, max_score=max_score))
                
                # 强信号标识
                strong_badge = ''
                if is_strong:
                    strong_badge = '<span class="ml-2 px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 text-xs font-bold">✅ 強信號</span>'
                
                # 确定总分颜色
                total_color = '#f59e0b' if is_strong else '#3b82f6'
                score_items_str = ''.join(score_items)
                
                score_section = """
                            <div class="pt-4 border-t border-terminal-border">
                                <h4 class="text-xs font-semibold text-terminal-muted uppercase tracking-wider mb-3 flex items-center gap-2">
                                    <span>📊 {strategy_name} 評分明細</span>
                                    {strong_badge}
                                </h4>
                                <div class="space-y-2">
                                    {score_items_str}
                                </div>
                                <div class="mt-3 pt-2 border-t border-terminal-border/50 flex items-center justify-between">
                                    <span class="text-sm text-gray-400">總分</span>
                                    <span class="font-mono font-bold text-lg" style="color: {total_color}">{total_score:.0f}分</span>
                                </div>
                            </div>""".format(strategy_name=strategy_name, strong_badge=strong_badge, score_items_str=score_items_str, total_color=total_color, total_score=total_score)
                
                html_parts.append(score_section)
            
            # 构建风险管理区域
            if risk_management:
                entry_price = risk_management.get('entry_price', 0)
                stop_loss = risk_management.get('stop_loss', 0)
                stop_loss_pct = risk_management.get('stop_loss_pct', 0)
                take_profit_1 = risk_management.get('take_profit_1', 0)
                take_profit_2 = risk_management.get('take_profit_2', 0)
                rr_ratio = risk_management.get('risk_reward_ratio', 0)
                
                risk_section = """
                            <div class="pt-4 border-t border-terminal-border">
                                <h4 class="text-xs font-semibold text-terminal-muted uppercase tracking-wider mb-3 flex items-center gap-2">
                                    <span>🎯 風險管理</span>
                                </h4>
                                <div class="grid grid-cols-2 gap-2 text-sm">
                                    <div class="flex justify-between py-1">
                                        <span class="text-gray-400">入場價</span>
                                        <span class="font-mono text-blue-400">{currency_symbol}{entry_price:.2f}</span>
                                    </div>
                                    <div class="flex justify-between py-1">
                                        <span class="text-gray-400">止損價</span>
                                        <span class="font-mono text-red-400">{currency_symbol}{stop_loss:.2f} (-{stop_loss_pct:.0f}%)</span>
                                    </div>
                                    <div class="flex justify-between py-1">
                                        <span class="text-gray-400">止盈1</span>
                                        <span class="font-mono text-green-400">{currency_symbol}{take_profit_1:.2f}</span>
                                    </div>
                                    <div class="flex justify-between py-1">
                                        <span class="text-gray-400">止盈2</span>
                                        <span class="font-mono text-green-400">{currency_symbol}{take_profit_2:.2f}</span>
                                    </div>
                                </div>
                                <div class="mt-2 pt-2 border-t border-terminal-border/50 flex items-center justify-between">
                                    <span class="text-xs text-gray-400">風險收益比</span>
                                    <span class="font-mono text-sm text-amber-400">1:{rr_ratio:.1f}</span>
                                </div>
                            </div>""".format(currency_symbol=currency_symbol, entry_price=entry_price, stop_loss=stop_loss, stop_loss_pct=stop_loss_pct, take_profit_1=take_profit_1, take_profit_2=take_profit_2, rr_ratio=rr_ratio)
                
                html_parts.append(risk_section)
        
        return ''.join(html_parts)

    def get_html_filename(self) -> str:
        """获取 HTML 文件名"""
        return f"{self.base_filename}.html"
