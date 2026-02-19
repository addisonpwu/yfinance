"""
报告写入器

负责报告文件的实时输出和格式化，支持 TXT 和 HTML 格式
"""
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import re


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
    
    # 策略颜色映射
    STRATEGY_COLORS = {
        'momentum_breakout': '#10b981',      # 绿色
        'volatility_squeeze': '#8b5cf6',     # 紫色
        'accumulation_acceleration': '#f59e0b',  # 橙色
        'signal_scorer': '#3b82f6',          # 蓝色
        'market_regime': '#6b7280'           # 灰色
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
        except Exception:
            pass
    
    def _build_html(self, results: List[Dict[str, Any]], market: str) -> str:
        """构建完整的 HTML 报告 - 增强可视化"""
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
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --primary-light: #93c5fd;
            --success: #10b981;
            --success-light: #d1fae5;
            --warning: #f59e0b;
            --warning-light: #fef3c7;
            --danger: #ef4444;
            --danger-light: #fee2e2;
            --bg: #f1f5f9;
            --card: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 32px;
            border-radius: 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 300px;
            height: 300px;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
        }}
        
        .header h1 {{
            font-size: 1.75rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
            position: relative;
        }}
        
        .header .meta {{
            display: flex;
            gap: 24px;
            margin-top: 16px;
            flex-wrap: wrap;
            position: relative;
        }}
        
        .header .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            opacity: 0.95;
            font-size: 0.9rem;
        }}
        
        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .stat-card {{
            background: var(--card);
            border-radius: 16px;
            padding: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
        }}
        
        .stat-card .icon {{
            font-size: 1.5rem;
            margin-bottom: 8px;
        }}
        
        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stat-card .label {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 4px;
        }}
        
        /* Strategy Tags */
        .strategy-section {{
            background: var(--card);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }}
        
        .strategy-section h3 {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .strategy-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .strategy-tag {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 2px solid;
            padding: 10px 18px;
            border-radius: 12px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s;
        }}
        
        .strategy-tag:hover {{
            transform: scale(1.02);
        }}
        
        .strategy-tag .name {{
            font-weight: 600;
        }}
        
        .strategy-tag .count {{
            background: var(--primary);
            color: white;
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
        }}
        
        /* Stock Cards */
        .stocks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 20px;
        }}
        
        .stock-card {{
            background: var(--card);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }}
        
        .stock-card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }}
        
        /* Stock Header */
        .stock-header {{
            padding: 20px;
            position: relative;
        }}
        
        .stock-header.bullish {{
            background: linear-gradient(135deg, var(--success-light) 0%, #ecfdf5 100%);
            border-left: 4px solid var(--success);
        }}
        
        .stock-header.bearish {{
            background: linear-gradient(135deg, var(--danger-light) 0%, #fef2f2 100%);
            border-left: 4px solid var(--danger);
        }}
        
        .stock-header.neutral {{
            background: linear-gradient(135deg, var(--warning-light) 0%, #fffbeb 100%);
            border-left: 4px solid var(--warning);
        }}
        
        .stock-header .title-row {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        
        .stock-header h3 {{
            font-size: 1.1rem;
            color: var(--text);
            max-width: 250px;
        }}
        
        .stock-header .symbol {{
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            background: var(--primary);
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        
        .stock-strategies {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        
        .stock-strategies .badge {{
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
        }}
        
        /* Score Gauge */
        .score-gauge {{
            display: flex;
            gap: 16px;
            padding: 16px 20px;
            background: #fafbfc;
            border-bottom: 1px solid var(--border);
        }}
        
        .gauge-item {{
            flex: 1;
            text-align: center;
        }}
        
        .gauge-item .gauge-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 6px;
        }}
        
        .gauge-item .gauge-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .gauge-item .gauge-fill.high {{ background: linear-gradient(90deg, var(--success), #34d399); }}
        .gauge-item .gauge-fill.medium {{ background: linear-gradient(90deg, var(--warning), #fbbf24); }}
        .gauge-item .gauge-fill.low {{ background: linear-gradient(90deg, var(--danger), #f87171); }}
        
        .gauge-item .score-text {{
            font-size: 1.1rem;
            font-weight: 700;
        }}
        
        .gauge-item .label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        
        .score-high {{ color: var(--success); }}
        .score-medium {{ color: var(--warning); }}
        .score-low {{ color: var(--danger); }}
        
        .stock-body {{
            padding: 16px 20px;
        }}
        
        /* Data Grid */
        .data-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-bottom: 16px;
        }}
        
        .data-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: #fafbfc;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}
        
        .data-item .label {{
            color: var(--text-muted);
            font-size: 0.8rem;
        }}
        
        .data-item .value {{
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .data-item .value.positive {{
            color: var(--success);
        }}
        
        .data-item .value.negative {{
            color: var(--danger);
        }}
        
        /* Direction Indicator */
        .direction-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .direction-indicator.bullish {{
            background: var(--success-light);
            color: #065f46;
        }}
        
        .direction-indicator.bearish {{
            background: var(--danger-light);
            color: #991b1b;
        }}
        
        .direction-indicator.neutral {{
            background: var(--warning-light);
            color: #92400e;
        }}
        
        /* News Section */
        .news-section {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 12px;
            padding: 14px;
            margin-top: 12px;
        }}
        
        .news-section h4 {{
            color: #0369a1;
            margin-bottom: 10px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .news-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .news-item {{
            padding: 10px 12px;
            background: white;
            border-radius: 8px;
            border: 1px solid #bae6fd;
        }}
        
        .news-title {{
            color: #0c4a6e;
            text-decoration: none;
            font-size: 0.8rem;
            line-height: 1.4;
            display: block;
        }}
        
        .news-title:hover {{
            color: var(--primary);
            text-decoration: underline;
        }}
        
        .news-meta {{
            display: flex;
            gap: 10px;
            margin-top: 4px;
            font-size: 0.7rem;
            color: #64748b;
        }}
        
        /* AI Analysis */
        .ai-section {{
            margin-top: 12px;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        
        .ai-header {{
            background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
            padding: 12px 14px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .ai-header h4 {{
            color: #92400e;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .ai-header .toggle {{
            transition: transform 0.3s;
        }}
        
        .ai-section.collapsed .toggle {{
            transform: rotate(-90deg);
        }}
        
        .ai-content {{
            padding: 14px;
            background: #fffef5;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.8rem;
            color: #78350f;
            white-space: pre-wrap;
            line-height: 1.7;
        }}
        
        .ai-section.collapsed .ai-content {{
            display: none;
        }}
        
        .ai-model {{
            margin-top: 10px;
            font-size: 0.75rem;
            color: #a16207;
            text-align: right;
            padding-top: 8px;
            border-top: 1px dashed #fde047;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            margin-top: 40px;
        }}
        
        .footer p {{
            margin-bottom: 4px;
        }}
        
        /* Print Styles */
        @media print {{
            body {{ background: white; }}
            .container {{ max-width: 100%; }}
            .stock-card {{ break-inside: avoid; }}
            .header {{ box-shadow: none; }}
            .ai-content {{ max-height: none; }}
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .stocks-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header {{ padding: 24px; }}
            .header h1 {{ font-size: 1.4rem; }}
            .data-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
    <script>
        function toggleAI(id) {{
            const section = document.getElementById(id);
            section.classList.toggle('collapsed');
        }}
    </script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📈 股票筛选报告</h1>
            <div class="meta">
                <div class="meta-item"><span>🌏</span><span>市场: {market}</span></div>
                <div class="meta-item"><span>📅</span><span>{self._start_time.strftime('%Y-%m-%d %H:%M')}</span></div>
                <div class="meta-item"><span>⏱️</span><span>耗时: {duration:.1f} 秒</span></div>
                <div class="meta-item"><span>✅</span><span>筛选: {len(results)} 只</span></div>
            </div>
        </div>
        
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="icon">📊</div>
                <div class="value">{len(results)}</div>
                <div class="label">符合条件股票</div>
            </div>
            <div class="stat-card">
                <div class="icon">🎯</div>
                <div class="value">{len(strategy_stats)}</div>
                <div class="label">命中策略数</div>
            </div>
            <div class="stat-card">
                <div class="icon">🏆</div>
                <div class="value">{max(strategy_stats.values()) if strategy_stats else 0}</div>
                <div class="label">最高命中次数</div>
            </div>
            <div class="stat-card">
                <div class="icon">⚡</div>
                <div class="value">{duration:.0f}s</div>
                <div class="label">分析耗时</div>
            </div>
        </div>
        
        <!-- Strategy Stats -->
        <div class="strategy-section">
            <h3>策略命中统计</h3>
            <div class="strategy-tags">
                {''.join([f'<div class="strategy-tag" style="border-color: {self.STRATEGY_COLORS.get(s, "#3b82f6")}"><span class="name">{self.STRATEGY_NAMES.get(s, s)}</span><span class="count">{c}</span></div>' for s, c in sorted(strategy_stats.items(), key=lambda x: -x[1])])}
            </div>
        </div>
        
        <!-- Stock Cards -->
        <div class="stocks-grid">
            {''.join([self._build_stock_card(r, market, i) for i, r in enumerate(results)])}
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>报告生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="font-size: 0.8rem; margin-top: 8px;">⚠️ 本报告仅供参考，不构成投资建议</p>
        </div>
    </div>
</body>
</html>'''
    
    def _build_stock_card(self, result: Dict[str, Any], market: str, index: int = 0) -> str:
        """构建单只股票的 HTML 卡片 - 增强可视化"""
        info = result.get('info', {})
        symbol = result.get('symbol', '')
        strategies = result.get('strategies', [])
        ai_analysis = result.get('ai_analysis')
        news = result.get('news', [])
        
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
        change_str = f"{change_52w * 100:+.1f}%" if isinstance(change_52w, (int, float)) else "N/A"
        
        # 解析 AI 分析
        ai_summary = ai_analysis.get('summary', '') if ai_analysis else ''
        direction, confidence, tech_score, fund_score, total_score = self._parse_ai_summary(ai_summary)
        
        # 确定方向样式
        direction_class = 'neutral'
        direction_icon = '➖'
        if direction == '看涨':
            direction_class = 'bullish'
            direction_icon = '📈'
        elif direction == '看跌':
            direction_class = 'bearish'
            direction_icon = '📉'
        
        # 评分等级
        def get_score_class(score):
            if score >= 7:
                return 'high'
            elif score >= 5:
                return 'medium'
            return 'low'
        
        # 评分条宽度
        tech_width = min(tech_score * 10, 100) if tech_score else 0
        fund_width = min(fund_score * 10, 100) if fund_score else 0
        total_width = min(total_score * 10, 100) if total_score else 0
        
        return f'''
        <div class="stock-card">
            <div class="stock-header {direction_class}">
                <div class="title-row">
                    <h3>{info.get('longName', symbol)}</h3>
                    <span class="symbol">{symbol.replace('.HK', '') if market.upper() == 'HK' else symbol}</span>
                </div>
                <div class="stock-strategies">
                    {''.join([f'<span class="badge" style="background: {self.STRATEGY_COLORS.get(s, "#10b981")}">{self.STRATEGY_NAMES.get(s, s)}</span>' for s in strategies])}
                </div>
            </div>
            
            {'' if total_score == 0 else f'''
            <div class="score-gauge">
                <div class="gauge-item">
                    <div class="gauge-bar"><div class="gauge-fill {get_score_class(tech_score)}" style="width: {tech_width}%"></div></div>
                    <div class="score-text score-{get_score_class(tech_score)}">{tech_score}/10</div>
                    <div class="label">技术面</div>
                </div>
                <div class="gauge-item">
                    <div class="gauge-bar"><div class="gauge-fill {get_score_class(fund_score)}" style="width: {fund_width}%"></div></div>
                    <div class="score-text score-{get_score_class(fund_score)}">{fund_score}/10</div>
                    <div class="label">基本面</div>
                </div>
                <div class="gauge-item">
                    <div class="gauge-bar"><div class="gauge-fill {get_score_class(total_score)}" style="width: {total_width}%"></div></div>
                    <div class="score-text score-{get_score_class(total_score)}">{total_score}/10</div>
                    <div class="label">综合</div>
                </div>
            </div>
            '''}
            
            <div class="stock-body">
                <div class="data-grid">
                    <div class="data-item">
                        <span class="label">🏢 行业</span>
                        <span class="value">{info.get('sector', 'N/A')[:12]}</span>
                    </div>
                    <div class="data-item">
                        <span class="label">💰 市值</span>
                        <span class="value">{market_cap_str}</span>
                    </div>
                    <div class="data-item">
                        <span class="label">📊 成交量</span>
                        <span class="value">{volume_str}</span>
                    </div>
                    <div class="data-item">
                        <span class="label">📈 市盈率</span>
                        <span class="value">{pe_ratio_str}</span>
                    </div>
                    <div class="data-item">
                        <span class="label">📅 52周涨跌</span>
                        <span class="value {change_class}">{change_str}</span>
                    </div>
                    <div class="data-item">
                        <span class="label">🔗 链接</span>
                        <span class="value">{f'<a href="{info.get("website", "#")}" target="_blank" style="color: var(--primary);">官网</a>' if info.get('website') else 'N/A'}</span>
                    </div>
                </div>
                
                {f'''<div class="direction-indicator {direction_class}">{direction_icon} {direction} ({confidence})</div>''' if direction else ''}
                
                {self._build_news_section_html(news) if news else ''}
                
                {self._build_ai_section_html(ai_analysis, index) if ai_analysis else ''}
            </div>
        </div>'''
    
    def _build_news_section_html(self, news: list) -> str:
        """构建新闻显示区域"""
        if not news:
            return ''
        
        news_items = []
        for item in news[:3]:
            title = item.get('title', 'N/A')
            link = item.get('link', '#')
            published = item.get('published', '')
            
            news_items.append(f'''
            <div class="news-item">
                <a href="{link}" target="_blank" class="news-title">{title[:60]}{'...' if len(title) > 60 else ''}</a>
                <div class="news-meta">
                    <span>{published}</span>
                </div>
            </div>''')
        
        return f'''
        <div class="news-section">
            <h4>📰 近期新闻 ({len(news)} 条)</h4>
            <div class="news-list">
                {''.join(news_items)}
            </div>
        </div>'''
    
    def _build_ai_section_html(self, ai_analysis: Dict[str, Any], index: int) -> str:
        """构建可折叠的 AI 分析区域"""
        if not ai_analysis:
            return ''
        
        summary = ai_analysis.get('summary', 'N/A')
        model = ai_analysis.get('model_used', 'N/A')
        
        return f'''
        <div class="ai-section collapsed" id="ai-{index}">
            <div class="ai-header" onclick="toggleAI('ai-{index}')">
                <h4>🤖 AI 分析详情</h4>
                <span class="toggle">▼</span>
            </div>
            <div class="ai-content">{summary}</div>
            <div class="ai-model">模型: {model}</div>
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