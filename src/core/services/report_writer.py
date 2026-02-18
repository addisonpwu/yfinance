"""
报告写入器

负责报告文件的实时输出和格式化
"""
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class ReportWriter:
    """报告写入器"""
    
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
    
    def __init__(self, filename: str = None, market: str = 'HK'):
        """
        初始化报告写入器
        
        Args:
            filename: 输出文件名
            market: 市场代码
        """
        self.filename = filename or self._generate_filename(market)
        self.market = market
        self._lock = threading.Lock()
        self._initialized = False
    
    def _generate_filename(self, market: str) -> str:
        """生成默认文件名"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        return f"{market.lower()}_stocks_{today_str}.txt"
    
    def initialize(self) -> bool:
        """
        初始化报告文件
        
        Returns:
            是否成功创建
        """
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.write("--- 最终筛选结果 (详细) ---\n")
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
        
        with self._lock:
            try:
                with open(self.filename, 'a', encoding='utf-8') as f:
                    f.write(f"\n✅ {info.get('longName', symbol)} ({symbol})\n")
                    f.write(f"   - 符合策略: {strategies}\n")
                    f.write(f"   - 产业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}\n")
                    f.write(f"   - 市值: {market_cap_str}\n")
                    f.write(f"   - 流通股本: {float_shares_str}\n")
                    f.write(f"   - 成交量: {volume_str}\n")
                    f.write(f"   - 市盈率 (PE): {pe_ratio_str}\n")
                    f.write(f"   - 网站: {info.get('website', 'N/A')}\n")
                    
                    if ai_analysis:
                        f.write(f"   --- AI 综合分析 ---\n")
                        f.write(f"     {ai_analysis.get('summary', 'N/A')}\n")
                        f.write(f"     模型: {ai_analysis.get('model_used', 'N/A')}\n")
                    else:
                        f.write(f"   --- AI 分析未完成 ---\n")
            except Exception as e:
                print(f"写入报告时出错: {e}")
    
    def write_summary(self, results: List[Dict[str, Any]], market: str) -> None:
        """
        写入摘要列表
        
        Args:
            results: 股票分析结果列表
            market: 市场代码
        """
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
            
            formatted_stocks.append(f"{exchange_name}:{symbol} ({long_name})")
        
        summary_lines = [
            "\n" + "=" * 50,
            "--- 摘要列表 (便于复制到交易软件) ---",
            ", ".join(formatted_stocks)
        ]
        
        try:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
        except Exception as e:
            print(f"写入摘要时出错: {e}")
    
    def get_filename(self) -> str:
        """获取输出文件名"""
        return self.filename
    
    def exists(self) -> bool:
        """检查文件是否存在"""
        return os.path.exists(self.filename)
    
    def read_content(self) -> str:
        """读取文件内容"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取报告文件时发生错误: {e}")
            return ""
