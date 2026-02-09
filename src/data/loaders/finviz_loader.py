"""
Finviz 数据获取模块

从 Finviz 获取额外数据:
- 资金流向 (Sector/Industry Performance)
- 分析师评级 (Analyst Ratings)
- 目标价 (Price Targets)
- 内部交易数据 (Insider Transactions)
"""

import requests
import pandas as pd
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
from src.utils.logger import get_data_logger

# Finviz API 基础 URL
FINVIZ_API_BASE = "https://finviz.com/api"

# 请求头
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class FinvizLoader:
    """Finviz 数据加载器"""

    def __init__(self, api_key: str = None):
        """
        初始化 Finviz 加载器

        Args:
            api_key: Finviz API Key (可选，部分功能需要)
        """
        self.api_key = api_key
        self.logger = get_data_logger()
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def get_quote_data(self, symbol: str) -> Optional[Dict]:
        """
        获取股票 quote 数据

        Args:
            symbol: 股票代码 (例如: AAPL)

        Returns:
            包含各种技术/基本面指标的字典
        """
        try:
            url = f"{FINVIZ_API_BASE}/quote.ashx"
            params = {"t": symbol}
            if self.api_key:
                params["token"] = self.api_key

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # 解析 CSV 格式的响应
            data = self._parse_quote_response(response.text)

            if data:
                self.logger.info(f"成功获取 {symbol} 的 Finviz quote 数据")
                return data
            else:
                self.logger.warning(f"无法获取 {symbol} 的 Finviz quote 数据")
                return None

        except requests.RequestException as e:
            self.logger.error(f"获取 {symbol} Finviz quote 数据失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"解析 {symbol} Finviz quote 数据失败: {e}")
            return None

    def _parse_quote_response(self, response_text: str) -> Dict:
        """
        解析 Finviz quote API 响应 (CSV格式)

        Args:
            response_text: CSV 格式的响应文本

        Returns:
            解析后的字典
        """
        data = {}
        try:
            # CSV 格式: id=value
            for line in response_text.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip()
            return data
        except Exception as e:
            self.logger.error(f"解析 CSV 响应失败: {e}")
            return data

    def get_analyst_data(self, symbol: str) -> Optional[Dict]:
        """
        获取分析师评级数据

        Args:
            symbol: 股票代码

        Returns:
            包含分析师评级的字典
        """
        try:
            url = f"{FINVIZ_API_BASE}/analyst.ashx"
            params = {"t": symbol}
            if self.api_key:
                params["token"] = self.api_key

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            return self._parse_analyst_response(response.text)

        except requests.RequestException as e:
            self.logger.error(f"获取 {symbol} 分析师数据失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"解析 {symbol} 分析师数据失败: {e}")
            return None

    def _parse_analyst_response(self, response_text: str) -> Dict:
        """解析分析师响应"""
        data = {
            "strong_buy": 0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "strong_sell": 0,
        }
        try:
            # 尝试解析 JSON
            import json
            data = json.loads(response_text)
            return data
        except json.JSONDecodeError:
            # 尝试解析 CSV
            for line in response_text.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    if "strongBuy" in key.lower():
                        data["strong_buy"] = int(value)
                    elif key.lower() == "buy":
                        data["buy"] = int(value)
                    elif key.lower() == "hold":
                        data["hold"] = int(value)
                    elif key.lower() == "sell":
                        data["sell"] = int(value)
                    elif "strongSell" in key.lower():
                        data["strong_sell"] = int(value)
        except Exception as e:
            self.logger.error(f"解析分析师响应失败: {e}")
        return data

    def get_insider_data(self, symbol: str) -> Optional[List[Dict]]:
        """
        获取内部交易数据 (Insider Trading)

        Args:
            symbol: 股票代码

        Returns:
            内部交易记录列表
        """
        try:
            url = f"{FINVIZ_API_BASE}/insider.ashx"
            params = {"t": symbol}
            if self.api_key:
                params["token"] = self.api_key

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            return self._parse_insider_response(response.text)

        except requests.RequestException as e:
            self.logger.error(f"获取 {symbol} 内部交易数据失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"解析 {symbol} 内部交易数据失败: {e}")
            return None

    def _parse_insider_response(self, response_text: str) -> List[Dict]:
        """解析内部交易响应"""
        transactions = []
        try:
            # 尝试解析 JSON
            import json
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
            return transactions
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.logger.error(f"解析内部交易响应失败: {e}")
        return transactions

    def get_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """
        获取基本面数据

        Args:
            symbol: 股票代码

        Returns:
            包含估值、收益、股息等指标的字典
        """
        quote_data = self.get_quote_data(symbol)
        if not quote_data:
            return None

        # 提取关键基本面指标
        fundamental = {}

        # Valuation Ratios
        if "P/E" in quote_data:
            fundamental["pe_ratio"] = float(quote_data.get("P/E", 0))
        if "Forward P/E" in quote_data:
            fundamental["forward_pe"] = float(quote_data.get("Forward P/E", 0))
        if "PEG" in quote_data:
            fundamental["peg_ratio"] = float(quote_data.get("PEG", 0))
        if "P/B" in quote_data:
            fundamental["pb_ratio"] = float(quote_data.get("P/B", 0))
        if "P/S" in quote_data:
            fundamental["ps_ratio"] = float(quote_data.get("P/S", 0))
        if "P/C" in quote_data:
            fundamental["pc_ratio"] = float(quote_data.get("P/C", 0))
        if "P/FCF" in quote_data:
            fundamental["pfcf_ratio"] = float(quote_data.get("P/FCF", 0))

        # EPS
        if "EPS (ttm)" in quote_data:
            fundamental["eps_ttm"] = float(quote_data.get("EPS (ttm)", 0))
        if "EPS growth this year" in quote_data:
            fundamental["eps_growth_year"] = float(quote_data.get("EPS growth this year", 0).replace("%", ""))
        if "EPS growth next 5 years" in quote_data:
            fundamental["eps_growth_5y"] = float(quote_data.get("EPS growth next 5 years", 0).replace("%", ""))

        # Profitability
        if "ROE" in quote_data:
            fundamental["roe"] = float(quote_data.get("ROE", 0).replace("%", ""))
        if "ROA" in quote_data:
            fundamental["roa"] = float(quote_data.get("ROA", 0).replace("%", ""))
        if "ROI" in quote_data:
            fundamental["roi"] = float(quote_data.get("ROI", 0).replace("%", ""))
        if "Profit Margin" in quote_data:
            fundamental["profit_margin"] = float(quote_data.get("Profit Margin", 0).replace("%", ""))
        if "Operating Margin" in quote_data:
            fundamental["operating_margin"] = float(quote_data.get("Operating Margin", 0).replace("%", ""))

        # Dividends
        if "Dividend" in quote_data:
            dividend = quote_data.get("Dividend", "")
            fundamental["dividend_yield"] = float(dividend.replace("%", "")) if "%" in dividend else float(dividend)
        if "Dividend %" in quote_data:
            fundamental["dividend_yield"] = float(quote_data.get("Dividend %", 0).replace("%", ""))

        # Growth
        if "Revenue growth" in quote_data:
            fundamental["revenue_growth"] = float(quote_data.get("Revenue growth", 0).replace("%", ""))
        if "Earnings growth" in quote_data:
            fundamental["earnings_growth"] = float(quote_data.get("Earnings growth", 0).replace("%", ""))

        # Performance
        if "Perf Week" in quote_data:
            fundamental["perf_week"] = float(quote_data.get("Perf Week", 0).replace("%", ""))
        if "Perf Month" in quote_data:
            fundamental["perf_month"] = float(quote_data.get("Perf Month", 0).replace("%", ""))
        if "Perf Quarter" in quote_data:
            fundamental["perf_quarter"] = float(quote_data.get("Perf Quarter", 0).replace("%", ""))
        if "Perf YTD" in quote_data:
            fundamental["perf_ytd"] = float(quote_data.get("Perf YTD", 0).replace("%", ""))
        if "Perf Year" in quote_data:
            fundamental["perf_year"] = float(quote_data.get("Perf Year", 0).replace("%", ""))
        if "Beta" in quote_data:
            fundamental["beta"] = float(quote_data.get("Beta", 0))
        if "ATR" in quote_data:
            fundamental["atr"] = float(quote_data.get("ATR", 0))
        if "Average Volume" in quote_data:
            avg_vol = quote_data.get("Average Volume", "")
            fundamental["avg_volume"] = self._parse_volume(avg_vol)
        if "Relative Volume" in quote_data:
            fundamental["relative_volume"] = float(quote_data.get("Relative Volume", 0))
        if "Short Interest" in quote_data:
            fundamental["short_interest"] = float(quote_data.get("Short Interest", 0).replace("M", ""))
        if "Short Ratio" in quote_data:
            fundamental["short_ratio"] = float(quote_data.get("Short Ratio", 0))
        if "Target Price" in quote_data:
            fundamental["target_price"] = float(quote_data.get("Target Price", 0))
        if "Analyst Rec" in quote_data:
            fundamental["analyst_rating"] = quote_data.get("Analyst Rec", "")
        if "Insider Own" in quote_data:
            fundamental["insider_ownership"] = float(quote_data.get("Insider Own", 0).replace("%", ""))
        if "Inst Own" in quote_data:
            fundamental["institutional_ownership"] = float(quote_data.get("Inst Own", 0).replace("%", ""))

        return fundamental if fundamental else None

    def _parse_volume(self, volume_str: str) -> int:
        """解析成交量字符串"""
        try:
            if "M" in volume_str:
                return int(float(volume_str.replace("M", "")) * 1_000_000)
            elif "K" in volume_str:
                return int(float(volume_str.replace("K", "")) * 1_000)
            else:
                return int(volume_str)
        except (ValueError, TypeError):
            return 0

    def get_all_data(self, symbol: str) -> Optional[Dict]:
        """
        获取所有可用的 Finviz 数据

        Args:
            symbol: 股票代码

        Returns:
            包含所有 Finviz 数据的字典
        """
        all_data = {}

        # 获取 Quote 数据 (包含大部分指标)
        quote_data = self.get_quote_data(symbol)
        if quote_data:
            all_data.update(quote_data)

        # 获取基本面分析数据
        fundamental = self.get_fundamental_data(symbol)
        if fundamental:
            all_data["fundamental"] = fundamental

        # 获取分析师数据
        analyst_data = self.get_analyst_data(symbol)
        if analyst_data:
            all_data["analyst"] = analyst_data

        # 获取内部交易数据
        insider_data = self.get_insider_data(symbol)
        if insider_data:
            all_data["insider"] = insider_data

        return all_data if all_data else None


def get_finviz_data(symbol: str, api_key: str = None) -> Optional[Dict]:
    """
    便捷函数：获取股票的 Finviz 数据

    Args:
        symbol: 股票代码 (例如: AAPL)
        api_key: Finviz API Key (可选)

    Returns:
        包含 Finviz 数据的字典
    """
    loader = FinvizLoader(api_key=api_key)
    return loader.get_all_data(symbol)


# 如果需要单独安装 beautifulsoup4，可以取消下面的注释
# try:
#     from bs4 import BeautifulSoup
# except ImportError:
#     print("请安装 beautifulsoup4: pip install beautifulsoup4")
