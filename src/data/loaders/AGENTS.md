# Data Loaders

**Generated:** 2026-04-05 | **Commit:** 9f0acb5 | **Branch:** master

## Overview
数据获取和技术指标计算层。所有加载器与外部数据源 (Yahoo Finance、Finviz、RSS feeds) 交互，返回标准化的 DataFrame 或股票列表。

## Key Files

- **yahoo_loader.py** - 通过 yfinance 的主要数据源。获取 OHLCV 数据，计算技术指标，管理 CSV 缓存。所有数据流经的中心。
- **divergence.py** - RSI/MACD/价格背离检测模式。返回 `DivergenceSignal` 数据类和强度评分。
- **fundamentals.py** - EPS、收入、现金流、利润率数据获取，通过 yfinance `.info` 和 `.financials`。
- **news_service.py** - 统一新闻聚合，来自 Yahoo RSS 和 Google News。返回 `NewsResult` 合并来源。
- **finviz_loader.py** - Finviz 筛选器数据：分析师评级、目标价、内部交易。
- **us_loader.py / hk_loader.py** - 特定市场的股票列表加载，从 statementdog.com 和 hkex.com.hk。
- **stock_list_loader_enhanced.py** - 增强版加载器，带过滤、行业分类和元数据。
- **google_news_loader.py** - Google News RSS 解析器，带情感分析。
- **fibonacci.py / ichimoku.py** - 附加技术指标计算。

## Conventions

- 所有加载器返回 `pd.DataFrame` 或 `List[str]` - 无自定义包装对象
- 使用 `OptimizedCache` 持久化：获取前检查 `cache_service.get(key)`
- 技术指标作为列添加：`RSI_14`, `MACD`, `MACD_Signal`, `BB_Upper`, `BB_Lower`, `ATR_14`, `MA_5`, `MA_10`, `MA_20`, `MA_50`, `MA_200`, `BBP`, `CMO`, `Williams_R`, `Stoch_K`
- 处理 Yahoo Finance 的 SSL 上下文：如果可用，使用 `certifi.where()`
- 计算指标前验证调整状态：检查 `hist.attrs.get('auto_adjusted')`

## Anti-Patterns

- **Never** 无缓存检查获取 - 始终先调用 `cache_service.get()` 并设置适当的 TTL
- **Never** 重复计算指标 - 计算前检查 `if 'RSI_14' not in df.columns`
- **Never** 在生产环境忽略 SSL 错误 - 使用 certifi 设置正确的 SSL 上下文，不要禁用验证
- **Never** 分钟线数据获取全部历史 - 1m 间隔使用 `period='7d'` 避免大量下载
- **Never** 原地修改输入 DataFrame - 添加列前始终 `.copy()`

## Dependencies

```python
from src.data.cache.cache_service import OptimizedCache  # 所有缓存
from src.utils.exceptions import DataFetchException, CacheException  # 错误处理
from src.utils.logger import get_data_logger  # 结构化日志
```

## Example Usage

```python
from src.data.loaders import YahooFinanceRepository

repo = YahooFinanceRepository()
hist = repo.get_historical_data('AAPL', market='US', interval='1d')
# 返回带所有预计算技术指标的 DataFrame
```
