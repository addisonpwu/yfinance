# Core Services

**Generated:** 2026-04-05 | **Commit:** 9f0acb5 | **Branch:** master

## Overview
核心业务服务层，编排股票分析管道和报告生成。服务处理数据协调、策略执行、AI 分析集成和线程安全输出。

## Key Files

**analysis_service.py**: 主编排器，`run_analysis()` 入口点。协调 MarketDataService、CacheVersionManager、StockAnalyzer 和 ReportWriter。通过 ThreadPoolExecutor 处理并行执行。

**stock_analyzer.py**: 单只股票分析，通过 `_rate_limit_request()` 速率限制。集成 YahooFinanceRepository、StrategyEngine 和 AIAnalysisService (多提供商支持)。返回 `StockAnalysisResult` 实体。

**report_writer.py**: 线程安全的 HTML/JSON/TXT 报告生成，使用 `threading.RLock()`。支持 `_min_update_interval` 防抖的增量写入。通过临时文件 + `os.rename()` 原子写入。`_parse_multi_provider_summary()` 中的正则解析多提供商 AI 输出。

**chart_renderer.py**: 使用 Lightweight Charts 的 K 线可视化。生成响应式 HTML，包含 MA、BB、MACD、RSI 叠加和成交量柱状图。

**cache_version_manager.py**: 通过 `version.txt` 文件进行缓存版本控制。`check_version()` 根据日期和间隔类型确定是否需要同步。

**market_data_service.py**: 股票列表检索和市场健康评估。获取 ^GSPC (美股) 或 ^HSI (港股)。返回包含趋势方向的 `MarketData` 数据类。

**progress_tracker.py**: ETA 计算和进度条渲染。在并行股票分析期间使用。

## Conventions

- 线程安全: 报告写入器使用 `threading.RLock()`
- 原子写入: 写入临时文件，然后 `os.rename()`
- 增量写入: `_incremental_update_html_unlocked()` 使用 `_min_update_interval = 0.3s` 防抖
- 速率限制: `_rate_limit_request()` 使用 `base_delay=0.4s` 和指数退避
- 多提供商: `providers` 列表通过 `ThreadPoolExecutor` 并行执行
- 正则模式: `_parse_multi_provider_summary()` 仅匹配 `--- PROVIDER 分析 ---` 格式
- 版本检查: 任何数据获取前调用 `CacheVersionManager.check_version()`

## Anti-Patterns

- **Never** 直接写入报告文件 - 使用 `ReportWriter.write_stock_result()`
- **Never** 跳过股票分析器中的速率限制 - 始终使用 `_rate_limit_request()`
- **Never** 无版本检查缓存 - 使用 `CacheVersionManager.is_sync_needed`
- **Never** 不持有 `self._lock` 修改 `_results`
- **Never** 当 `is_sync_needed=True` 时禁用 `force_refresh` 检查
- **Never** 不使用 `---` 分隔符正则解析多提供商输出

## Dependencies

服务导入自:
- `src.data.loaders` (yahoo_loader, stock_list_loader)
- `src.ai.analyzer` (service.py 多提供商 AI)
- `src.core.strategies` (StrategyEngine, loader)
- `src.core.models` (entities.py 数据类)
