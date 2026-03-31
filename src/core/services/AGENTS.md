# Core Services - AGENTS.md

## OVERVIEW
Core business services orchestrating the stock analysis pipeline and report generation. Services handle data coordination, strategy execution, AI analysis integration, and thread-safe output generation.

## KEY FILES

**analysis_service.py**: Main orchestrator with `run_analysis()` entry point. Coordinates MarketDataService, CacheVersionManager, StockAnalyzer, and ReportWriter. Handles parallel execution via ThreadPoolExecutor.

**stock_analyzer.py**: Individual stock analysis with rate limiting via `_rate_limit_request()`. Integrates with YahooFinanceRepository, StrategyEngine, and AIAnalysisService (multi-provider support). Returns `StockAnalysisResult` entities.

**report_writer.py**: Thread-safe HTML/JSON/TXT report generation using `threading.RLock()`. Supports incremental writes with `_min_update_interval` debouncing. Atomic file writes via temp file + rename. Regex parsing in `_parse_multi_provider_summary()` for multi-provider AI output.

**chart_renderer.py**: Candlestick visualization using Lightweight Charts. Generates responsive HTML with MA, BB, MACD, RSI overlays and volume bars.

**cache_version_manager.py**: Cache versioning via `version.txt` files per market. `check_version()` determines if sync is needed based on date + interval type.

**market_data_service.py**: Stock list retrieval and market health assessment. Fetches ^GSPC for US, ^HSI for HK. Returns `MarketData` dataclass with trend direction.

**progress_tracker.py**: ETA calculation and progress bar rendering. Used during parallel stock analysis.

## CONVENTIONS

- Thread safety: Use `threading.RLock()` for report writer state
- Atomic writes: Write to temp file, then `os.rename()`
- Incremental writes: `_incremental_update_html_unlocked()` with `_min_update_interval = 0.3s` debouncing
- Rate limiting: `_rate_limit_request()` with `base_delay=0.4s` and exponential backoff
- Multi-provider: `providers` list with parallel execution via `ThreadPoolExecutor`
- Regex patterns: `_parse_multi_provider_summary()` matches `--- PROVIDER 分析 ---` format only
- Version check: `CacheVersionManager.check_version()` before any data fetch

## ANTI-PATTERNS

- Never write directly to report file - use `ReportWriter.write_stock_result()`
- Don't skip rate limiting in stock_analyzer - always use `_rate_limit_request()`
- Never cache without version check - use `CacheVersionManager.is_sync_needed`
- Don't modify `_results` without holding `self._lock`
- Never disable `force_refresh` check when `is_sync_needed=True`
- Don't parse multi-provider output without the `---` delimiter regex

## DEPENDENCIES

Services import from:
- `src.data.loaders` (yahoo_loader, stock_list_loader)
- `src.ai.analyzer` (service.py for multi-provider AI)
- `src.core.strategies` (StrategyEngine, loader)
- `src.core.models` (entities.py for dataclasses)
