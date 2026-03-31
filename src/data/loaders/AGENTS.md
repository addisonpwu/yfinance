# Data Loaders

Data fetching and technical indicator calculation layer. All loaders interface with external data sources (Yahoo Finance, Finviz, RSS feeds) and return standardized DataFrames or stock lists.

## Key Files

- **yahoo_loader.py** - Primary data source via yfinance. Fetches OHLCV data, calculates technical indicators, manages CSV caching. The hub all data flows through.
- **divergence.py** - RSI/MACD/price divergence detection patterns. Returns `DivergenceSignal` dataclass with strength scoring.
- **fundamentals.py** - EPS, revenue, cash flow, margin data fetching via yfinance `.info` and `.financials`.
- **news_service.py** - Unified news aggregation from Yahoo RSS and Google News. Returns `NewsResult` with merged sources.
- **finviz_loader.py** - Finviz screener data: analyst ratings, price targets, insider transactions.
- **us_loader.py / hk_loader.py** - Market-specific stock list loading from statementdog.com and hkex.com.hk.
- **stock_list_loader_enhanced.py** - Enhanced loader with filtering, sector classification, and metadata.
- **google_news_loader.py** - Google News RSS parser with sentiment analysis.
- **fibonacci.py / ichimoku.py** - Additional technical indicator calculations.

## Conventions

- All loaders return `pd.DataFrame` or `List[str]` - no custom wrapper objects
- Use `OptimizedCache` for persistence: check `cache_service.get(key)` before fetching
- Technical indicators added as columns: `RSI_14`, `MACD`, `MACD_Signal`, `BB_Upper`, `BB_Lower`, `ATR_14`, `MA_5`, `MA_10`, `MA_20`, `MA_50`, `MA_200`, `BBP`, `CMO`, `Williams_R`, `Stoch_K`
- Handle SSL context for Yahoo Finance: use `certifi.where()` if available
- Validate adjustment status before calculating indicators: check `hist.attrs.get('auto_adjusted')`

## Anti-Patterns

- **Never fetch without cache check** - Always call `cache_service.get()` first with appropriate TTL
- **Do not calculate indicators twice** - Check `if 'RSI_14' not in df.columns` before computing
- **Never ignore SSL errors in production** - Set proper SSL context with certifi, do not disable verification
- **Do not fetch all history for minute data** - Use `period='7d'` for 1m interval to avoid massive downloads
- **Do not modify input DataFrames in place** - Always `.copy()` before adding columns

## Dependencies

```python
from src.data.cache.cache_service import OptimizedCache  # All caching
from src.utils.exceptions import DataFetchException, CacheException  # Error handling
from src.utils.logger import get_data_logger  # Structured logging
```

## Example Usage

```python
from src.data.loaders import YahooFinanceRepository

repo = YahooFinanceRepository()
hist = repo.get_historical_data('AAPL', market='US', interval='1d')
# Returns DataFrame with OHLCV + all technical indicators pre-calculated
```
