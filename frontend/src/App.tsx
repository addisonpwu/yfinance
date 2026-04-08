import { useState, useEffect, useCallback } from 'react'
import { stockApi, newsApi } from './api/client'
import { useAnalysisTask } from './hooks/useAnalysisTask'
import { AnalysisTriggerButton } from './components/AnalysisTriggerButton'
import { AnalysisProgressPanel } from './components/AnalysisProgressPanel'
import { AnalysisResultViewer } from './components/AnalysisResultViewer'
import type { Stock, News } from './types/api'

const PAGE_SIZE = 20

function App() {
  const [stocks, setStocks] = useState<Stock[]>([])
  const [news, setNews] = useState<News[]>([])
  const [stockTotal, setStockTotal] = useState(0)
  const [newsTotal, setNewsTotal] = useState(0)
  const [loadingStocks, setLoadingStocks] = useState(true)
  const [loadingNews, setLoadingNews] = useState(true)

  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [currentMarket, setCurrentMarket] = useState<string | null>(null)
  const [stockPage, setStockPage] = useState(0)
  const [newsPage, setNewsPage] = useState(0)
  const [search, setSearch] = useState('')

  const { activeTask, isRunning, isCompleted, triggerAnalysis } = useAnalysisTask()
  const [showAnalysisPanel, setShowAnalysisPanel] = useState(false)
  const [showResults, setShowResults] = useState(false)

  const fetchStocks = useCallback(async () => {
    setLoadingStocks(true)
    try {
      const data = await stockApi.list({
        market: currentMarket || undefined,
        skip: stockPage * PAGE_SIZE,
        limit: PAGE_SIZE,
      })
      setStocks(data.items)
      setStockTotal(data.total)
    } catch (err) {
      console.error('Failed to fetch stocks:', err)
    } finally {
      setLoadingStocks(false)
    }
  }, [currentMarket, stockPage])

  const fetchNews = useCallback(async () => {
    setLoadingNews(true)
    try {
      const data = await newsApi.list({
        stock_symbol: selectedSymbol || undefined,
        skip: newsPage * 20,
        limit: 20,
      })
      setNews(data.items)
      setNewsTotal(data.total)
    } catch (err) {
      console.error('Failed to fetch news:', err)
    } finally {
      setLoadingNews(false)
    }
  }, [selectedSymbol, newsPage])

  useEffect(() => { fetchStocks() }, [fetchStocks])
  useEffect(() => { fetchNews() }, [fetchNews])

  const handleSelectStock = (symbol: string) => {
    setSelectedSymbol(selectedSymbol === symbol ? null : symbol)
    setNewsPage(0)
  }

  const handleMarketChange = (market: string | null) => {
    setCurrentMarket(market)
    setStockPage(0)
  }

  const handleTriggerAnalysis = (symbol: string, market: string) => {
    triggerAnalysis(symbol, market).then((result) => {
      setShowAnalysisPanel(true)
      setShowResults(false)
      if (result.existing) {
        // Task already running, just show panel
      }
    }).catch((err) => {
      console.error('Failed to trigger analysis:', err)
      alert(`觸發分析失敗: ${err.message}`)
    })
  }

  const handleReanalyze = () => {
    if (activeTask) {
      triggerAnalysis(activeTask.symbol, activeTask.market).catch(console.error)
      setShowResults(false)
    }
  }

  const filteredStocks = search
    ? stocks.filter(s => s.symbol.toLowerCase().includes(search.toLowerCase()) || s.name.includes(search))
    : stocks

  const stockPages = Math.ceil(stockTotal / PAGE_SIZE)
  const newsPages = Math.ceil(newsTotal / 20)

  const formatTime = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime()
    const hours = Math.floor(diff / 3600000)
    if (hours < 1) return '刚刚'
    if (hours < 24) return `${hours}h`
    return `${Math.floor(hours / 24)}d`
  }

  return (
    <div>
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo">
              <div className="logo-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                  <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <span className="logo-text">Stock Analysis</span>
            </div>
            <div className="status-badge">
              <span className="status-dot"></span>
              <span>API Connected</span>
              <span>·</span>
              <span>{stockTotal} stocks</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <div className="grid">
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Stocks
                </span>
                {selectedSymbol && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--color-primary)' }}>
                    Selected: {selectedSymbol}
                  </span>
                )}
              </div>
              <div className="card-body">
                <input
                  className="search-input"
                  placeholder="Search stocks..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />

                <div className="filter-group">
                  <button
                    className={`filter-btn ${!currentMarket ? 'active' : ''}`}
                    onClick={() => handleMarketChange(null)}
                  >
                    All ({stockTotal})
                  </button>
                  <button
                    className={`filter-btn ${currentMarket === 'HK' ? 'active' : ''}`}
                    onClick={() => handleMarketChange('HK')}
                  >
                    HK
                  </button>
                  <button
                    className={`filter-btn ${currentMarket === 'US' ? 'active' : ''}`}
                    onClick={() => handleMarketChange('US')}
                  >
                    US
                  </button>
                </div>

                {loadingStocks ? (
                  [...Array(6)].map((_, i) => <div key={i} className="skeleton" />)
                ) : filteredStocks.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p style={{ fontSize: '0.875rem' }}>No stocks found</p>
                  </div>
                ) : (
                  filteredStocks.map(stock => (
                    <div
                      key={stock.id}
                      className={`stock-item ${selectedSymbol === stock.symbol ? 'selected' : ''}`}
                      onClick={() => handleSelectStock(stock.symbol)}
                    >
                      <div className="stock-info">
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span className="stock-symbol">{stock.symbol}</span>
                            <span className={`market-tag ${stock.market.toLowerCase()}`}>{stock.market}</span>
                          </div>
                          <div className="stock-name">{stock.name}</div>
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <AnalysisTriggerButton
                          symbol={stock.symbol}
                          market={stock.market}
                          onTrigger={handleTriggerAnalysis}
                          disabled={false}
                          isRunning={isRunning && activeTask?.symbol === stock.symbol}
                        />
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          fontSize: '0.625rem',
                          color: stock.positive_news_count > 0 ? '#22c55e' : 'var(--color-text-dim)'
                        }}>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z" />
                          </svg>
                          {stock.positive_news_count}
                        </div>
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          fontSize: '0.625rem',
                          color: stock.negative_news_count > 0 ? '#ef4444' : 'var(--color-text-dim)'
                        }}>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z" />
                          </svg>
                          {stock.negative_news_count}
                        </div>
                        <span className="stock-time">{formatTime(stock.updated_at)}</span>
                      </div>
                    </div>
                  ))
                )}

                {stockPages > 1 && (
                  <div className="pagination">
                    <span className="pagination-info">
                      {stockPage * PAGE_SIZE + 1}-{Math.min((stockPage + 1) * PAGE_SIZE, stockTotal)} / {stockTotal}
                    </span>
                    <div className="pagination-btns">
                      <button
                        className="page-btn"
                        disabled={stockPage === 0}
                        onClick={() => setStockPage(p => p - 1)}
                      >
                        ‹
                      </button>
                      <button
                        className="page-btn"
                        disabled={stockPage >= stockPages - 1}
                        onClick={() => setStockPage(p => p + 1)}
                      >
                        ›
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                  </svg>
                  News
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-dim)' }}>
                  {newsTotal} items
                </span>
              </div>
              <div className="card-body">
                {loadingNews ? (
                  [...Array(5)].map((_, i) => <div key={i} className="skeleton" style={{ height: '100px' }} />)
                ) : news.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                      </svg>
                    </div>
                    <p style={{ fontSize: '0.875rem' }}>No news available</p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--color-text-dim)', marginTop: '0.25rem' }}>
                      {selectedSymbol ? `No news for ${selectedSymbol}` : 'Select a stock to filter news'}
                    </p>
                  </div>
                ) : (
                  news.map(item => (
                    <a
                      key={item.id}
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="news-item"
                    >
                      <div className="news-header">
                        <span className="news-symbol">{item.stock_symbol}</span>
                        <span className="news-time">{formatTime(item.publish_time)}</span>
                      </div>
                      <div className="news-title">{item.title}</div>
                      {item.content && <div className="news-content">{item.content}</div>}
                    </a>
                  ))
                )}

                {newsPages > 1 && (
                  <div className="pagination">
                    <span className="pagination-info">
                      {newsPage * 20 + 1}-{Math.min((newsPage + 1) * 20, newsTotal)} / {newsTotal}
                    </span>
                    <div className="pagination-btns">
                      <button
                        className="page-btn"
                        disabled={newsPage === 0}
                        onClick={() => setNewsPage(p => p - 1)}
                      >
                        ‹
                      </button>
                      <button
                        className="page-btn"
                        disabled={newsPage >= newsPages - 1}
                        onClick={() => setNewsPage(p => p + 1)}
                      >
                        ›
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Panel Overlay */}
        {showAnalysisPanel && activeTask && (
          <div className="analysis-overlay" onClick={() => { setShowAnalysisPanel(false); setShowResults(false); }}>
            <div className="analysis-panel-wrapper" onClick={(e) => e.stopPropagation()}>
              {showResults && isCompleted ? (
                <AnalysisResultViewer
                  task={activeTask}
                  onClose={() => { setShowAnalysisPanel(false); setShowResults(false); }}
                  onReanalyze={handleReanalyze}
                />
              ) : (
                <AnalysisProgressPanel
                  task={activeTask}
                  onClose={() => { setShowAnalysisPanel(false); setShowResults(false); }}
                  onViewResults={() => setShowResults(true)}
                />
              )}
            </div>
          </div>
        )}

        <footer className="footer">
          <div className="container">
            <div className="footer-content">
              <span>Stock Analysis System</span>
              <span>PostgreSQL + FastAPI + React</span>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}

export default App
