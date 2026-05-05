import { useState, useEffect, useCallback, lazy, Suspense } from 'react'
import { stockApi, newsApi, aiAnalysisApi } from './api/client'
import { useAnalysisTask } from './hooks/useAnalysisTask'
import { ErrorBoundary } from './components/ErrorBoundary'
import { StockTable } from './components/StockTable'
import { NewsList } from './components/NewsList'
import { AIAnalysisViewer } from './components/AIAnalysisViewer'
import type { Stock, News, AIAnalysis } from './types/api'

const AnalysisProgressPanel = lazy(() => import('./components/AnalysisProgressPanel').then(m => ({ default: m.AnalysisProgressPanel })))
const AnalysisResultViewer = lazy(() => import('./components/AnalysisResultViewer').then(m => ({ default: m.AnalysisResultViewer })))

const PAGE_SIZE = 20

function App() {
  const [stocks, setStocks] = useState<Stock[]>([])
  const [news, setNews] = useState<News[]>([])
  const [aiAnalyses, setAiAnalyses] = useState<AIAnalysis[]>([])
  const [stockTotal, setStockTotal] = useState(0)
  const [newsTotal, setNewsTotal] = useState(0)
  const [loadingStocks, setLoadingStocks] = useState(true)
  const [loadingNews, setLoadingNews] = useState(true)
  const [loadingAiAnalyses, setLoadingAiAnalyses] = useState(false)

  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [currentMarket, setCurrentMarket] = useState<string | null>(null)
  const [stockPage, setStockPage] = useState(0)
  const [newsPage, setNewsPage] = useState(0)
  const [mobileTab, setMobileTab] = useState(0)

  const { activeTask, isCompleted, triggerAnalysis } = useAnalysisTask()
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

  const fetchAiAnalyses = useCallback(async () => {
    if (!selectedSymbol) {
      setAiAnalyses([])
      return
    }
    setLoadingAiAnalyses(true)
    try {
      const data = await aiAnalysisApi.getHistory(selectedSymbol, 0, 20)
      setAiAnalyses(data.items)
    } catch (err) {
      console.error('Failed to fetch AI analyses:', err)
      setAiAnalyses([])
    } finally {
      setLoadingAiAnalyses(false)
    }
  }, [selectedSymbol])

  useEffect(() => { fetchStocks() }, [fetchStocks])
  useEffect(() => { fetchNews() }, [fetchNews])
  useEffect(() => { fetchAiAnalyses() }, [fetchAiAnalyses])

  const handleSelectStock = (symbol: string | null) => {
    setSelectedSymbol(selectedSymbol === symbol ? null : symbol)
    setNewsPage(0)
  }

  const handleMarketChange = (market: string | null) => {
    setCurrentMarket(market)
    setStockPage(0)
  }

  const handleReanalyze = () => {
    if (activeTask) {
      triggerAnalysis(activeTask.symbol, activeTask.market).then(() => {
        setShowResults(false)
        // Refresh AI analyses after re-trigger
        setTimeout(() => fetchAiAnalyses(), 2000)
      }).catch(console.error)
    }
  }

  const scrollToTab = (index: number) => {
    setMobileTab(index)
    const grid = document.querySelector('.grid')
    if (grid) {
      const cardWidth = grid.clientWidth
      grid.scrollTo({ left: cardWidth * index, behavior: 'smooth' })
    }
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
              <div>
                <span className="logo-text">Stock<span>Analysis</span></span>
                <div className="status-badge">
                  <span className="status-dot"></span>
                  <span>System Online</span>
                  <span style={{ color: '#404040' }}>|</span>
                  <span>{stockTotal} Assets</span>
                </div>
              </div>
            </div>
            <div className="status-badge" style={{ fontSize: '0.6875rem', background: '#1a1a1a', padding: '0.375rem 0.75rem', borderRadius: '9999px', border: '1px solid #242424' }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2">
                <ellipse cx="12" cy="5" rx="9" ry="3" />
                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
              </svg>
              <span>PostgreSQL + FastAPI</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <ErrorBoundary>
          <div className="grid">
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2">
                    <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Stocks
                </span>
                {selectedSymbol && (
                  <span style={{ fontSize: '0.6875rem', color: '#f97316', fontFamily: 'JetBrains Mono, monospace' }}>
                    {selectedSymbol}
                  </span>
                )}
              </div>
              <div className="card-body">
                <StockTable
                  stocks={stocks}
                  total={stockTotal}
                  loading={loadingStocks}
                  selectedSymbol={selectedSymbol}
                  onSelect={handleSelectStock}
                  currentMarket={currentMarket}
                  onMarketChange={handleMarketChange}
                  page={stockPage}
                  pageSize={PAGE_SIZE}
                  onPageChange={setStockPage}
                />
              </div>
            </div>

            {/* AI Analysis Panel */}
            <div className="card ai-analysis-card">
              <div className="card-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: 'linear-gradient(135deg, #ea580c, #facc15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                      <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <span className="card-title" style={{ marginBottom: 0 }}>
                      AI Analysis
                    </span>
                    <span style={{ fontSize: '0.625rem', color: '#525252' }}>
                      {selectedSymbol ? 'Analysis complete' : 'Ready to analyze'}
                    </span>
                  </div>
                </div>
                {selectedSymbol && (
                  <span style={{ fontSize: '0.6875rem', color: '#f97316', fontFamily: 'JetBrains Mono, monospace' }}>
                    {aiAnalyses.length} items
                  </span>
                )}
              </div>
              <AIAnalysisViewer
                analyses={aiAnalyses}
                loading={loadingAiAnalyses}
                selectedSymbol={selectedSymbol}
              />
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#eab308" strokeWidth="2">
                    <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                  </svg>
                  News Feed
                </span>
                <span style={{
                  fontSize: '0.6875rem',
                  color: '#525252',
                  fontFamily: 'JetBrains Mono, monospace',
                  background: '#1a1a1a',
                  padding: '0.25rem 0.5rem',
                  borderRadius: '4px'
                }}>
                  {newsTotal}
                </span>
              </div>
              <div className="card-body">
                <NewsList
                  news={news}
                  loading={loadingNews}
                  stockSymbol={selectedSymbol || undefined}
                />
              </div>
            </div>
          </div>
          </ErrorBoundary>
        </div>

        {/* Analysis Panel Overlay */}
        {showAnalysisPanel && activeTask && (
          <div className="analysis-overlay" onClick={() => { setShowAnalysisPanel(false); setShowResults(false); }}>
            <div className="analysis-panel-wrapper" onClick={(e) => e.stopPropagation()}>
              <Suspense fallback={
                <div className="analysis-panel" style={{ padding: '2rem', textAlign: 'center' }}>
                  <div className="spinner" style={{ margin: '0 auto 1rem' }} />
                  <div style={{ fontSize: '0.75rem', color: '#525252' }}>Loading analysis panel...</div>
                </div>
              }>
                {showResults && isCompleted ? (
                  <AnalysisResultViewer
                    task={activeTask}
                    onClose={() => { setShowAnalysisPanel(false); setShowResults(false); }}
                    onReanalyze={handleReanalyze}
                    onViewHistory={() => {
                      setShowAnalysisPanel(false)
                      setShowResults(false)
                      fetchAiAnalyses()
                    }}
                  />
                ) : (
                  <AnalysisProgressPanel
                    task={activeTask}
                    onClose={() => { setShowAnalysisPanel(false); setShowResults(false); }}
                    onViewResults={() => setShowResults(true)}
                  />
                )}
              </Suspense>
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

      {/* Mobile Tab Indicator */}
      <div className="mobile-tab-indicator">
        <button className={`mobile-tab-btn ${mobileTab === 0 ? 'active' : ''}`} onClick={() => scrollToTab(0)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span>Stocks</span>
        </button>
        <button className={`mobile-tab-btn ${mobileTab === 1 ? 'active' : ''}`} onClick={() => scrollToTab(1)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <span>AI</span>
        </button>
        <button className={`mobile-tab-btn ${mobileTab === 2 ? 'active' : ''}`} onClick={() => scrollToTab(2)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
          </svg>
          <span>News</span>
        </button>
      </div>
    </div>
  )
}

export default App
