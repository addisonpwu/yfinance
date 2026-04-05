import { useState, useEffect, useCallback } from 'react'
import { stockApi, newsApi } from './api/client'
import { StockTable } from './components/StockTable'
import { NewsList } from './components/NewsList'
import type { Stock, News } from './types/api'

const PAGE_SIZE = 20

function App() {
  const [stocks, setStocks] = useState<Stock[]>([])
  const [news, setNews] = useState<News[]>([])
  const [stockTotal, setStockTotal] = useState(0)
  const [loadingStocks, setLoadingStocks] = useState(true)
  const [loadingNews, setLoadingNews] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [currentMarket, setCurrentMarket] = useState<string | null>(null)
  const [stockPage, setStockPage] = useState(0)
  const [newsPage, setNewsPage] = useState(0)

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
      setError(err instanceof Error ? err.message : '获取股票列表失败')
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
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取新闻列表失败')
    } finally {
      setLoadingNews(false)
    }
  }, [selectedSymbol, newsPage])

  useEffect(() => {
    fetchStocks()
  }, [fetchStocks])

  useEffect(() => {
    fetchNews()
  }, [fetchNews])

  const handleMarketChange = (market: string | null) => {
    setCurrentMarket(market)
    setStockPage(0)
  }

  const handleSelectStock = (symbol: string | null) => {
    setSelectedSymbol(symbol)
    setNewsPage(0)
  }

  return (
    <div className="min-h-screen bg-surface-950">
      <header className="border-b border-surface-800 bg-surface-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <svg
                  className="w-8 h-8 text-primary-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                  />
                </svg>
                <h1 className="text-xl font-bold text-gradient">
                  Stock Analysis
                </h1>
              </div>
              <span className="text-xs text-surface-500 font-mono bg-surface-800 px-2 py-1 rounded">
                v1.0.0
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-surface-400">API 已连接</span>
              </div>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary text-sm"
              >
                API 文档
              </a>
            </div>
          </div>
        </div>
      </header>

      {error && (
        <div className="max-w-[1800px] mx-auto px-6 pt-4">
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-auto text-surface-400 hover:text-surface-200"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      <main className="max-w-[1800px] mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-140px)]">
          <StockTable
            stocks={stocks}
            total={stockTotal}
            loading={loadingStocks}
            selectedSymbol={selectedSymbol}
            onSelect={handleSelectStock}
            onMarketChange={handleMarketChange}
            currentMarket={currentMarket}
            page={stockPage}
            pageSize={PAGE_SIZE}
            onPageChange={setStockPage}
          />
          <NewsList
            news={news}
            loading={loadingNews}
            stockSymbol={selectedSymbol || undefined}
          />
        </div>

        <footer className="mt-8 pt-6 border-t border-surface-800 text-center text-xs text-surface-500">
          <div className="flex items-center justify-center gap-4">
            <span>Stock Analysis System</span>
            <span>•</span>
            <span>PostgreSQL + FastAPI + React</span>
            <span>•</span>
            <span>{new Date().getFullYear()}</span>
          </div>
        </footer>
      </main>
    </div>
  )
}

export default App
