import { useState, useMemo } from 'react'
import type { Stock } from '../types/api'
import { formatTime } from '../utils/time'

interface StockTableProps {
  stocks: Stock[]
  total: number
  loading: boolean
  selectedSymbol: string | null
  onSelect: (symbol: string | null) => void
  onMarketChange: (market: string | null) => void
  currentMarket: string | null
  page: number
  pageSize: number
  onPageChange: (page: number) => void
}

export function StockTable({
  stocks,
  total,
  loading,
  selectedSymbol,
  onSelect,
  page,
  pageSize,
  onPageChange,
}: StockTableProps) {
  const [search, setSearch] = useState('')

  const filteredStocks = useMemo(() => {
    if (!search) return stocks
    const term = search.toLowerCase()
    return stocks.filter(
      s =>
        s.symbol.toLowerCase().includes(term) ||
        s.name.toLowerCase().includes(term)
    )
  }, [stocks, search])

  const totalPages = Math.ceil(total / pageSize)

  if (loading) {
    return (
      <div className="flex-1 flex flex-col gap-3">
        {[...Array(8)].map((_, i) => (
          <div
            key={i}
            className="loading-skeleton h-16 w-full"
            style={{ animationDelay: `${i * 0.1}s` }}
          ></div>
        ))}
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="mb-4">
        <div className="relative">
          <svg
            className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            placeholder="搜索股票代码或名称..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="input-field pl-11"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-dark-400 hover:text-white transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto -mx-2 px-2">
        {filteredStocks.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-dark-400">
            <svg className="w-12 h-12 mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm">暂无匹配的股票</p>
          </div>
        ) : (
          <div className="space-y-2">
            {filteredStocks.map((stock, index) => (
              <div
                key={stock.id}
                onClick={() =>
                  onSelect(selectedSymbol === stock.symbol ? null : stock.symbol)
                }
                className={`table-row rounded-xl p-4 cursor-pointer ${
                  selectedSymbol === stock.symbol ? 'selected' : ''
                }`}
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center font-mono text-sm font-bold ${
                      stock.market === 'HK' 
                        ? 'bg-purple-500/20 text-purple-400' 
                        : 'bg-cyan-500/20 text-cyan-400'
                    }`}>
                      {stock.symbol.slice(0, 2)}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-semibold text-white">
                          {stock.symbol}
                        </span>
                        <span className={`market-badge ${stock.market.toLowerCase()}`}>
                          {stock.market}
                        </span>
                      </div>
                      <p className="text-sm text-dark-300 mt-0.5">
                        {stock.name}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-dark-400 font-mono">
                      {formatTime(stock.updated_at)}
                    </div>
                    {selectedSymbol === stock.symbol && (
                      <div className="flex items-center gap-1 text-brand-400 text-xs mt-1">
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                        查看新闻
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/5">
          <div className="text-xs text-dark-400">
            显示 {page * pageSize + 1}-{Math.min((page + 1) * pageSize, total)} / {total}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page === 0}
              className="btn-ghost text-xs disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <div className="flex items-center gap-1">
              {[...Array(Math.min(5, totalPages))].map((_, i) => {
                const pageNum = Math.max(0, Math.min(page - 2, totalPages - 5)) + i
                if (pageNum >= totalPages) return null
                return (
                  <button
                    key={pageNum}
                    onClick={() => onPageChange(pageNum)}
                    className={`w-8 h-8 rounded-lg text-xs font-medium transition-all ${
                      page === pageNum
                        ? 'bg-brand-500 text-white'
                        : 'text-dark-400 hover:bg-white/5'
                    }`}
                  >
                    {pageNum + 1}
                  </button>
                )
              })}
            </div>
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages - 1}
              className="btn-ghost text-xs disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
