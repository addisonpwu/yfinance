import { useState, useMemo } from 'react'
import type { Stock } from '../types/api'

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
  onMarketChange,
  currentMarket,
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

  const formatTime = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className="card flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-surface-100">
          股票列表
          <span className="ml-2 text-sm font-normal text-surface-400">
            ({total} 支股票)
          </span>
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onMarketChange(null)}
            className={`btn text-sm ${
              !currentMarket
                ? 'btn-primary'
                : 'btn-secondary'
            }`}
          >
            全部
          </button>
          <button
            onClick={() => onMarketChange('HK')}
            className={`btn text-sm ${
              currentMarket === 'HK'
                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                : 'btn-secondary'
            }`}
          >
            港股
          </button>
          <button
            onClick={() => onMarketChange('US')}
            className={`btn text-sm ${
              currentMarket === 'US'
                ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                : 'btn-secondary'
            }`}
          >
            美股
          </button>
        </div>
      </div>

      <div className="mb-4">
        <input
          type="text"
          placeholder="搜索股票代码或名称..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="input"
        />
      </div>

      <div className="flex-1 overflow-auto">
        <table className="table">
          <thead className="sticky top-0 bg-surface-900">
            <tr>
              <th>代码</th>
              <th>名称</th>
              <th>市场</th>
              <th>更新时间</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={4} className="text-center py-8 text-surface-400">
                  <div className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    加载中...
                  </div>
                </td>
              </tr>
            ) : filteredStocks.length === 0 ? (
              <tr>
                <td colSpan={4} className="text-center py-8 text-surface-400">
                  暂无数据
                </td>
              </tr>
            ) : (
              filteredStocks.map(stock => (
                <tr
                  key={stock.id}
                  onClick={() =>
                    onSelect(selectedSymbol === stock.symbol ? null : stock.symbol)
                  }
                  className={`cursor-pointer ${
                    selectedSymbol === stock.symbol
                      ? 'bg-primary-600/20 border-l-2 border-l-primary-500'
                      : ''
                  }`}
                >
                  <td className="font-mono font-medium text-surface-100">
                    {stock.symbol}
                  </td>
                  <td className="text-surface-200">{stock.name}</td>
                  <td>
                    <span
                      className={stock.market === 'HK' ? 'badge-hk' : 'badge-us'}
                    >
                      {stock.market}
                    </span>
                  </td>
                  <td className="text-surface-400 text-xs">
                    {formatTime(stock.updated_at)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-surface-800">
          <div className="text-sm text-surface-400">
            第 {page + 1} / {totalPages} 页
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page === 0}
              className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              上一页
            </button>
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages - 1}
              className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              下一页
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
