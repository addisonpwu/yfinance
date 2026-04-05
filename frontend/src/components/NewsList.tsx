import type { News } from '../types/api'

interface NewsListProps {
  news: News[]
  loading: boolean
  stockSymbol?: string
}

export function NewsList({ news, loading, stockSymbol }: NewsListProps) {
  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const hours = Math.floor(diff / (1000 * 60 * 60))
    const days = Math.floor(hours / 24)

    if (hours < 1) return '刚刚'
    if (hours < 24) return `${hours} 小时前`
    if (days < 7) return `${days} 天前`
    return date.toLocaleDateString('zh-CN', {
      month: 'short',
      day: 'numeric',
    })
  }

  return (
    <div className="card flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-surface-100">
          最新新闻
          {stockSymbol && (
            <span className="ml-2 text-sm font-mono text-primary-400">
              {stockSymbol}
            </span>
          )}
        </h2>
        <span className="text-sm text-surface-400">{news.length} 条</span>
      </div>

      <div className="flex-1 overflow-auto space-y-3">
        {loading ? (
          <div className="flex items-center justify-center py-8 text-surface-400">
            <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
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
        ) : news.length === 0 ? (
          <div className="text-center py-8 text-surface-400">
            暂无新闻数据
          </div>
        ) : (
          news.map(item => (
            <a
              key={item.id}
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block card-hover group"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-xs text-primary-400">
                      {item.stock_symbol}
                    </span>
                    <span className="text-xs text-surface-500">
                      {formatTime(item.publish_time)}
                    </span>
                  </div>
                  <h3 className="text-sm font-medium text-surface-100 group-hover:text-primary-400 transition-colors line-clamp-2">
                    {item.title}
                  </h3>
                  {item.content && (
                    <p className="mt-2 text-xs text-surface-400 line-clamp-2">
                      {item.content}
                    </p>
                  )}
                </div>
                <svg
                  className="w-4 h-4 text-surface-500 group-hover:text-primary-400 transition-colors flex-shrink-0 mt-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                  />
                </svg>
              </div>
            </a>
          ))
        )}
      </div>
    </div>
  )
}
