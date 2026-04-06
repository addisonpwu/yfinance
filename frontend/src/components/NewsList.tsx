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
    if (hours < 24) return `${hours}h ago`
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString('zh-CN', {
      month: 'short',
      day: 'numeric',
    })
  }

  if (loading) {
    return (
      <div className="flex-1 flex flex-col gap-4">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className="loading-skeleton h-28 w-full"
            style={{ animationDelay: `${i * 0.1}s` }}
          ></div>
        ))}
      </div>
    )
  }

  if (news.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-dark-400">
        <div className="w-16 h-16 rounded-2xl bg-dark-800/50 flex items-center justify-center mb-4">
          <svg className="w-8 h-8 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
          </svg>
        </div>
        <p className="text-sm font-medium">暂无新闻数据</p>
        <p className="text-xs text-dark-500 mt-1">
          {stockSymbol ? `没有找到 ${stockSymbol} 的相关新闻` : '点击股票查看相关新闻'}
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto -mx-2 px-2">
      <div className="space-y-3">
        {news.map((item, index) => (
          <a
            key={item.id}
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="news-card block group"
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <div className="flex items-start gap-4">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 font-mono text-xs font-bold ${
                item.stock_symbol.includes('HK') 
                  ? 'bg-purple-500/20 text-purple-400' 
                  : 'bg-cyan-500/20 text-cyan-400'
              }`}>
                {item.stock_symbol.slice(0, 2)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-mono text-xs font-semibold text-brand-400">
                    {item.stock_symbol}
                  </span>
                  <span className="text-xs text-dark-500">
                    {formatTime(item.publish_time)}
                  </span>
                </div>
                <h3 className="text-sm font-medium text-white group-hover:text-brand-400 transition-colors line-clamp-2 leading-relaxed">
                  {item.title}
                </h3>
                {item.content && (
                  <p className="mt-2 text-xs text-dark-400 line-clamp-2 leading-relaxed">
                    {item.content}
                  </p>
                )}
                <div className="flex items-center gap-2 mt-3">
                  <span className="text-xs text-dark-500 group-hover:text-brand-400 transition-colors flex items-center gap-1">
                    阅读更多
                    <svg
                      className="w-3 h-3 transform group-hover:translate-x-1 transition-transform"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M14 5l7 7m0 0l-7 7m7-7H3"
                      />
                    </svg>
                  </span>
                </div>
              </div>
              <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="w-8 h-8 rounded-lg bg-brand-500/20 flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-brand-400"
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
              </div>
            </div>
          </a>
        ))}
      </div>
    </div>
  )
}
