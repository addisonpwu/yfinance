import type { AIAnalysis } from '../types/api'

interface AIAnalysisViewerProps {
  analyses: AIAnalysis[]
  loading: boolean
  selectedSymbol: string | null
}

export function AIAnalysisViewer({ analyses, loading, selectedSymbol }: AIAnalysisViewerProps) {
  const formatTime = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime()
    const hours = Math.floor(diff / 3600000)
    if (hours < 1) return '刚刚'
    if (hours < 24) return `${hours}h`
    return `${Math.floor(hours / 24)}d`
  }

  const getRecommendationClass = (rec: string | null) => {
    if (!rec) return 'hold'
    const upper = rec.toUpperCase()
    if (upper.includes('BUY') || upper.includes('買入')) return 'buy'
    if (upper.includes('SELL') || upper.includes('賣出')) return 'sell'
    return 'hold'
  }

  const getRecommendationText = (rec: string | null) => {
    if (!rec) return 'HOLD'
    const upper = rec.toUpperCase()
    if (upper.includes('BUY') || upper.includes('買入')) return 'BUY'
    if (upper.includes('SELL') || upper.includes('賣出')) return 'SELL'
    return 'HOLD'
  }

  if (loading) {
    return (
      <div className="ai-analysis-body">
        {[...Array(3)].map((_, i) => <div key={i} className="skeleton" style={{ height: '100px' }} />)}
      </div>
    )
  }

  if (!selectedSymbol) {
    return (
      <div className="ai-analysis-body">
        <div className="ai-empty-state">
          <div className="ai-empty-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <p>AI Analysis</p>
          <p className="ai-empty-hint">Select a stock to view AI analysis</p>
        </div>
      </div>
    )
  }

  if (analyses.length === 0) {
    return (
      <div className="ai-analysis-body">
        <div className="ai-empty-state">
          <div className="ai-empty-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p>No analysis yet</p>
          <p className="ai-empty-hint">Click the 🤖 button to trigger AI analysis</p>
        </div>
      </div>
    )
  }

  return (
    <div className="ai-analysis-body">
      {analyses.map((analysis) => (
        <div key={analysis.id} className="ai-summary-card">
          <div className="ai-summary-header">
            <span className="ai-model-name">{analysis.model_used?.split('/').pop() || analysis.provider}</span>
            <span className={`ai-badge ${getRecommendationClass(analysis.recommendation)}`}>
              {getRecommendationText(analysis.recommendation)}
            </span>
          </div>
          <div className="ai-confidence">
            置信度: {Math.round((analysis.confidence || 0) * 100)}% · {analysis.provider}
          </div>
          {analysis.summary && (
            <div className="ai-summary-text" style={{ marginTop: '0.5rem' }}>
              {analysis.summary}
            </div>
          )}
          <div className="ai-meta">
            <span className="ai-analysis-time">{formatTime(analysis.analyzed_at)}</span>
            {analysis.interval && (
              <span className="ai-meta-item">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12,6 12,12 16,14" />
                </svg>
                {analysis.interval}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
