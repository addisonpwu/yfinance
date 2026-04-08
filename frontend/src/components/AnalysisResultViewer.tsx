import { useState } from 'react'
import type { AnalysisTaskStatus } from '../types/api'

interface AnalysisResultViewerProps {
  task: AnalysisTaskStatus
  onClose: () => void
  onReanalyze: () => void
}

export function AnalysisResultViewer({ task, onClose, onReanalyze }: AnalysisResultViewerProps) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null)

  const results = task.results
  if (results.length === 0) {
    return (
      <div className="analysis-panel">
        <div className="panel-header">
          <h3>分析結果 — {task.symbol}</h3>
        </div>
        <p className="empty-result">沒有可用的分析結果</p>
        <div className="panel-actions">
          <button className="btn-reanalyze" onClick={onReanalyze}>重新分析</button>
          <button className="btn-close" onClick={onClose}>關閉</button>
        </div>
      </div>
    )
  }

  // Calculate consensus
  const directions = results.map((r) => r.direction)
  const bullish = directions.filter((d) => d === '看漲' || d === '上升').length
  const bearish = directions.filter((d) => d === '看跌' || d === '下降').length
  const neutral = results.length - bullish - bearish

  let consensusDir = '中性'
  if (bullish > bearish && bullish > neutral) {
    consensusDir = '看漲'
  } else if (bearish > bullish && bearish > neutral) {
    consensusDir = '看跌'
  }
  const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length

  const toggleExpand = (modelUsed: string) => {
    setExpandedModel(expandedModel === modelUsed ? null : modelUsed)
  }

  return (
    <div className="analysis-panel">
      <div className="panel-header">
        <h3>✅ 分析完成 — {task.symbol}</h3>
      </div>

      <div className="consensus-banner">
        <span className="consensus-direction">{consensusDir}</span>
        <span className="consensus-separator">|</span>
        <span className="consensus-confidence">
          平均置信度: {Math.round(avgConfidence * 100)}%
        </span>
        <span className="consensus-votes">
          ({results.length} 個模型: 看漲 {bullish} / 中性 {neutral} / 看跌 {bearish})
        </span>
      </div>

      <div className="results-list">
        {results.map((result, index) => {
          const modelName = result.model_used.split('/').pop() || result.model_used
          const isExpanded = expandedModel === result.model_used
          return (
            <div key={index} className="result-card">
              <div className="result-header" onClick={() => toggleExpand(result.model_used)}>
                <span className="result-model">{modelName}</span>
                <span className={`result-badge ${result.direction === '看漲' || result.direction === '上升' ? 'bullish' : result.direction === '看跌' || result.direction === '下降' ? 'bearish' : 'neutral'}`}>
                  {result.direction}
                </span>
                <span className="result-confidence">{Math.round(result.confidence * 100)}%</span>
                <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
              </div>
              {isExpanded && (
                <div className="result-body">
                  <pre className="result-summary">{result.summary}</pre>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="panel-actions">
        <button className="btn-reanalyze" onClick={onReanalyze}>重新分析</button>
        <button className="btn-close" onClick={onClose}>關閉</button>
      </div>
    </div>
  )
}
