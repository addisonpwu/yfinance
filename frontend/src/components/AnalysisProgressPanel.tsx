import type { AnalysisTaskStatus } from '../types/api'

interface AnalysisProgressPanelProps {
  task: AnalysisTaskStatus
  onClose: () => void
  onViewResults: () => void
}

export function AnalysisProgressPanel({ task, onClose, onViewResults }: AnalysisProgressPanelProps) {
  const { progress, current_model, current_step, status } = task
  const totalModels = progress.total_models
  const completedCount = progress.completed_models.length

  const pct = totalModels > 0 ? Math.round((completedCount / totalModels) * 100) : 0

  return (
    <div className="analysis-panel">
      <div className="panel-header">
        <h3>
          {status === 'completed' ? '✅ 分析完成' : status === 'failed' ? '❌ 分析失敗' : '🤖 分析進行中...'}
          {' '}{task.symbol}
        </h3>
        {status === 'running' && <span className="spinner-small" />}
      </div>

      {status === 'running' && (
        <>
          <div className="progress-bar-container">
            <div className="progress-bar" style={{ width: `${pct}%` }} />
            <span className="progress-text">{completedCount}/{totalModels}</span>
          </div>

          <div className="model-list">
            {/* Completed models */}
            {progress.completed_models.map((model) => {
              const result = task.results.find((r) => r.model_used.includes(model.split('/').pop() || model))
              return (
                <div key={model} className="model-item completed">
                  <span className="model-status-icon success">✅</span>
                  <span className="model-name">{model}</span>
                  {result && (
                    <span className="model-confidence">
                      {result.direction} | {Math.round(result.confidence * 100)}%
                    </span>
                  )}
                </div>
              )
            })}

            {/* Current model */}
            {current_model && (
              <div className="model-item running">
                <span className="model-status-icon running">⏳</span>
                <span className="model-name">{current_model}</span>
                <span className="model-step">{current_step}</span>
              </div>
            )}

            {/* Pending models */}
            {task.results.length + (current_model ? 1 : 0) < totalModels &&
              Array.from({ length: totalModels - completedCount - (current_model ? 1 : 0) }).map((_, i) => (
                <div key={`pending-${i}`} className="model-item pending">
                  <span className="model-status-icon">⏸</span>
                  <span className="model-name">等待中...</span>
                </div>
              ))}

            {/* Failed models */}
            {progress.failed_models.map((model) => (
              <div key={model} className="model-item failed">
                <span className="model-status-icon error">❌</span>
                <span className="model-name">{model}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {status === 'failed' && (
        <div className="error-message">
          <p>錯誤: {task.error || '未知錯誤'}</p>
        </div>
      )}

      <div className="panel-actions">
        {status === 'completed' && (
          <button className="btn-view" onClick={onViewResults}>查看結果</button>
        )}
        <button className="btn-close" onClick={onClose}>關閉</button>
      </div>
    </div>
  )
}
