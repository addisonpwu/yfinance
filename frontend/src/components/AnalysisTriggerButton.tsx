import { useState } from 'react'

interface AnalysisTriggerButtonProps {
  symbol: string
  market: string
  onTrigger: (symbol: string, market: string) => void
  disabled?: boolean
  isRunning?: boolean
}

export function AnalysisTriggerButton({
  symbol,
  market,
  onTrigger,
  disabled = false,
  isRunning = false,
}: AnalysisTriggerButtonProps) {
  const [showConfirm, setShowConfirm] = useState(false)

  const handleClick = () => {
    if (disabled || isRunning) return
    setShowConfirm(true)
  }

  const handleConfirm = () => {
    setShowConfirm(false)
    onTrigger(symbol, market)
  }

  const handleCancel = () => {
    setShowConfirm(false)
  }

  return (
    <>
      <button
        className={`ai-trigger-btn ${isRunning ? 'running' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={handleClick}
        disabled={disabled || isRunning}
        title={isRunning ? '分析進行中...' : '啟動 NVIDIA AI 分析'}
      >
        {isRunning ? (
          <span className="spinner" />
        ) : (
          <span>🤖</span>
        )}
      </button>

      {showConfirm && (
        <div className="modal-overlay" onClick={handleCancel}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>🤖 NVIDIA AI 多模型分析</h3>
            <p className="modal-symbol">{symbol}</p>
            <p className="modal-desc">
              將使用所有已配置的 NVIDIA 模型逐一分析（3 步流程：趨勢判斷 → 關鍵價位 → 綜合分析）。
              每個模型的結果將自動儲存至數據庫。
            </p>
            <p className="modal-warning">
              ⚠️ 分析可能需要數分鐘，請保持頁面開啟。
            </p>
            <div className="modal-actions">
              <button className="btn-cancel" onClick={handleCancel}>取消</button>
              <button className="btn-confirm" onClick={handleConfirm}>確認開始</button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
