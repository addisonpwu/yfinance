import { useState, useEffect, useCallback, useRef } from 'react'
import { aiAnalysisApi } from '../api/client'
import type { AnalysisTaskStatus, AnalysisTriggerResponse } from '../types/api'

interface UseAnalysisTaskReturn {
  activeTask: AnalysisTaskStatus | null
  isRunning: boolean
  isCompleted: boolean
  isFailed: boolean
  triggerAnalysis: (symbol: string, market: string, interval?: string, force?: boolean) => Promise<AnalysisTriggerResponse>
  clearTask: () => void
}

const POLL_INTERVAL_MS = 2000

export function useAnalysisTask(): UseAnalysisTaskReturn {
  const [activeTask, setActiveTask] = useState<AnalysisTaskStatus | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const clearTask = useCallback(() => {
    clearPoll()
    setActiveTask(null)
    setIsRunning(false)
  }, [clearPoll])

  // Poll task status when a task is running
  useEffect(() => {
    if (!activeTask || activeTask.status !== 'running') {
      if (activeTask?.status !== 'running') {
        setIsRunning(false)
      }
      return
    }

    setIsRunning(true)
    pollRef.current = setInterval(async () => {
      try {
        const updated = await aiAnalysisApi.getTaskStatus(activeTask.task_id)
        setActiveTask(updated)

        if (updated.status === 'completed' || updated.status === 'failed') {
          setIsRunning(false)
          clearPoll()
        }
      } catch {
        // Poll error — keep polling, task may still be running
      }
    }, POLL_INTERVAL_MS)

    return () => clearPoll()
  }, [activeTask?.task_id, activeTask?.status, clearPoll])

  const triggerAnalysis = useCallback(
    async (symbol: string, market: string, interval = '1d', force = false) => {
      const result = await aiAnalysisApi.trigger(symbol, market, interval, force)
      setActiveTask({
        task_id: result.task_id,
        symbol: result.symbol,
        market: result.market,
        interval: result.interval,
        status: result.existing ? 'running' : 'queued',
        current_model: null,
        current_step: result.existing ? 'Task already in progress' : 'Queued...',
        progress: {
          current_model_index: 0,
          total_models: result.models.length,
          current_model_name: '',
          completed_models: [],
          failed_models: [],
        },
        results: [],
        error: null,
        started_at: new Date().toISOString(),
        completed_at: null,
      })
      setIsRunning(true)
      return result
    },
    [],
  )

  const isCompleted = activeTask?.status === 'completed'
  const isFailed = activeTask?.status === 'failed'

  return { activeTask, isRunning, isCompleted, isFailed, triggerAnalysis, clearTask }
}
