export interface Stock {
  id: number
  symbol: string
  name: string
  market: 'US' | 'HK'
  created_at: string
  updated_at: string
  positive_news_count: number
  negative_news_count: number
}

export interface StockListResponse {
  items: Stock[]
  total: number
  skip: number
  limit: number
}

export interface News {
  id: number
  stock_id: number
  stock_symbol: string
  title: string
  content: string | null
  publish_time: string
  url: string
  created_at: string
}

export interface NewsListResponse {
  items: News[]
  total: number
  skip: number
  limit: number
}

export interface ApiError {
  detail: string
}

// AI Analysis types
export interface AIAnalysis {
  id: number
  stock_id: number
  provider: string
  model_used: string
  interval: string
  summary: string
  confidence: number
  recommendation: string | null
  entry_price: number | null
  exit_price: number | null
  stop_loss: number | null
  detailed_analysis: Record<string, unknown> | null
  error: string | null
  analyzed_at: string
  created_at: string
}

export interface AIAnalysisListResponse {
  items: AIAnalysis[]
  total: number
  skip: number
  limit: number
}

export interface AnalysisTriggerResponse {
  task_id: string
  status: string
  symbol: string
  market: string
  interval: string
  models: string[]
  existing: boolean
  message?: string
}

export interface AnalysisTaskProgress {
  current_model_index: number
  total_models: number
  current_model_name: string
  completed_models: string[]
  failed_models: string[]
}

export interface AnalysisTaskStatus {
  task_id: string
  symbol: string
  market: string
  interval: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  current_model: string | null
  current_step: string
  progress: AnalysisTaskProgress
  results: Array<{
    provider: string
    model_used: string
    summary: string
    confidence: number
    detailed_analysis: Record<string, unknown> | null
    direction: string
  }>
  error: string | null
  started_at: string | null
  completed_at: string | null
}
