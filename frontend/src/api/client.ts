import type {
  Stock, StockListResponse, News, NewsListResponse,
  AIAnalysis, AIAnalysisListResponse, AnalysisTriggerResponse, AnalysisTaskStatus,
} from '../types/api'

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  if (response.status === 204) {
    return {} as T
  }

  return response.json()
}

export const stockApi = {
  list: (params?: {
    market?: string;
    skip?: number;
    limit?: number;
    sort_by?: string;
    sort_order?: 'asc' | 'desc';
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.market) searchParams.set('market', params.market)
    if (params?.skip) searchParams.set('skip', String(params.skip))
    if (params?.limit) searchParams.set('limit', String(params.limit))
    if (params?.sort_by) searchParams.set('sort_by', params.sort_by)
    if (params?.sort_order) searchParams.set('sort_order', params.sort_order)
    const query = searchParams.toString()
    return fetchApi<StockListResponse>(`/api/v1/stocks/${query ? `?${query}` : ''}`)
  },

  get: (symbol: string) => {
    return fetchApi<Stock>(`/api/v1/stocks/${symbol}`)
  },

  create: (data: { symbol: string; name: string; market: string }) => {
    return fetchApi<Stock>('/api/v1/stocks/', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  delete: (symbol: string) => {
    return fetchApi<void>(`/api/v1/stocks/${symbol}`, {
      method: 'DELETE',
    })
  },
}

export const newsApi = {
  list: (params?: {
    stock_symbol?: string
    skip?: number
    limit?: number
    start_time?: string
    end_time?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.stock_symbol) searchParams.set('stock_symbol', params.stock_symbol)
    if (params?.skip) searchParams.set('skip', String(params.skip))
    if (params?.limit) searchParams.set('limit', String(params.limit))
    if (params?.start_time) searchParams.set('start_time', params.start_time)
    if (params?.end_time) searchParams.set('end_time', params.end_time)
    const query = searchParams.toString()
    return fetchApi<NewsListResponse>(`/api/v1/news/${query ? `?${query}` : ''}`)
  },

  get: (id: number) => {
    return fetchApi<News>(`/api/v1/news/${id}`)
  },

  delete: (id: number) => {
    return fetchApi<void>(`/api/v1/news/${id}`, {
      method: 'DELETE',
    })
  },
}

export const aiAnalysisApi = {
  trigger: (symbol: string, market: string, interval = '1d', force = false) => {
    return fetchApi<AnalysisTriggerResponse>(`/api/v1/ai-analyses/${symbol}/trigger`, {
      method: 'POST',
      body: JSON.stringify({ interval, force, market }),
    })
  },

  getTaskStatus: (taskId: string) => {
    return fetchApi<AnalysisTaskStatus>(`/api/v1/ai-analyses/tasks/${taskId}`)
  },

  getLatest: (symbol: string, provider?: string, interval?: string) => {
    const params = new URLSearchParams()
    if (provider) params.set('provider', provider)
    if (interval) params.set('interval', interval)
    const query = params.toString()
    return fetchApi<AIAnalysis>(`/api/v1/ai-analyses/${symbol}/latest${query ? `?${query}` : ''}`)
  },

  getHistory: (symbol: string, skip = 0, limit = 30) => {
    return fetchApi<AIAnalysisListResponse>(`/api/v1/ai-analyses/${symbol}?skip=${skip}&limit=${limit}`)
  },
}
