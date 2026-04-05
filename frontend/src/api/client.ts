import type { Stock, StockListResponse, News, NewsListResponse } from '../types/api'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

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
  list: (params?: { market?: string; skip?: number; limit?: number }) => {
    const searchParams = new URLSearchParams()
    if (params?.market) searchParams.set('market', params.market)
    if (params?.skip) searchParams.set('skip', String(params.skip))
    if (params?.limit) searchParams.set('limit', String(params.limit))
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
