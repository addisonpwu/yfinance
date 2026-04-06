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
