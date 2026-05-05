import type { BrokerRating, BrokerRatingConsensus } from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export const brokerRatingApi = {
  /**
   * Import broker ratings from JSON body
   */
  async import(data: any[]): Promise<{ message: string }> {
    const res = await fetch(`${API_BASE}/api/v1/broker-ratings/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Import failed');
    return res.json();
  },

  /**
   * Get ratings for a stock
   */
  async list(stockId: number, limit = 50): Promise<BrokerRating[]> {
    const res = await fetch(`${API_BASE}/api/v1/broker-ratings/list`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stock_id: stockId, limit }),
    });
    if (!res.ok) throw new Error('Failed to fetch ratings');
    return res.json();
  },

  /**
   * Get ratings for multiple stocks in one batch call
   */
  async listBatch(stockIds: number[], limit = 5): Promise<Record<number, BrokerRating[]>> {
    const res = await fetch(`${API_BASE}/api/v1/broker-ratings/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stock_ids: stockIds, limit }),
    });
    if (!res.ok) throw new Error('Failed to fetch batch ratings');
    return res.json();
  },

  /**
   * Get latest rating per broker for a stock
   */
  async getLatest(stockId: number): Promise<Record<string, BrokerRating>> {
    const res = await fetch(`${API_BASE}/api/v1/broker-ratings/latest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stock_id: stockId }),
    });
    if (!res.ok) throw new Error('Failed to fetch latest ratings');
    return res.json();
  },

  /**
   * Get consensus rating for a stock
   */
  async getConsensus(stockId: number): Promise<BrokerRatingConsensus> {
    const res = await fetch(`${API_BASE}/api/v1/broker-ratings/consensus`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stock_id: stockId }),
    });
    if (!res.ok) throw new Error('Failed to fetch consensus');
    return res.json();
  },
};
