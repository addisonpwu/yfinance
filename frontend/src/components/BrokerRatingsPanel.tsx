import type { BrokerRating } from '../types/api';

interface BrokerRatingsPanelProps {
  ratings: BrokerRating[];
  loading: boolean;
}

const CHINESE_RATING_MAP: Record<string, string> = {
  '强烈推荐': 'STRONG_BUY',
  '买入': 'BUY',
  '增持': 'OVERWEIGHT',
  '中性': 'NEUTRAL',
  '减持': 'UNDERWEIGHT',
  '卖出': 'SELL',
};

const RATING_BADGES: Record<string, string> = {
  'STRONG_BUY': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  'BUY': 'bg-green-500/10 text-green-400 border-green-500/20',
  'OVERWEIGHT': 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  'NEUTRAL': 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  'UNDERWEIGHT': 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  'SELL': 'bg-red-500/10 text-red-400 border-red-500/20',
};

const RATING_LABELS: Record<string, string> = {
  'STRONG_BUY': '強烈買入',
  'BUY': '買入',
  'OVERWEIGHT': '增持',
  'NEUTRAL': '中性',
  'UNDERWEIGHT': '減持',
  'SELL': '賣出',
};

export function BrokerRatingsPanel({ ratings, loading }: BrokerRatingsPanelProps) {
  if (loading) {
    return <div className="skeleton" style={{ height: '80px' }} />;
  }

  if (!ratings.length) {
    return <div className="empty-result">無券商評級數據</div>;
  }

  return (
    <div className="broker-ratings-list">
      {ratings.map((r) => {
        const normalized = CHINESE_RATING_MAP[r.rating] || r.rating;
        const badgeClass = RATING_BADGES[normalized] || 'bg-neutral-500/10 text-neutral-400 border-neutral-500/20';
        const label = RATING_LABELS[normalized] || r.rating;

        return (
          <div key={r.id} className="broker-rating-item">
            <div className="broker-rating-header">
              <span className="broker-rating-broker">{r.broker}</span>
              <span className={`broker-rating-badge ${badgeClass}`}>{label}</span>
            </div>
            {r.reason && <div className="broker-rating-reason">{r.reason}</div>}
            <div className="broker-rating-date">{r.rating_date}</div>
          </div>
        );
      })}
    </div>
  );
}
