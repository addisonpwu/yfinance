interface MobileTabsProps {
  activeTab: number;
  onTabChange: (index: number) => void;
}

export function MobileTabs({ activeTab, onTabChange }: MobileTabsProps) {
  return (
    <div className="mobile-tab-indicator">
      <button
        className={`mobile-tab-btn ${activeTab === 0 ? 'active' : ''}`}
        onClick={() => onTabChange(0)}
        aria-label="Stocks tab"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <span>Stocks</span>
      </button>
      <button
        className={`mobile-tab-btn ${activeTab === 1 ? 'active' : ''}`}
        onClick={() => onTabChange(1)}
        aria-label="AI analysis tab"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
        <span>AI</span>
      </button>
      <button
        className={`mobile-tab-btn ${activeTab === 2 ? 'active' : ''}`}
        onClick={() => onTabChange(2)}
        aria-label="News tab"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
        </svg>
        <span>News</span>
      </button>
    </div>
  );
}