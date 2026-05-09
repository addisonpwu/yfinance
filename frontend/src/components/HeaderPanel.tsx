interface HeaderPanelProps {
  stockTotal: number;
}

export function HeaderPanel({ stockTotal }: HeaderPanelProps) {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <div>
              <span className="logo-text">Stock<span>Analysis</span></span>
              <div className="status-badge">
                <span className="status-dot"></span>
                <span>System Online</span>
                <span style={{ color: '#404040' }}>|</span>
                <span>{stockTotal} Assets</span>
              </div>
            </div>
          </div>
          <div className="status-badge" style={{ fontSize: '0.6875rem', background: '#1a1a1a', padding: '0.375rem 0.75rem', borderRadius: '9999px', border: '1px solid #242424' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2">
              <ellipse cx="12" cy="5" rx="9" ry="3" />
              <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
              <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
            </svg>
            <span>PostgreSQL + FastAPI</span>
          </div>
        </div>
      </div>
    </header>
  );
}