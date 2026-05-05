import { Component, type ReactNode, type ErrorInfo } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught:', error, errorInfo)
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null })
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          color: '#ef4444',
          textAlign: 'center',
          minHeight: '200px'
        }}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginBottom: '1rem' }}>
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <h3 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 600, color: '#e5e5e5' }}>
            Something went wrong
          </h3>
          <p style={{ margin: '0 0 1rem', fontSize: '0.75rem', color: '#737373', maxWidth: '400px', lineHeight: 1.5 }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={this.handleReset}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              border: '1px solid rgba(234, 88, 12, 0.3)',
              background: 'rgba(234, 88, 12, 0.1)',
              color: '#f97316',
              fontSize: '0.8125rem',
              cursor: 'pointer'
            }}
          >
            Try Again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}