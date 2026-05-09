interface ErrorBannerProps {
  error: string | null;
  onDismiss: () => void;
}

export function ErrorBanner({ error, onDismiss }: ErrorBannerProps) {
  if (!error) return null;

  return (
    <div style={{
      background: '#ef4444', color: 'white', padding: '0.5rem 1rem',
      textAlign: 'center', fontSize: '0.75rem', cursor: 'pointer'
    }} onClick={onDismiss}>
      {error} (click to dismiss)
    </div>
  );
}