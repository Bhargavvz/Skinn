const RISK_ICONS = { HIGH: '🔴', MEDIUM: '🟠', LOW: '🟢' }

export default function History({ items }) {
    if (!items || items.length === 0) {
        return (
            <section className="history-section">
                <h2>📋 Analysis History</h2>
                <div className="glass-card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>
                        No analyses yet. Start by uploading an image!
                    </p>
                </div>
            </section>
        )
    }

    return (
        <section className="history-section">
            <h2>📋 Analysis History</h2>
            <div className="history-grid">
                {items.map((item) => (
                    <div className="history-card" key={item.id}>
                        <img src={item.image} alt={item.prediction?.class_name} />
                        <div className="history-card-body">
                            <h4>
                                {RISK_ICONS[item.risk?.level] || '⚪'} {item.prediction?.class_name}
                            </h4>
                            <div className="confidence-text">
                                Confidence: {((item.prediction?.confidence || 0) * 100).toFixed(1)}%
                            </div>
                            <div className={`risk-badge risk-${item.risk?.level || 'LOW'}`} style={{ fontSize: '0.7rem', padding: '0.2rem 0.6rem' }}>
                                {item.risk?.level} Risk
                            </div>
                            <div className="timestamp">{item.timestamp}</div>
                        </div>
                    </div>
                ))}
            </div>
        </section>
    )
}
