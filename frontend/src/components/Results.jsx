const RISK_ICONS = { HIGH: '🔴', MEDIUM: '🟠', LOW: '🟢' }

export default function Results({ result, uploadedImage, onNewAnalysis }) {
    if (result?.error) {
        return (
            <section className="results-section">
                <div className="error-card">
                    <h3>❌ Analysis Failed</h3>
                    <p>{result.error}</p>
                    <button className="btn-new" onClick={onNewAnalysis} style={{ marginTop: '1rem' }}>
                        Try Again
                    </button>
                </div>
            </section>
        )
    }

    const { prediction, risk_assessment, probabilities, top_3, gradcam, metadata } = result
    const riskLevel = risk_assessment?.level || 'LOW'
    const riskIcon = RISK_ICONS[riskLevel] || '⚪'

    // Sort probabilities for chart
    const sortedProbs = Object.entries(probabilities || {})
        .sort(([, a], [, b]) => b - a)

    return (
        <section className="results-section">
            <div className="results-header">
                <h2>🔬 Analysis Results</h2>
                <button className="btn-new" onClick={onNewAnalysis}>
                    + New Analysis
                </button>
            </div>

            <div className="results-grid">
                {/* Prediction Card */}
                <div className="glass-card">
                    <div className="card-title">🎯 Diagnosis</div>
                    <div className="prediction-main">
                        <div style={{ flex: 1 }}>
                            <div className="prediction-name">{prediction?.class_name}</div>
                            <div className="prediction-desc">{prediction?.description}</div>
                        </div>
                        <div className="confidence-badge">
                            {((prediction?.confidence || 0) * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div className={`risk-badge risk-${riskLevel}`}>
                        {riskIcon} {riskLevel} Risk
                    </div>
                    <div className="risk-action">
                        <strong>Recommended:</strong> {risk_assessment?.action}
                    </div>
                </div>

                {/* Uploaded Image */}
                <div className="glass-card image-card">
                    <div className="card-title">📷 Uploaded Image</div>
                    {uploadedImage && <img src={uploadedImage} alt="Uploaded lesion" />}
                </div>

                {/* Confidence Chart */}
                <div className="glass-card">
                    <div className="card-title">📊 Confidence Scores</div>
                    <div className="confidence-bars">
                        {sortedProbs.map(([name, prob], i) => (
                            <div className="conf-bar-row" key={name}>
                                <span className="conf-bar-label" title={name}>{name}</span>
                                <div className="conf-bar-track">
                                    <div
                                        className={`conf-bar-fill ${i === 0 ? 'top' : ''}`}
                                        style={{
                                            width: `${prob * 100}%`,
                                            animationDelay: `${i * 0.1}s`,
                                            opacity: prob > 0.01 ? 1 : 0.3,
                                        }}
                                    />
                                </div>
                                <span className="conf-bar-value">{(prob * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Grad-CAM */}
                <div className="glass-card image-card">
                    <div className="card-title">🔥 Grad-CAM Heatmap</div>
                    {gradcam ? (
                        <img src={`data:image/png;base64,${gradcam}`} alt="Grad-CAM visualization" />
                    ) : (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                            Grad-CAM visualization not available
                        </p>
                    )}
                </div>

                {/* Top Predictions */}
                <div className="glass-card full-width">
                    <div className="card-title">🏆 Top Predictions</div>
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                        {(top_3 || []).map((item, i) => {
                            const colors = ['#7c3aed', '#3b82f6', '#06b6d4']
                            return (
                                <div key={i} style={{
                                    flex: 1,
                                    minWidth: '180px',
                                    background: `rgba(${i === 0 ? '124,58,237' : i === 1 ? '59,130,246' : '6,182,212'}, 0.08)`,
                                    border: `1px solid rgba(${i === 0 ? '124,58,237' : i === 1 ? '59,130,246' : '6,182,212'}, 0.2)`,
                                    borderRadius: 'var(--radius-md)',
                                    padding: '1rem',
                                }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                                        #{i + 1}
                                    </div>
                                    <div style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                                        {item.class_name}
                                    </div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 800, color: colors[i] }}>
                                        {(item.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                            )
                        })}
                    </div>

                    {metadata && (
                        <div style={{
                            marginTop: '1rem',
                            paddingTop: '1rem',
                            borderTop: '1px solid rgba(255,255,255,0.05)',
                            fontSize: '0.75rem',
                            color: 'var(--text-muted)',
                            display: 'flex',
                            gap: '1.5rem',
                            flexWrap: 'wrap',
                        }}>
                            <span>⏱️ {metadata.inference_time_ms?.toFixed(0)}ms</span>
                            <span>📁 {metadata.filename}</span>
                            <span>🤖 v{metadata.model_version}</span>
                        </div>
                    )}
                </div>
            </div>
        </section>
    )
}
