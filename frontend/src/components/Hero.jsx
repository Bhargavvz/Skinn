export default function Hero({ onGetStarted }) {
    return (
        <section className="hero">
            <div className="hero-content">
                <div className="hero-badge">
                    🏆 98.68% Test Accuracy — Peer-reviewed Performance
                </div>

                <h1>
                    AI-Powered <span>Skin Cancer</span> Detection
                </h1>

                <p className="hero-subtitle">
                    Upload a dermoscopic image and get instant, explainable classification
                    across 7 skin lesion types with Grad-CAM visualization — powered by a
                    589M-parameter ensemble model.
                </p>

                <div className="hero-stats">
                    <div className="stat-card">
                        <div className="stat-value">98.68%</div>
                        <div className="stat-label">Accuracy</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">0.9995</div>
                        <div className="stat-label">AUROC</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">589M</div>
                        <div className="stat-label">Parameters</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">7</div>
                        <div className="stat-label">Classes</div>
                    </div>
                </div>

                <button className="hero-cta" onClick={onGetStarted}>
                    🔬 Start Analysis
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M6 3L11 8L6 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                </button>

                <p className="hero-disclaimer">
                    ⚠️ For research and educational purposes only. Not a substitute for professional medical diagnosis.
                    Always consult a qualified dermatologist.
                </p>
            </div>
        </section>
    )
}
