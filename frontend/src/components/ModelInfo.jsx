import { useState, useEffect } from 'react'

export default function ModelInfo({ apiUrl }) {
    const [info, setInfo] = useState(null)

    useEffect(() => {
        fetch(`${apiUrl}/api/model-info`)
            .then(r => r.json())
            .then(setInfo)
            .catch(() => setInfo(null))
    }, [apiUrl])

    // Fallback data if API is not available
    const data = info || {
        name: 'SkinGuard AI',
        version: '1.0.0',
        architecture: 'EVA-02-Large + ConvNeXt-V2-Large + Swin-V2-Base Ensemble',
        parameters: '589.3M',
        accuracy: 98.68,
        auroc: 0.9995,
        num_classes: 7,
        image_size: 384,
        training: {
            dataset: 'marmal88/skin_cancer (HAM10000)',
            epochs: 40,
            training_time: '4.26 hours',
            gpu: 'NVIDIA H100 80GB',
        },
        classes: [
            'actinic_keratoses', 'basal_cell_carcinoma',
            'benign_keratosis-like_lesions', 'dermatofibroma',
            'melanocytic_Nevi', 'melanoma', 'vascular_lesions'
        ],
    }

    const details = [
        ['Architecture', data.architecture],
        ['Parameters', data.parameters],
        ['Test Accuracy', `${data.accuracy}%`],
        ['AUROC', data.auroc],
        ['Classes', data.num_classes],
        ['Input Size', `${data.image_size}×${data.image_size}px`],
        ['Dataset', data.training?.dataset],
        ['Training Epochs', data.training?.epochs],
        ['Training Time', data.training?.training_time],
        ['GPU', data.training?.gpu],
    ]

    return (
        <section className="model-section">
            <h2>🧠 Model Architecture</h2>

            <div className="model-grid">
                <div className="glass-card">
                    <div className="card-title">📋 Model Details</div>
                    {details.map(([label, value]) => (
                        <div className="model-detail" key={label}>
                            <span className="model-detail-label">{label}</span>
                            <span className="model-detail-value">{value}</span>
                        </div>
                    ))}
                </div>

                <div className="glass-card">
                    <div className="card-title">🏗️ Ensemble Architecture</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {[
                            { name: 'EVA-02-Large', params: '304.1M', input: '448px', color: '#7c3aed' },
                            { name: 'ConvNeXt-V2-Large', params: '196.4M', input: '384px', color: '#3b82f6' },
                            { name: 'Swin-V2-Base', params: '86.9M', input: '384px', color: '#06b6d4' },
                        ].map((b) => (
                            <div key={b.name} style={{
                                padding: '0.75rem 1rem',
                                background: `rgba(${b.color === '#7c3aed' ? '124,58,237' : b.color === '#3b82f6' ? '59,130,246' : '6,182,212'}, 0.08)`,
                                border: `1px solid ${b.color}33`,
                                borderRadius: 'var(--radius-sm)',
                                borderLeft: `3px solid ${b.color}`,
                            }}>
                                <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.25rem' }}>
                                    {b.name}
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    {b.params} params · {b.input} input
                                </div>
                            </div>
                        ))}

                        <div style={{
                            padding: '0.75rem',
                            background: 'rgba(16,185,129,0.08)',
                            border: '1px solid rgba(16,185,129,0.2)',
                            borderRadius: 'var(--radius-sm)',
                            textAlign: 'center',
                        }}>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                                Attention Fusion → Classifier
                            </div>
                            <div style={{ fontWeight: 700, color: 'var(--accent-green)' }}>
                                7-Class Output
                            </div>
                        </div>
                    </div>

                    <div style={{ marginTop: '1rem' }}>
                        <div className="card-title">🏷️ Supported Classes</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                            {(data.classes || []).map((cls) => (
                                <span key={cls} style={{
                                    fontSize: '0.7rem',
                                    padding: '0.25rem 0.6rem',
                                    background: 'rgba(124,58,237,0.1)',
                                    border: '1px solid rgba(124,58,237,0.2)',
                                    borderRadius: '100px',
                                    color: 'var(--text-secondary)',
                                }}>
                                    {cls}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}
