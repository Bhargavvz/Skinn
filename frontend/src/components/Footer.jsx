export default function Footer() {
    return (
        <footer className="footer">
            <p>
                <strong>SkinGuard AI</strong> — Built with ❤️ using PyTorch, FastAPI, and React
            </p>
            <p style={{ marginTop: '0.25rem' }}>
                EVA-02 + ConvNeXt-V2 + Swin-V2 Ensemble · 98.68% Accuracy · 0.9995 AUROC
            </p>
            <p style={{ marginTop: '0.5rem', fontSize: '0.7rem' }}>
                ⚠️ For educational/research purposes only. Not a medical diagnostic tool.
            </p>
        </footer>
    )
}
