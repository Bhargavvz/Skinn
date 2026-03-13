import { useState, useRef } from 'react'

export default function Upload({ onAnalyze, loading }) {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [dragOver, setDragOver] = useState(false)
    const fileInputRef = useRef(null)

    const handleFile = (f) => {
        if (f && f.type.startsWith('image/')) {
            setFile(f)
            setPreview(URL.createObjectURL(f))
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragOver(false)
        const f = e.dataTransfer.files[0]
        handleFile(f)
    }

    const handleDragOver = (e) => {
        e.preventDefault()
        setDragOver(true)
    }

    const handleDragLeave = () => setDragOver(false)

    const handleClick = () => fileInputRef.current?.click()

    const handleChange = (e) => handleFile(e.target.files[0])

    const handleSubmit = () => {
        if (file && !loading) onAnalyze(file)
    }

    return (
        <section className="upload-section">
            <h2>Analyze <span>Skin Lesion</span></h2>
            <p className="upload-subtitle">
                Upload a dermoscopic image for AI-powered classification
            </p>

            <div
                className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={handleClick}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleChange}
                />

                {!preview ? (
                    <>
                        <div className="upload-zone-icon">📤</div>
                        <h3>Drop your image here</h3>
                        <p>or click to browse — supports JPEG, PNG, WEBP</p>
                    </>
                ) : (
                    <div className="upload-preview">
                        <img src={preview} alt="Preview" />
                        <p className="preview-name">{file?.name}</p>
                    </div>
                )}
            </div>

            {preview && (
                <div style={{ textAlign: 'center' }}>
                    <button
                        className="btn-analyze"
                        onClick={handleSubmit}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <div className="spinner"></div>
                                Analyzing...
                            </>
                        ) : (
                            <>🔬 Analyze Image</>
                        )}
                    </button>
                </div>
            )}
        </section>
    )
}
