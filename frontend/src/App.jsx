import { useState } from 'react'
import Hero from './components/Hero'
import Upload from './components/Upload'
import Results from './components/Results'
import History from './components/History'
import ModelInfo from './components/ModelInfo'
import Footer from './components/Footer'
import './App.css'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [history, setHistory] = useState([])
  const [activeSection, setActiveSection] = useState('home')

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  const handleAnalyze = async (file) => {
    setLoading(true)
    setResult(null)
    setUploadedImage(URL.createObjectURL(file))

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      setActiveSection('results')

      // Add to history
      setHistory(prev => [{
        id: Date.now(),
        image: URL.createObjectURL(file),
        prediction: data.prediction,
        risk: data.risk_assessment,
        timestamp: new Date().toLocaleString(),
      }, ...prev].slice(0, 10))

    } catch (error) {
      console.error('Analysis failed:', error)
      setResult({ error: error.message })
    } finally {
      setLoading(false)
    }
  }

  const handleNewAnalysis = () => {
    setResult(null)
    setUploadedImage(null)
    setActiveSection('upload')
  }

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-brand" onClick={() => setActiveSection('home')}>
          <span className="nav-icon">🛡️</span>
          <span className="nav-title">SkinGuard<span className="nav-ai">AI</span></span>
        </div>
        <div className="nav-links">
          <button className={`nav-link ${activeSection === 'home' ? 'active' : ''}`} onClick={() => setActiveSection('home')}>Home</button>
          <button className={`nav-link ${activeSection === 'upload' ? 'active' : ''}`} onClick={() => setActiveSection('upload')}>Analyze</button>
          {history.length > 0 && (
            <button className={`nav-link ${activeSection === 'history' ? 'active' : ''}`} onClick={() => setActiveSection('history')}>
              History <span className="badge">{history.length}</span>
            </button>
          )}
          <button className={`nav-link ${activeSection === 'model' ? 'active' : ''}`} onClick={() => setActiveSection('model')}>Model</button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {activeSection === 'home' && (
          <Hero onGetStarted={() => setActiveSection('upload')} />
        )}

        {activeSection === 'upload' && !result && (
          <Upload onAnalyze={handleAnalyze} loading={loading} />
        )}

        {(activeSection === 'results' || (activeSection === 'upload' && result)) && result && (
          <Results
            result={result}
            uploadedImage={uploadedImage}
            onNewAnalysis={handleNewAnalysis}
          />
        )}

        {activeSection === 'history' && (
          <History items={history} />
        )}

        {activeSection === 'model' && (
          <ModelInfo apiUrl={API_URL} />
        )}
      </main>

      <Footer />
    </div>
  )
}

export default App
