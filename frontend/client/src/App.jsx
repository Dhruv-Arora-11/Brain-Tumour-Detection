import { useState } from 'react'
import './App.css'

const CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
const CLASS_COLORS = ['#ff6b6b', '#ffa94d', '#51cf66', '#748ffc']

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  function handleFile(f) {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  function handleDrag(e) {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true)
    if (e.type === 'dragleave') setDragActive(false)
  }

  function handleDrop(e) {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  function handleInputChange(e) {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

async function handleSubmit() {
  if (!file) return
  setLoading(true)
  setResult(null)

  const formData = new FormData()
  formData.append('file', file)   // ✅ must match backend

  try {
    const res = await fetch('https://braintumourdetection-ymya.onrender.com//predict', {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      throw new Error("Server error")
    }

    const data = await res.json()

    const classIndex = data.prediction

    setResult({
      classIndex,
      className: CLASS_NAMES[classIndex],
      confidence: 1,
      probabilities: CLASS_NAMES.map((_, i) =>
        i === classIndex ? 1 : 0
      )
    })

  } catch (err) {
    console.error(err)
    setResult({ error: 'Could not connect to server' })
  } finally {
    setLoading(false)
  }
}

  function handleReset() {
    setFile(null)
    setPreview(null)
    setResult(null)
  }

  return (
    <div className="container">
      <div className="glow glow-1"></div>
      <div className="glow glow-2"></div>

      <h1 className="title">
        <span className="title-icon">🧠</span> NeuroScan
      </h1>
      <p className="subtitle">AI-Powered Brain Tumor Classification</p>

      <div className="card">
        {!preview ? (
          <div
            className={`dropzone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleInputChange}
              hidden
            />
            <div className="dropzone-icon">📁</div>
            <p className="dropzone-text">Drag & drop an MRI scan here</p>
            <p className="dropzone-hint">or click to browse</p>
          </div>
        ) : (
          <div className="preview-section">
            <div className="image-wrapper">
              <img src={preview} alt="MRI Preview" className="preview-image" />
            </div>

            {loading && (
              <div className="loader">
                <div className="spinner"></div>
                <p>Analyzing scan...</p>
              </div>
            )}

            {result && !result.error && (
              <div className="results">
                <h2 className="results-title">Diagnosis</h2>
                <div
                  className="predicted-class"
                  style={{ color: CLASS_COLORS[result.classIndex] }}
                >
                  {result.className}
                </div>
                <div className="confidence">
                  {(result.confidence * 100).toFixed(1)}% Confidence
                </div>
                <div className="probabilities">
                  {CLASS_NAMES.map((name, i) => (
                    <div key={name} className="prob-row">
                      <span className="prob-label">{name}</span>
                      <div className="prob-bar-bg">
                        <div
                          className="prob-bar-fill"
                          style={{
                            width: `${(result.probabilities[i] * 100).toFixed(1)}%`,
                            background: CLASS_COLORS[i],
                          }}
                        ></div>
                      </div>
                      <span className="prob-value">
                        {(result.probabilities[i] * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result && result.error && (
              <div className="error-msg">{result.error}</div>
            )}

            <div className="actions">
              {!loading && !result && (
                <button className="btn btn-primary" onClick={handleSubmit}>
                  Analyze Scan
                </button>
              )}
              <button className="btn btn-secondary" onClick={handleReset}>
                New Scan
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
