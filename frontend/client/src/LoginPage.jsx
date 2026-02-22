import { useState } from 'react'

export default function LoginPage({ onLogin }) {
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [isRegister, setIsRegister] = useState(false)
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    async function handleSubmit(e) {
        e.preventDefault()
        setError('')
        setLoading(true)

        const endpoint = isRegister ? '/api/register' : '/api/login'

        try {
            const res = await fetch(`http://localhost:5000${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            })
            const data = await res.json()

            if (!res.ok) {
                setError(data.error || 'Something went wrong')
            } else {
                onLogin(data.user)
            }
        } catch {
            setError('Could not connect to server')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="container">
            <div className="glow glow-1"></div>
            <div className="glow glow-2"></div>

            <h1 className="title">NeuroScan</h1>
            <p className="login-subtitle">Brain Tumor Classification</p>

            <div className="card login-card">
                <h2 className="login-heading">{isRegister ? 'Create Account' : 'Login'}</h2>

                <form onSubmit={handleSubmit} className="login-form">
                    <input
                        type="text"
                        placeholder="Username"
                        value={username}
                        onChange={e => setUsername(e.target.value)}
                        className="login-input"
                        required
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        className="login-input"
                        required
                    />

                    {error && <div className="error-msg">{error}</div>}

                    <button type="submit" className="btn btn-primary login-btn" disabled={loading}>
                        {loading ? 'Please wait...' : isRegister ? 'Register' : 'Login'}
                    </button>
                </form>

                <p className="login-toggle">
                    {isRegister ? 'Already have an account?' : "Don't have an account?"}{' '}
                    <span onClick={() => { setIsRegister(!isRegister); setError('') }}>
                        {isRegister ? 'Login' : 'Register'}
                    </span>
                </p>
            </div>
        </div>
    )
}
