import React, { useEffect, useState } from 'react'
import './App.css'
import { supabase } from './lib/supabase'
import InspectionCard from './components/InspectionCard'
import Dashboard from './components/Dashboard'
import AdminPanel from './components/AdminPanel'

type AppView = 'inspection' | 'dashboard' | 'history' | 'admin'

function App() {
  const [currentView, setCurrentView] = useState<AppView>('inspection')
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession()
        setUser(session?.user || null)
        setLoading(false)
      } catch (error) {
        console.error('Auth check failed:', error)
        setLoading(false)
      }
    }

    checkAuth()

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null)
    })

    return () => {
      authListener?.subscription.unsubscribe()
    }
  }, [])

  if (loading) {
    return (
      <div className="app-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    )
  }

  if (!user) {
    return <LoginScreen setUser={setUser} />
  }

  return (
    <div className="app">
      <nav className="app-nav">
        <div className="nav-left">
          <h1 className="app-title">Defect Inspector</h1>
        </div>
        <div className="nav-center">
          <button
            className={`nav-button ${currentView === 'inspection' ? 'active' : ''}`}
            onClick={() => setCurrentView('inspection')}
          >
            Inspect
          </button>
          <button
            className={`nav-button ${currentView === 'history' ? 'active' : ''}`}
            onClick={() => setCurrentView('history')}
          >
            History
          </button>
          <button
            className={`nav-button ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => setCurrentView('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={`nav-button ${currentView === 'admin' ? 'active' : ''}`}
            onClick={() => setCurrentView('admin')}
          >
            Templates
          </button>
        </div>
        <div className="nav-right">
          <button
            className="logout-button"
            onClick={async () => {
              await supabase.auth.signOut()
              setUser(null)
            }}
          >
            Logout
          </button>
        </div>
      </nav>

      <main className="app-main">
        {currentView === 'inspection' && <InspectionCard />}
        {currentView === 'dashboard' && <Dashboard />}
        {currentView === 'history' && <HistoryView />}
        {currentView === 'admin' && <AdminPanel />}
      </main>
    </div>
  )
}

function LoginScreen({ setUser }: { setUser: (user: any) => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const { data, error: authError } = isSignUp
        ? await supabase.auth.signUp({ email, password })
        : await supabase.auth.signInWithPassword({ email, password })

      if (authError) throw authError

      setUser(data.user)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>Defect Inspector</h1>
        <p className="subtitle">Tinder-style defect detection system</p>

        <form onSubmit={handleAuth}>
          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={loading}
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={loading}
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button type="submit" disabled={loading} className="submit-button">
            {loading ? 'Loading...' : isSignUp ? 'Sign Up' : 'Sign In'}
          </button>
        </form>

        <p className="toggle-auth">
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button
            type="button"
            onClick={() => setIsSignUp(!isSignUp)}
            className="toggle-button"
          >
            {isSignUp ? 'Sign In' : 'Sign Up'}
          </button>
        </p>
      </div>
    </div>
  )
}

function HistoryView() {
  const [records, setRecords] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const { data, error } = await supabase
          .from('defect_records')
          .select('*')
          .order('review_timestamp', { ascending: false })
          .limit(50)

        if (error) throw error
        setRecords(data || [])
      } catch (error) {
        console.error('Failed to load history:', error)
      } finally {
        setLoading(false)
      }
    }

    loadHistory()
  }, [])

  if (loading) {
    return <div className="view-loading">Loading history...</div>
  }

  return (
    <div className="history-view">
      <h2>Inspection History</h2>
      <div className="records-list">
        {records.length === 0 ? (
          <p className="empty-state">No inspection history yet</p>
        ) : (
          records.map((record) => (
            <div key={record.id} className="history-item">
              <div className="item-header">
                <span className={`decision-badge ${record.user_decision}`}>
                  {record.user_decision}
                </span>
                <span className="timestamp">
                  {new Date(record.review_timestamp).toLocaleString()}
                </span>
              </div>
              {record.notes && <p className="item-notes">{record.notes}</p>}
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default App
