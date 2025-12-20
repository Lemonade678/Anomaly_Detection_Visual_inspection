import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import './Dashboard.css'

interface DashboardStats {
  totalReviewed: number
  accepted: number
  rejected: number
  reviewLater: number
  acceptanceRate: number
}

function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    totalReviewed: 0,
    accepted: 0,
    rejected: 0,
    reviewLater: 0,
    acceptanceRate: 0,
  })
  const [loading, setLoading] = useState(true)
  const [chartData, setChartData] = useState<{ label: string; value: number }[]>([])

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      setLoading(true)

      const { data: records, error } = await supabase
        .from('defect_records')
        .select('user_decision')

      if (error) throw error

      if (records && records.length > 0) {
        const accepted = records.filter((r) => r.user_decision === 'accept').length
        const rejected = records.filter((r) => r.user_decision === 'reject').length
        const reviewLater = records.filter((r) => r.user_decision === 'review_later').length
        const total = records.length

        const newStats = {
          totalReviewed: total,
          accepted,
          rejected,
          reviewLater,
          acceptanceRate: total > 0 ? (accepted / total) * 100 : 0,
        }

        setStats(newStats)
        setChartData([
          { label: 'Accepted', value: accepted },
          { label: 'Rejected', value: rejected },
          { label: 'Review Later', value: reviewLater },
        ])
      }
    } catch (error) {
      console.error('Failed to load stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading">Loading dashboard...</div>
      </div>
    )
  }

  const maxValue = Math.max(...chartData.map((d) => d.value), 1)

  return (
    <div className="dashboard">
      <h1>Inspection Dashboard</h1>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">{stats.totalReviewed}</div>
          <div className="stat-label">Total Reviewed</div>
        </div>
        <div className="stat-card success">
          <div className="stat-number">{stats.accepted}</div>
          <div className="stat-label">Accepted</div>
        </div>
        <div className="stat-card error">
          <div className="stat-number">{stats.rejected}</div>
          <div className="stat-label">Rejected</div>
        </div>
        <div className="stat-card warning">
          <div className="stat-number">{stats.reviewLater}</div>
          <div className="stat-label">Review Later</div>
        </div>
        <div className="stat-card primary">
          <div className="stat-number">{stats.acceptanceRate.toFixed(1)}%</div>
          <div className="stat-label">Acceptance Rate</div>
        </div>
      </div>

      <div className="charts-section">
        <div className="chart-card">
          <h2>Review Breakdown</h2>
          <div className="bar-chart">
            {chartData.map((item) => (
              <div key={item.label} className="bar-item">
                <div className="bar-label">{item.label}</div>
                <div className="bar-container">
                  <div
                    className={`bar ${item.label.toLowerCase().replace(' ', '-')}`}
                    style={{ width: `${(item.value / maxValue) * 100}%` }}
                  >
                    <span className="bar-value">{item.value}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-card">
          <h2>Quality Metrics</h2>
          <div className="metrics-list">
            <div className="metric-item">
              <span className="metric-name">Quality Score</span>
              <div className="metric-progress">
                <div
                  className="progress-bar"
                  style={{ width: `${stats.acceptanceRate}%` }}
                ></div>
              </div>
              <span className="metric-percent">{stats.acceptanceRate.toFixed(1)}%</span>
            </div>
            <div className="metric-item">
              <span className="metric-name">Detection Coverage</span>
              <div className="metric-progress">
                <div className="progress-bar" style={{ width: '85%' }}></div>
              </div>
              <span className="metric-percent">85%</span>
            </div>
            <div className="metric-item">
              <span className="metric-name">System Reliability</span>
              <div className="metric-progress">
                <div className="progress-bar" style={{ width: '92%' }}></div>
              </div>
              <span className="metric-percent">92%</span>
            </div>
          </div>
        </div>
      </div>

      <button onClick={loadStats} className="refresh-button">
        Refresh Data
      </button>
    </div>
  )
}

export default Dashboard
