import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import { TestImage, GoldenImage, AnalysisResult } from '../types'
import DefectViewer from './DefectViewer'
import './InspectionCard.css'

function InspectionCard() {
  const [images, setImages] = useState<(TestImage & { golden_image?: GoldenImage })[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [feedback, setFeedback] = useState<string>('')
  const [showDefectViewer, setShowDefectViewer] = useState(false)
  const [notes, setNotes] = useState<string>('')
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    loadPendingImages()
  }, [])

  const loadPendingImages = async () => {
    try {
      setLoading(true)
      const { data, error } = await supabase
        .from('test_images')
        .select(`
          *,
          golden_image:golden_image_id (*)
        `)
        .eq('processed', true)
        .order('upload_timestamp', { ascending: false })
        .limit(50)

      if (error) throw error

      const imageIds = new Set(
        (
          await supabase
            .from('defect_records')
            .select('test_image_id')
        ).data?.map((r) => r.test_image_id) || []
      )

      const unreviewed = (data || []).filter((img) => !imageIds.has(img.id))
      setImages(unreviewed)
      setCurrentIndex(0)
    } catch (error) {
      console.error('Failed to load images:', error)
      setFeedback('Failed to load images')
    } finally {
      setLoading(false)
    }
  }

  const currentImage = images[currentIndex]

  const handleDecision = async (decision: 'accept' | 'reject' | 'review_later') => {
    if (!currentImage || processing) return

    try {
      setProcessing(true)
      const { error } = await supabase.from('defect_records').insert({
        test_image_id: currentImage.id,
        user_decision: decision,
        notes: notes.trim() || null,
      })

      if (error) throw error

      setFeedback(`Inspection result recorded: ${decision.replace('_', ' ')}`)
      setNotes('')

      setTimeout(() => {
        setFeedback('')
        if (currentIndex < images.length - 1) {
          setCurrentIndex(currentIndex + 1)
        } else {
          setFeedback('All images have been reviewed')
          loadPendingImages()
        }
      }, 1500)
    } catch (error) {
      console.error('Failed to save decision:', error)
      setFeedback('Error recording inspection result')
    } finally {
      setProcessing(false)
    }
  }

  const navigateImage = (direction: 'prev' | 'next') => {
    if (direction === 'prev' && currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
      setNotes('')
    } else if (direction === 'next' && currentIndex < images.length - 1) {
      setCurrentIndex(currentIndex + 1)
      setNotes('')
    }
  }

  if (loading) {
    return (
      <div className="inspection-container">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading inspection queue...</p>
        </div>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="inspection-container">
        <div className="empty-state">
          <h2>No Images in Queue</h2>
          <p>All inspection items have been processed. Upload new test images to begin.</p>
          <button onClick={loadPendingImages} className="reload-button">
            Refresh Queue
          </button>
        </div>
      </div>
    )
  }

  if (!currentImage) {
    return null
  }

  const analysisResult = currentImage.detected_defects as AnalysisResult | null

  return (
    <div className="inspection-container">
      <div className="header">
        <h1>Image Inspection</h1>
        <div className="progress-info">
          <span className="progress">
            Image {currentIndex + 1} of {images.length}
          </span>
          <div className="navigation-controls">
            <button
              className="nav-btn"
              onClick={() => navigateImage('prev')}
              disabled={currentIndex === 0}
              title="Previous Image"
            >
              Previous
            </button>
            <button
              className="nav-btn"
              onClick={() => navigateImage('next')}
              disabled={currentIndex === images.length - 1}
              title="Next Image"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      <div className="inspection-card">
        <div className="card-images">
          <div className="image-column">
            <div className="image-label">Reference Image</div>
            {currentImage.golden_image?.image_path && (
              <img
                src={currentImage.golden_image.image_path}
                alt="Reference"
                className="inspection-image"
              />
            )}
          </div>

          <div className="divider"></div>

          <div className="image-column">
            <div className="image-label">Test Image</div>
            {currentImage.image_path && (
              <img
                src={currentImage.image_path}
                alt="Test"
                className="inspection-image"
              />
            )}
          </div>
        </div>

        {analysisResult && (
          <div className="analysis-panel">
            <div className="analysis-header">
              <h3>Analysis Results</h3>
              <span className={`verdict-badge ${analysisResult.verdict.toLowerCase()}`}>
                {analysisResult.verdict}
              </span>
            </div>

            <div className="metrics">
              <div className="metric">
                <span className="metric-label">Defect Score</span>
                <span className="metric-value">
                  {analysisResult.overall_defect_score.toFixed(1)}%
                </span>
                <div className="metric-bar">
                  <div
                    className="metric-fill"
                    style={{ width: `${analysisResult.overall_defect_score}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric">
                <span className="metric-label">Anomaly Count</span>
                <span className="metric-value">{analysisResult.anomaly_count} / 9</span>
              </div>

              <div className="metric">
                <span className="metric-label">Confidence</span>
                <span className="metric-value">
                  {(analysisResult.average_confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="grid-visualization">
              <h4>Affected Regions (3x3 Grid)</h4>
              <div className="grid">
                {Array.from({ length: 9 }).map((_, idx) => (
                  <div
                    key={idx}
                    className={`grid-cell ${
                      analysisResult.defect_locations.includes(idx) ? 'anomaly' : 'normal'
                    }`}
                    title={`Region ${idx + 1}`}
                  >
                    {idx + 1}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="decision-panel">
        <div className="notes-section">
          <label htmlFor="inspection-notes">Inspection Notes:</label>
          <textarea
            id="inspection-notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add notes about this inspection (optional)"
            rows={3}
            disabled={processing}
          />
        </div>

        <div className="action-buttons">
          <button
            className="action-btn reject"
            onClick={() => handleDecision('reject')}
            disabled={processing}
            title="Mark as defective"
          >
            Reject
          </button>
          <button
            className="action-btn review"
            onClick={() => handleDecision('review_later')}
            disabled={processing}
            title="Flag for additional review"
          >
            Review Later
          </button>
          <button
            className="action-btn accept"
            onClick={() => handleDecision('accept')}
            disabled={processing}
            title="Mark as acceptable"
          >
            Accept
          </button>
          <button
            className="action-btn examine"
            onClick={() => setShowDefectViewer(true)}
            disabled={processing}
            title="View detailed analysis"
          >
            Examine Details
          </button>
        </div>
      </div>

      {feedback && <div className="feedback-toast">{feedback}</div>}

      {showDefectViewer && analysisResult && currentImage.golden_image && (
        <DefectViewer
          testImageUrl={currentImage.image_path}
          masterImageUrl={currentImage.golden_image.image_path}
          analysisResult={analysisResult}
          onClose={() => setShowDefectViewer(false)}
        />
      )}
    </div>
  )
}

export default InspectionCard
