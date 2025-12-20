import React, { useEffect, useState, useRef } from 'react'
import { supabase } from '../lib/supabase'
import { TestImage, GoldenImage, AnalysisResult } from '../types'
import DefectViewer from './DefectViewer'
import './InspectionCard.css'

interface SwipeState {
  startX: number
  startY: number
  currentX: number
  currentY: number
  isDragging: boolean
}

function InspectionCard() {
  const [images, setImages] = useState<(TestImage & { golden_image?: GoldenImage })[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [swipeState, setSwipeState] = useState<SwipeState>({
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0,
    isDragging: false,
  })
  const [feedback, setFeedback] = useState<string>('')
  const [showDefectViewer, setShowDefectViewer] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)

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
        .limit(20)

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

  const handleMouseDown = (e: React.MouseEvent) => {
    setSwipeState((prev) => ({
      ...prev,
      startX: e.clientX,
      startY: e.clientY,
      isDragging: true,
    }))
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!swipeState.isDragging) return

    setSwipeState((prev) => ({
      ...prev,
      currentX: e.clientX,
      currentY: e.clientY,
    }))
  }

  const handleMouseUp = async () => {
    if (!swipeState.isDragging) return

    const deltaX = swipeState.currentX - swipeState.startX
    const deltaY = swipeState.currentY - swipeState.startY
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)

    if (distance > 50) {
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        if (deltaX > 0) {
          await handleSwipe('right')
        } else {
          await handleSwipe('left')
        }
      } else if (deltaY < -50) {
        await handleSwipe('up')
      }
    }

    setSwipeState((prev) => ({
      ...prev,
      currentX: prev.startX,
      currentY: prev.startY,
      isDragging: false,
    }))
  }

  const handleSwipe = async (direction: 'left' | 'right' | 'up') => {
    if (!currentImage) return

    try {
      const decision = direction === 'right' ? 'accept' : direction === 'left' ? 'reject' : 'review_later'

      const { error } = await supabase.from('defect_records').insert({
        test_image_id: currentImage.id,
        user_decision: decision,
        swipe_direction: direction,
      })

      if (error) throw error

      setFeedback(`Marked as ${decision}`)
      setTimeout(() => {
        if (currentIndex < images.length - 1) {
          setCurrentIndex(currentIndex + 1)
        } else {
          setFeedback('No more images to review!')
        }
      }, 300)
    } catch (error) {
      console.error('Failed to save decision:', error)
      setFeedback('Error saving decision')
    }
  }

  if (loading) {
    return (
      <div className="inspection-container">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading images...</p>
        </div>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="inspection-container">
        <div className="empty-state">
          <h2>No Images to Review</h2>
          <p>All pending images have been reviewed!</p>
          <button onClick={loadPendingImages} className="reload-button">
            Check for New Images
          </button>
        </div>
      </div>
    )
  }

  if (!currentImage) {
    return null
  }

  const analysisResult = currentImage.detected_defects as AnalysisResult | null
  const cardOffset = swipeState.isDragging ? swipeState.currentX - swipeState.startX : 0

  return (
    <div className="inspection-container">
      <div className="header">
        <h1>Defect Inspector</h1>
        <p className="progress">
          Image {currentIndex + 1} of {images.length}
        </p>
      </div>

      <div
        className="card-wrapper"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        ref={cardRef}
      >
        <div className="inspection-card" style={{ transform: `translateX(${cardOffset}px)` }}>
          <div className="card-images">
            <div className="image-column">
              <div className="image-label">Master Image</div>
              {currentImage.golden_image?.image_path && (
                <img
                  src={currentImage.golden_image.image_path}
                  alt="Master"
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
      </div>

      <div className="action-buttons">
        <button
          className="action-btn reject"
          onClick={() => handleSwipe('left')}
          title="Reject (Swipe Left)"
        >
          ← Reject
        </button>
        <button
          className="action-btn crop"
          onClick={() => setShowDefectViewer(true)}
          title="View and Crop Defects"
        >
          🔍 Examine Defects
        </button>
        <button
          className="action-btn review"
          onClick={() => handleSwipe('up')}
          title="Review Later (Swipe Up)"
        >
          ↑ Review Later
        </button>
        <button
          className="action-btn accept"
          onClick={() => handleSwipe('right')}
          title="Accept (Swipe Right)"
        >
          Accept →
        </button>
      </div>

      {feedback && <div className="feedback-toast">{feedback}</div>}

      <div className="instructions">
        <p>← Swipe Left: Reject | 🔍 Click to examine | ↑ Swipe Up: Review Later | Swipe Right →: Accept</p>
      </div>

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
