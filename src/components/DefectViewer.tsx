import { useState, useEffect } from 'react'
import { AnalysisResult } from '../types'
import { ImageCropper } from '../utils/imageCropper'
import './DefectViewer.css'

interface DefectViewerProps {
  testImageUrl: string
  masterImageUrl: string
  analysisResult: AnalysisResult
  onClose: () => void
  imageWidth?: number
  imageHeight?: number
}

function DefectViewer({
  testImageUrl,
  masterImageUrl,
  analysisResult,
  onClose,
  imageWidth = 800,
  imageHeight = 600,
}: DefectViewerProps) {
  const [selectedSegment, setSelectedSegment] = useState<number | null>(null)
  const [croppedTestImage, setCroppedTestImage] = useState<string>('')
  const [croppedMasterImage, setCroppedMasterImage] = useState<string>('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (selectedSegment !== null) {
      loadCroppedImages(selectedSegment)
    }
  }, [selectedSegment])

  const loadCroppedImages = async (gridIndex: number) => {
    setLoading(true)
    try {
      const cropPromises = [
        ImageCropper.cropAndStoreDefect(testImageUrl, gridIndex, imageWidth, imageHeight),
        ImageCropper.cropAndStoreDefect(masterImageUrl, gridIndex, imageWidth, imageHeight),
      ]

      const [testCrop, masterCrop] = await Promise.all(cropPromises)
      setCroppedTestImage(testCrop.data)
      setCroppedMasterImage(masterCrop.data)
    } catch (error) {
      console.error('Failed to crop images:', error)
    } finally {
      setLoading(false)
    }
  }

  const selectedSegmentData = selectedSegment !== null ? analysisResult.segments[selectedSegment] : null

  return (
    <div className="defect-viewer-overlay" onClick={onClose}>
      <div className="defect-viewer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="viewer-header">
          <h2>Defect Details</h2>
          <button className="close-button" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="viewer-content">
          <div className="segment-selector">
            <h3>Grid Segments</h3>
            <div className="segment-grid">
              {analysisResult.segments.map((segment) => (
                <button
                  key={segment.segment_index}
                  className={`segment-button ${segment.anomaly_detected ? 'anomaly' : 'normal'} ${
                    selectedSegment === segment.segment_index ? 'selected' : ''
                  }`}
                  onClick={() => setSelectedSegment(segment.segment_index)}
                  title={`Region ${segment.segment_index + 1}: ${segment.verdict}`}
                >
                  {segment.segment_index + 1}
                </button>
              ))}
            </div>
          </div>

          {selectedSegmentData && (
            <div className="segment-details">
              <div className="details-header">
                <h3>Region {selectedSegmentData.segment_index + 1} Analysis</h3>
                <span className={`verdict-badge ${selectedSegmentData.verdict.toLowerCase()}`}>
                  {selectedSegmentData.verdict}
                </span>
              </div>

              <div className="segment-metrics">
                <div className="metric">
                  <span className="label">SSIM Score</span>
                  <span className="value">{selectedSegmentData.ssim_score.toFixed(4)}</span>
                </div>
                <div className="metric">
                  <span className="label">Pixel Diff</span>
                  <span className="value">{selectedSegmentData.pixel_diff_score.toFixed(2)}%</span>
                </div>
                <div className="metric">
                  <span className="label">Confidence</span>
                  <span className="value">{(selectedSegmentData.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {loading ? (
                <div className="loading">Loading crop images...</div>
              ) : (
                <div className="crop-comparison">
                  <div className="crop-column">
                    <div className="crop-label">Master Image (Cropped)</div>
                    {croppedMasterImage && (
                      <>
                        <img src={croppedMasterImage} alt="Master Crop" className="crop-image" />
                        <button
                          className="download-button"
                          onClick={() =>
                            ImageCropper.downloadCroppedImage(
                              croppedMasterImage,
                              `master-region-${selectedSegmentData.segment_index + 1}.png`
                            )
                          }
                        >
                          Download Master Crop
                        </button>
                      </>
                    )}
                  </div>

                  <div className="crop-column">
                    <div className="crop-label">Test Image (Cropped)</div>
                    {croppedTestImage && (
                      <>
                        <img src={croppedTestImage} alt="Test Crop" className="crop-image" />
                        <button
                          className="download-button"
                          onClick={() =>
                            ImageCropper.downloadCroppedImage(
                              croppedTestImage,
                              `test-region-${selectedSegmentData.segment_index + 1}.png`
                            )
                          }
                        >
                          Download Test Crop
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {selectedSegment === null && (
            <div className="empty-state">
              <p>Select a grid region to view detailed analysis and cropped images</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DefectViewer
