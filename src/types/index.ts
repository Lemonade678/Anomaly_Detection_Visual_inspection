export interface GoldenImage {
  id: string
  name: string
  image_path: string
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface SegmentResult {
  segment_index: number
  ssim_score: number
  pixel_diff_score: number
  anomaly_detected: boolean
  confidence: number
  verdict: string
}

export interface AnalysisResult {
  segments: SegmentResult[]
  anomaly_count: number
  overall_defect_score: number
  average_confidence: number
  verdict: string
  defect_locations: number[]
}

export interface TestImage {
  id: string
  golden_image_id: string
  image_path: string
  upload_timestamp: string
  processed: boolean
  overall_defect_score: number
  detected_defects: AnalysisResult | null
  created_at: string
}

export interface ImageSegment {
  id: string
  test_image_id: string
  golden_image_id: string
  segment_index: number
  segment_image_path: string
  ssim_score: number
  pixel_diff_score: number
  anomaly_detected: boolean
  confidence: number
  processed_at: string | null
}

export interface DefectRecord {
  id: string
  test_image_id: string
  user_decision: 'accept' | 'reject' | 'review_later'
  swipe_direction: 'left' | 'right' | 'up' | null
  confidence_override: number | null
  notes: string | null
  review_timestamp: string
  created_at: string
}

export interface InspectionLog {
  id: string
  user_id: string | null
  action: string
  image_id: string | null
  details: Record<string, unknown>
  created_at: string
}
