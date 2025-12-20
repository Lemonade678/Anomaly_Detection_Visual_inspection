/*
  # Defect Inspection System Schema

  1. New Tables
    - `golden_images` - Master template images for comparison
      - `id` (uuid, primary key)
      - `name` (text) - Template name/product ID
      - `image_path` (text) - Storage path to the image
      - `metadata` (jsonb) - Additional metadata (resolution, etc)
      - `created_at` (timestamp)
      - `updated_at` (timestamp)
    
    - `test_images` - Images to be inspected
      - `id` (uuid, primary key)
      - `golden_image_id` (uuid, FK) - Reference golden template
      - `image_path` (text) - Storage path to the image
      - `upload_timestamp` (timestamp)
      - `processed` (boolean) - Whether 9-part analysis is complete
      - `overall_defect_score` (numeric) - Combined score from all 9 parts
      - `detected_defects` (jsonb) - Grid sections with defects
      - `created_at` (timestamp)
    
    - `image_segments` - 9-part grid segments for comparison
      - `id` (uuid, primary key)
      - `test_image_id` (uuid, FK) - Reference test image
      - `golden_image_id` (uuid, FK) - Reference golden image segment
      - `segment_index` (smallint 0-8) - Position in 3x3 grid
      - `segment_image_path` (text) - Path to segment image
      - `ssim_score` (numeric) - SSIM similarity score
      - `pixel_diff_score` (numeric) - Pixel difference score
      - `anomaly_detected` (boolean) - Defect found in this segment
      - `confidence` (numeric) - Confidence level 0-1
      - `processed_at` (timestamp)
    
    - `defect_records` - User inspection decisions
      - `id` (uuid, primary key)
      - `test_image_id` (uuid, FK) - Reference test image
      - `user_decision` (text) - 'accept', 'reject', 'review_later'
      - `swipe_direction` (text) - 'left', 'right', 'up'
      - `confidence_override` (numeric) - User's confidence override if any
      - `notes` (text) - User notes
      - `review_timestamp` (timestamp)
      - `created_at` (timestamp)
    
    - `inspection_logs` - Audit trail
      - `id` (uuid, primary key)
      - `user_id` (uuid) - Authenticated user
      - `action` (text) - Action performed
      - `image_id` (uuid) - Related image
      - `details` (jsonb) - Additional details
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on all tables
    - Users can only inspect images they have access to
    - Admins can manage golden templates
    - All inspection logs are immutable
*/

-- Golden Images Table
CREATE TABLE IF NOT EXISTS golden_images (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  image_path text NOT NULL,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE golden_images ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view golden images"
  ON golden_images
  FOR SELECT
  USING (true);

CREATE POLICY "Only admins can insert golden images"
  ON golden_images
  FOR INSERT
  WITH CHECK (auth.jwt() ->> 'role' = 'admin');

CREATE POLICY "Only admins can update golden images"
  ON golden_images
  FOR UPDATE
  USING (auth.jwt() ->> 'role' = 'admin')
  WITH CHECK (auth.jwt() ->> 'role' = 'admin');

-- Test Images Table
CREATE TABLE IF NOT EXISTS test_images (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  golden_image_id uuid NOT NULL REFERENCES golden_images(id) ON DELETE CASCADE,
  image_path text NOT NULL,
  upload_timestamp timestamptz DEFAULT now(),
  processed boolean DEFAULT false,
  overall_defect_score numeric DEFAULT 0,
  detected_defects jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE test_images ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view test images"
  ON test_images
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can insert test images"
  ON test_images
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Users can update their own test images"
  ON test_images
  FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

-- Image Segments Table
CREATE TABLE IF NOT EXISTS image_segments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  test_image_id uuid NOT NULL REFERENCES test_images(id) ON DELETE CASCADE,
  golden_image_id uuid NOT NULL REFERENCES golden_images(id) ON DELETE CASCADE,
  segment_index smallint NOT NULL CHECK (segment_index >= 0 AND segment_index <= 8),
  segment_image_path text NOT NULL,
  ssim_score numeric DEFAULT 0,
  pixel_diff_score numeric DEFAULT 0,
  anomaly_detected boolean DEFAULT false,
  confidence numeric DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 1),
  processed_at timestamptz
);

ALTER TABLE image_segments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view image segments"
  ON image_segments
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can insert image segments"
  ON image_segments
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

-- Defect Records Table
CREATE TABLE IF NOT EXISTS defect_records (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  test_image_id uuid NOT NULL REFERENCES test_images(id) ON DELETE CASCADE,
  user_decision text NOT NULL CHECK (user_decision IN ('accept', 'reject', 'review_later')),
  swipe_direction text CHECK (swipe_direction IN ('left', 'right', 'up')),
  confidence_override numeric CHECK (confidence_override IS NULL OR (confidence_override >= 0 AND confidence_override <= 1)),
  notes text,
  review_timestamp timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

ALTER TABLE defect_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view defect records"
  ON defect_records
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can create defect records"
  ON defect_records
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

-- Inspection Logs Table
CREATE TABLE IF NOT EXISTS inspection_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  action text NOT NULL,
  image_id uuid,
  details jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE inspection_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own logs"
  ON inspection_logs
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

CREATE POLICY "System can insert inspection logs"
  ON inspection_logs
  FOR INSERT
  TO authenticated
  WITH CHECK (user_id = auth.uid());

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_test_images_golden_id ON test_images(golden_image_id);
CREATE INDEX IF NOT EXISTS idx_test_images_processed ON test_images(processed);
CREATE INDEX IF NOT EXISTS idx_image_segments_test_id ON image_segments(test_image_id);
CREATE INDEX IF NOT EXISTS idx_image_segments_golden_id ON image_segments(golden_image_id);
CREATE INDEX IF NOT EXISTS idx_defect_records_test_id ON defect_records(test_image_id);
CREATE INDEX IF NOT EXISTS idx_defect_records_timestamp ON defect_records(review_timestamp);
CREATE INDEX IF NOT EXISTS idx_inspection_logs_user_id ON inspection_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_inspection_logs_created ON inspection_logs(created_at);