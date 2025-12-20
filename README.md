# Defect Inspector - Tinder-Style Defect Detection System

A modern, web-based visual inspection system that uses a Tinder-like swipe interface for rapid defect detection and quality assurance. The system divides images into a 3x3 grid and analyzes each section independently against a master template using advanced image processing algorithms.

## Features

- **Tinder-Style Interface**: Intuitive swipe-based defect inspection workflow
- **9-Part Grid Analysis**: Divides test images into 3x3 grid sections for detailed comparison
- **Dual-Stage Detection**: SSIM pre-check followed by pixel-level matching analysis
- **Real-Time Feedback**: Instant visual feedback on defect locations and confidence scores
- **Dashboard & Analytics**: Comprehensive statistics on inspection history and quality metrics
- **Template Management**: Admin panel for uploading and managing golden master images
- **Authentication**: Secure user authentication with Supabase
- **Dark Modern UI**: Professional dark theme with intuitive navigation

## Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with Vite build tool
- **Language**: TypeScript for type safety
- **Styling**: CSS with CSS Grid and Flexbox for responsive layouts
- **State Management**: React hooks with Supabase client integration

### Backend
- **Database**: Supabase PostgreSQL with Row Level Security (RLS)
- **Edge Functions**: Deno-based serverless functions for image processing
- **Storage**: Supabase object storage for image management

### Image Processing (Python)
- **GridAnalyzer**: Custom module for 9-part image grid analysis
- **SSIM Detection**: Structural similarity index measurement for quick screening
- **Pixel Matching**: Detailed pixel-level difference detection using OpenCV
- **Image Alignment**: ORB feature matching for rotation/scale handling

## Project Structure

```
project/
├── src/
│   ├── components/
│   │   ├── InspectionCard.tsx    # Main swipe interface
│   │   ├── Dashboard.tsx         # Analytics dashboard
│   │   └── AdminPanel.tsx        # Template management
│   ├── lib/
│   │   └── supabase.ts          # Supabase client setup
│   ├── types/
│   │   └── index.ts             # TypeScript type definitions
│   ├── App.tsx                  # Main application component
│   ├── App.css                  # App styling
│   ├── index.css                # Global styles
│   └── main.tsx                 # React entry point
├── Modular_inspection/           # Python image processing
│   ├── grid_analyzer.py         # 9-part grid analysis
│   ├── align_revamp.py          # Image alignment (ORB)
│   ├── ssim.py                  # SSIM calculation
│   ├── pixel_match.py           # Pixel matching
│   └── ...
├── supabase/functions/
│   └── process-image/           # Edge Function for processing
├── index.html                   # HTML entry point
├── package.json                 # JavaScript dependencies
├── tsconfig.json               # TypeScript configuration
└── vite.config.ts              # Vite configuration
```

## Database Schema

### Tables

1. **golden_images** - Master template images
   - Stores uploaded golden/reference images
   - Used as comparison baseline for test images

2. **test_images** - Images to be inspected
   - Contains uploaded test images
   - Stores overall defect scores and analysis results

3. **image_segments** - 3x3 grid segment data
   - Individual segment analysis results
   - SSIM and pixel difference scores
   - Anomaly detection per segment

4. **defect_records** - User inspection decisions
   - Swipe decisions (accept/reject/review_later)
   - User notes and confidence overrides
   - Audit trail of inspection actions

5. **inspection_logs** - System audit logs
   - Comprehensive action logging
   - User activity tracking
   - Immutable inspection history

## How It Works

### Inspection Pipeline

1. **Upload Phase**
   - User uploads test image via web interface
   - Image stored in Supabase storage bucket
   - Test image record created in database

2. **Image Division**
   - Test image divided into 3x3 grid (9 segments)
   - Each segment compared against corresponding golden image segment

3. **Analysis Stage 1: SSIM Check**
   - Quick structural similarity assessment
   - If SSIM > 0.975, image marked as NORMAL
   - Otherwise, proceed to stage 2

4. **Analysis Stage 2: Pixel Matching**
   - Detailed pixel-level difference detection
   - Count anomalous pixels exceeding threshold
   - Calculate defect score based on anomaly count

5. **Results Visualization**
   - Display side-by-side comparison with analysis results
   - Highlight affected grid sections
   - Show confidence scores and metrics

6. **User Decision**
   - User swipes to make decision:
     - Swipe Right → Accept (no defect)
     - Swipe Left → Reject (has defect)
     - Swipe Up → Review Later

### Grid Analysis

Images are divided into a 3x3 grid:
```
[1] [2] [3]
[4] [5] [6]
[7] [8] [9]
```

Each section is analyzed independently, allowing detection of localized defects while maintaining context of overall image quality.

## Getting Started

### Prerequisites
- Node.js 16+ (for frontend)
- Python 3.8+ (for image processing backend)
- Supabase account with PostgreSQL database

### Installation

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Configure environment variables**
   - Update `.env` with your Supabase credentials
   - Already configured with VITE_SUPABASE_URL and VITE_SUPABASE_SUPABASE_ANON_KEY

3. **Install Python dependencies** (for backend processing)
   ```bash
   pip install opencv-python numpy scikit-image
   ```

### Running Locally

**Development server:**
```bash
npm run dev
```
The app will be available at `http://localhost:5173`

**Production build:**
```bash
npm run build
```

## Usage Guide

### For Inspectors

1. **Login**: Sign in with your Supabase credentials
2. **Inspect**: Click "Inspect" to start reviewing images
3. **Swipe**: Use mouse drag or touch swipe to make decisions
4. **View History**: Click "History" to see past decisions
5. **Check Analytics**: Click "Dashboard" for quality metrics

### For Administrators

1. **Navigate to Templates**: Click "Templates" in navigation
2. **Upload Master Image**:
   - Enter a descriptive name (e.g., "Product Model A v1")
   - Select an image file
   - Click "Upload Template"
3. **Manage Templates**: View all uploaded templates and delete as needed

## Key Technologies

- **React 18**: UI framework
- **TypeScript 5**: Type-safe JavaScript
- **Vite 4**: Fast build tool
- **Supabase**: Backend-as-a-Service (Auth, Database, Storage)
- **OpenCV**: Computer vision image processing
- **scikit-image**: SSIM and image metrics
- **Deno**: Edge Functions runtime

## API Integration

### Process Image Edge Function
```typescript
POST /functions/v1/process-image
{
  goldenImageId: string,
  testImagePath: string
}
```

Returns:
```json
{
  "message": "Image processing queued",
  "goldenImageId": "...",
  "testImagePath": "..."
}
```

## Security Features

- **Row Level Security (RLS)**: Database access controlled at row level
- **Authentication**: Supabase JWT-based authentication
- **Image Storage**: Secure cloud storage with access controls
- **Audit Logging**: Complete audit trail of all decisions
- **Data Privacy**: User data isolated and protected

## Performance

- **Frontend Build**: 330KB (gzip: 94KB)
- **CSS**: 18KB (gzip: 3.8KB)
- **Build Time**: ~4 seconds
- **Image Processing**: Fast SSIM pre-check followed by detailed analysis

## Configuration

### Thresholds (configurable in GridAnalyzer)

- **SSIM Threshold**: 0.975 (images above this are considered normal)
- **Pixel Difference Threshold**: 40 (pixel value difference to consider anomalous)
- **Count Threshold**: 1000 (minimum anomalous pixels to flag defect)
- **Grid Size**: 3x3 (9 segments per image)

## Limitations & Future Improvements

### Current Limitations
- Image processing backend requires Python environment
- Grid analysis requires aligned images for best results
- Manual decision required (not fully automated)

### Planned Improvements
- Automated image processing with Python backend integration
- Machine learning model for defect classification
- Batch processing mode for high-volume inspection
- Mobile app for on-site inspection
- Real-time camera feed processing
- Advanced filtering and search capabilities
- Multi-user collaboration features
- Detailed defect reports and export functionality

## Troubleshooting

### Build Issues
```bash
npm run build
```
Verify TypeScript compiles successfully before running Vite.

### Environment Variables
Ensure `.env` contains valid Supabase credentials:
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_SUPABASE_ANON_KEY`

### Database Connection
Check Supabase project is active and database migrations have been applied.

## License

See LICENSE file for details.

## Contributing

To extend or modify this system:

1. Add new components in `src/components/`
2. Update types in `src/types/index.ts`
3. Create new Edge Functions in `supabase/functions/`
4. Extend Python processing in `Modular_inspection/`
5. Run tests and build validation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the codebase structure
3. Consult Supabase documentation
4. Check React and TypeScript documentation
