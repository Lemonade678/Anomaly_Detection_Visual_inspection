"""Integrated Hybrid Inspector - MAIN ENTRY POINT for PCB Inspection.

This is the PRIMARY processing module for the Anomaly Detection Visual Inspection system.
Use this file as the main entry point instead of modular_inspection_integrated/gui.py.

Combines the best features from:
- hybrid_1.py / Modular_inspection_1 (CLAHE, multi-scale, valid_area_mask, grid analyzer)
- hybrid_2.py / modular_inspection2 (light sensitivity, multi-alignment, illumination normalization)

Usage (Command Line):
    python hybrid_integrated.py              # Launch GUI (default)
    python hybrid_integrated.py --demo       # Run pipeline demo
    python hybrid_integrated.py --grid-demo  # Run grid analysis demo
    python hybrid_integrated.py --help       # Show help

Programmatic Usage:
    # For inspection processing:
    from hybrid_integrated import run_inspection, run_inspection_with_config
    result = run_inspection(golden_image, test_image)
    
    # For GUI:
    from hybrid_integrated import run_gui
    run_gui()
"""
import os
import sys
import time
import csv
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# Import from unified integrated package
from modular_inspection_integrated.io import read_image
from modular_inspection_integrated.align import align_images, align_images_detailed
from modular_inspection_integrated.ssim import calc_ssim
from modular_inspection_integrated.pixel_match import run_pixel_matching, run_pixel_matching_multiscale
from modular_inspection_integrated.edge_detection import run_edge_detection
from modular_inspection_integrated.illumination import apply_light_sensitivity_mode, preprocess_pair
from modular_inspection_integrated.grid_analyzer import GridAnalyzer
from modular_inspection_integrated.config import (
    AlignmentMethod, LightSensitivityMode, LightSensitivityConfig,
    AlignmentConfig, PixelMatchConfig, InspectionConfig, get_default_config,
    load_substrate_config, get_inspection_params_from_substrate
)
from modular_inspection_integrated.qr_cropper import QRCodeExtractor, get_qr_json


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SSIM_PASS_THRESHOLD = 0.975
PIXEL_DIFF_THRESHOLD = 40
COUNT_THRESHOLD = 5000


# ==============================================================================
# ANOMALY LOCATION MAPPER (from V2)
# ==============================================================================

class AnomalyLocationMapper:
    """Maps and reports anomaly locations with pixel coordinates."""
    
    def __init__(self):
        self.anomaly_regions = []
        self.centroids = []
    
    def analyze_mask(self, anomaly_mask: np.ndarray, min_area: int = 50) -> dict:
        """Analyze anomaly mask to extract location information."""
        self.anomaly_regions = []
        self.centroids = []
        
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = anomaly_mask.shape
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw // 2, y + bh // 2
            
            sector_x = "Left" if cx < w/3 else ("Right" if cx > 2*w/3 else "Center")
            sector_y = "Top" if cy < h/3 else ("Bottom" if cy > 2*h/3 else "Middle")
            sector = f"{sector_y}-{sector_x}"
            
            region_info = {
                'id': len(self.anomaly_regions) + 1,
                'bbox': (x, y, bw, bh),
                'centroid': (cx, cy),
                'area_px': int(area),
                'sector': sector,
                'relative_pos': (round(cx/w * 100, 1), round(cy/h * 100, 1))
            }
            
            self.anomaly_regions.append(region_info)
            self.centroids.append((cx, cy))
        
        total_anomalous_pixels = np.count_nonzero(anomaly_mask)
        
        return {
            'regions': self.anomaly_regions,
            'total_regions': len(self.anomaly_regions),
            'total_anomalous_pixels': total_anomalous_pixels,
            'coverage_percent': round(total_anomalous_pixels / (h * w) * 100, 3),
            'image_size': (w, h)
        }
    
    def create_annotated_image(self, base_image: np.ndarray, 
                                show_centroids: bool = True,
                                show_labels: bool = True,
                                show_grid: bool = True) -> np.ndarray:
        """Create an annotated image with anomaly locations."""
        annotated = base_image.copy()
        h, w = annotated.shape[:2]
        
        if show_grid:
            grid_color = (40, 40, 40)
            cv2.line(annotated, (w//3, 0), (w//3, h), grid_color, 1)
            cv2.line(annotated, (2*w//3, 0), (2*w//3, h), grid_color, 1)
            cv2.line(annotated, (0, h//3), (w, h//3), grid_color, 1)
            cv2.line(annotated, (0, 2*h//3), (w, 2*h//3), grid_color, 1)
        
        for region in self.anomaly_regions:
            x, y, bw, bh = region['bbox']
            cx, cy = region['centroid']
            region_id = region['id']
            
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
            
            if show_centroids:
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(annotated, (cx, cy), 7, (0, 0, 255), 2)
            
            if show_labels:
                label = f"#{region_id} ({cx},{cy})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                label_y = max(y - 5, label_size[1] + 5)
                cv2.rectangle(annotated, (x, label_y - label_size[1] - 2), 
                             (x + label_size[0] + 4, label_y + 2), (0, 0, 0), -1)
                cv2.putText(annotated, label, (x + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated
    
    def get_summary_text(self) -> str:
        """Get a formatted text summary of anomaly locations."""
        if not self.anomaly_regions:
            return "No anomalies detected."
        
        lines = [f"Found {len(self.anomaly_regions)} anomaly region(s):"]
        for region in self.anomaly_regions:
            rel_x, rel_y = region['relative_pos']
            lines.append(
                f"  #{region['id']}: {region['sector']} "
                f"@ ({region['centroid'][0]}, {region['centroid'][1]}) "
                f"[{rel_x}%, {rel_y}%] - {region['area_px']}px"
            )
        return "\n".join(lines)


# ==============================================================================
# INTEGRATED INSPECTION PIPELINE
# ==============================================================================

def run_inspection(
    golden_image: np.ndarray,
    test_image: np.ndarray,
    config: InspectionConfig = None,
    alignment_method: AlignmentMethod = AlignmentMethod.AUTO,
    light_mode: LightSensitivityMode = LightSensitivityMode.AUTO,
    use_multi_scale: bool = False,
    normalize_lighting: bool = True,
    normalize_method: str = "match_histogram",
    verbose: bool = True
) -> dict:
    """Run the full integrated inspection pipeline.
    
    Combines:
    - Light sensitivity preprocessing
    - Multi-method alignment with valid_area_mask
    - SSIM pre-check
    - Enhanced pixel matching with confidence scores
    - Anomaly location mapping
    
    Args:
        golden_image: Master/golden reference image (BGR)
        test_image: Test image to inspect (BGR)
        config: Inspection configuration (uses defaults if None)
        alignment_method: Method for image alignment
        light_mode: Light sensitivity mode
        use_multi_scale: Use multi-scale detection
        normalize_lighting: Apply illumination normalization
        normalize_method: Normalization method
        verbose: Print progress messages
        
    Returns:
        Dictionary with complete inspection results
    """
    if config is None:
        config = get_default_config()
    
    start_time = time.time()
    anomaly_mapper = AnomalyLocationMapper()
    
    # -------------------------------------------------------------------------
    # STEP 1: Light Sensitivity Preprocessing
    # -------------------------------------------------------------------------
    if verbose:
        print("[1] Applying light sensitivity mode...")
    
    light_config = LightSensitivityConfig(mode=light_mode)
    golden_proc = apply_light_sensitivity_mode(golden_image, light_mode, light_config)
    test_proc = apply_light_sensitivity_mode(test_image, light_mode, light_config)
    
    # -------------------------------------------------------------------------
    # STEP 2: Resize to Match
    # -------------------------------------------------------------------------
    h, w = golden_proc.shape[:2]
    test_resized = cv2.resize(test_proc, (w, h))
    
    # -------------------------------------------------------------------------
    # STEP 3: Alignment
    # -------------------------------------------------------------------------
    if verbose:
        print(f"[2] Aligning images ({alignment_method.value})...")
    
    align_config = AlignmentConfig(method=alignment_method)
    aligned_image, (dx, dy), confidence, valid_mask = align_images(
        golden_proc, test_resized, method=alignment_method, config=align_config
    )
    
    if verbose:
        print(f"    Translation: dx={dx:.2f}, dy={dy:.2f}")
        print(f"    Confidence: {confidence:.4f}")
    
    if confidence < 0.1:
        return {
            'verdict': 'Error',
            'error': 'Alignment failed (low confidence)',
            'alignment_confidence': confidence,
            'processing_time': time.time() - start_time
        }
    
    # -------------------------------------------------------------------------
    # STEP 4: SSIM Pre-Check
    # -------------------------------------------------------------------------
    if verbose:
        print("[3] Running SSIM structural check...")
    
    ssim_score, ssim_heatmap = calc_ssim(golden_proc, aligned_image)
    
    if verbose:
        print(f"    SSIM Score: {ssim_score:.4f} (threshold: {SSIM_PASS_THRESHOLD})")
    
    if ssim_score > SSIM_PASS_THRESHOLD:
        if verbose:
            print("    >> SSIM PASS - Skipping pixel analysis")
        
        return {
            'verdict': 'Normal',
            'method': 'SSIM',
            'ssim_score': ssim_score,
            'alignment_confidence': confidence,
            'aligned_image': aligned_image,
            'ssim_heatmap': ssim_heatmap,
            'processing_time': time.time() - start_time
        }
    
    if verbose:
        print("    >> SSIM failed, proceeding to pixel analysis...")
    
    # -------------------------------------------------------------------------
    # STEP 5: Pixel Matching
    # -------------------------------------------------------------------------
    if verbose:
        print("[4] Running pixel matching analysis...")
    
    if use_multi_scale:
        pixel_result = run_pixel_matching_multiscale(
            golden_proc, aligned_image,
            pixel_thresh=PIXEL_DIFF_THRESHOLD,
            count_thresh=COUNT_THRESHOLD,
            valid_area_mask=valid_mask,
            normalize_lighting=normalize_lighting,
            normalize_method=normalize_method
        )
    else:
        pixel_result = run_pixel_matching(
            golden_proc, aligned_image,
            pixel_thresh=PIXEL_DIFF_THRESHOLD,
            count_thresh=COUNT_THRESHOLD,
            valid_area_mask=valid_mask,
            normalize_lighting=normalize_lighting,
            normalize_method=normalize_method
        )
    
    if verbose:
        print(f"    Area Score: {pixel_result['area_score']:.2f}%")
        print(f"    Anomaly Count: {pixel_result['anomaly_count']}")
        print(f"    Confidence: {pixel_result['confidence']:.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 6: Anomaly Location Mapping
    # -------------------------------------------------------------------------
    if pixel_result['verdict'] == 'Anomaly':
        if verbose:
            print("[5] Mapping anomaly locations...")
        
        location_data = anomaly_mapper.analyze_mask(pixel_result['anomaly_mask'])
        pixel_result['location_data'] = location_data
        pixel_result['location_summary'] = anomaly_mapper.get_summary_text()
        pixel_result['contour_map'] = anomaly_mapper.create_annotated_image(
            pixel_result['contour_map']
        )
        
        if verbose:
            print(f"    {anomaly_mapper.get_summary_text()}")
    
    # -------------------------------------------------------------------------
    # Final Result
    # -------------------------------------------------------------------------
    processing_time = time.time() - start_time
    
    result = {
        'verdict': pixel_result['verdict'],
        'method': 'Pixel Matching',
        'ssim_score': ssim_score,
        'area_score': pixel_result['area_score'],
        'anomaly_count': pixel_result['anomaly_count'],
        'anomalous_pixel_count': pixel_result['anomalous_pixel_count'],
        'confidence': pixel_result['confidence'],
        'alignment_confidence': confidence,
        'aligned_image': aligned_image,
        'heatmap': pixel_result['heatmap'],
        'contour_map': pixel_result['contour_map'],
        'anomaly_mask': pixel_result['anomaly_mask'],
        'ssim_heatmap': ssim_heatmap,
        'valid_mask': valid_mask,
        'processing_time': processing_time
    }
    
    if 'location_data' in pixel_result:
        result['location_data'] = pixel_result['location_data']
        result['location_summary'] = pixel_result['location_summary']
    
    if verbose:
        print("\n" + "=" * 60)
        if result['verdict'] == 'Anomaly':
            print("FINAL VERDICT: !! ANOMALY DETECTED !!")
        else:
            print("FINAL VERDICT: NORMAL")
        print(f"Processing time: {processing_time:.2f}s")
        print("=" * 60)
    
    return result


def export_results_as_json(result: dict, output_path: str = None) -> str:
    """Export inspection results to JSON file.
    
    Args:
        result: Result dict from run_inspection()
        output_path: Optional output file path (auto-generates if None)
        
    Returns:
        Path to saved JSON file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inspection_result_{timestamp}.json"
    
    # Convert numpy arrays and non-serializable types
    json_result = {
        'timestamp': datetime.now().isoformat(),
        'verdict': result.get('verdict', 'Unknown'),
        'method': result.get('method', 'Unknown'),
        'ssim_score': float(result.get('ssim_score', 0)),
        'area_score': float(result.get('area_score', 0)),
        'anomaly_count': int(result.get('anomaly_count', 0)),
        'anomalous_pixel_count': int(result.get('anomalous_pixel_count', 0)),
        'confidence': float(result.get('confidence', 0)),
        'alignment_confidence': float(result.get('alignment_confidence', 0)),
        'processing_time': float(result.get('processing_time', 0))
    }
    
    if 'location_data' in result:
        json_result['location_data'] = result['location_data']
    
    if 'location_summary' in result:
        json_result['location_summary'] = result['location_summary']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    return output_path


def run_inspection_with_config(
    golden_path: str,
    test_path: str,
    config_json_path: str = None,
    output_json: bool = True
) -> dict:
    """Run inspection using substrate config from JSON file.
    
    Args:
        golden_path: Path to golden/master image
        test_path: Path to test image
        config_json_path: Path to substrate config JSON (e.g., Configtype1.json)
        output_json: Whether to save results as JSON
        
    Returns:
        Inspection result dict
    """
    # Load images
    golden_image = read_image(golden_path)
    test_image = read_image(test_path)
    
    if golden_image is None:
        return {'verdict': 'Error', 'error': f'Failed to load golden image: {golden_path}'}
    if test_image is None:
        return {'verdict': 'Error', 'error': f'Failed to load test image: {test_path}'}
    
    # Load substrate config if provided
    if config_json_path:
        substrate_config = load_substrate_config(config_json_path)
        inspection_params = get_inspection_params_from_substrate(substrate_config)
        
        # Update thresholds from config
        global PIXEL_DIFF_THRESHOLD
        PIXEL_DIFF_THRESHOLD = inspection_params.get('pixel_threshold', PIXEL_DIFF_THRESHOLD)
    
    # Run inspection
    result = run_inspection(
        golden_image, test_image,
        alignment_method=AlignmentMethod.AUTO,
        light_mode=LightSensitivityMode.AUTO,
        verbose=True
    )
    
    # Export to JSON
    if output_json:
        json_path = export_results_as_json(result)
        result['json_output_path'] = json_path
        print(f"Results saved to: {json_path}")
    
    return result


# ==============================================================================
# DEMO FUNCTION
# ==============================================================================

def run_inspection_demo():
    """Demonstrates the integrated inspection pipeline."""
    
    print("=" * 60)
    print("INTEGRATED ANOMALY DETECTION PIPELINE DEMO")
    print("Combines Modular_inspection_1 + modular_inspection2")
    print("=" * 60)
    
    # Create synthetic test images
    print("\nCreating synthetic test images...")
    golden_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    # Add recognizable features
    cv2.circle(golden_image, (100, 100), 30, (255, 0, 0), -1)
    cv2.rectangle(golden_image, (200, 200), (300, 300), (0, 255, 0), -1)
    cv2.circle(golden_image, (400, 400), 50, (0, 0, 255), -1)
    
    # Test image with defect
    test_image = golden_image.copy()
    cv2.circle(test_image, (350, 150), 15, (255, 255, 255), -1)  # Simulated defect
    
    print(f"Golden image shape: {golden_image.shape}")
    print(f"Test image shape:   {test_image.shape}")
    
    # Run inspection
    result = run_inspection(
        golden_image, 
        test_image,
        alignment_method=AlignmentMethod.AUTO,
        light_mode=LightSensitivityMode.AUTO,
        use_multi_scale=False,
        verbose=True
    )
    
    print(f"\nReturned result: {result['verdict']}")
    
    return result


def run_grid_analysis_demo():
    """Demo of 9-part grid analysis."""
    
    print("\n" + "=" * 60)
    print("GRID ANALYSIS DEMO (9-Part)")
    print("=" * 60)
    
    # Create test images
    golden_image = np.random.randint(100, 150, (600, 600, 3), dtype=np.uint8)
    test_image = golden_image.copy()
    
    # Add defects to specific grid sectors
    cv2.rectangle(test_image, (10, 10), (150, 150), (255, 255, 255), -1)  # Sector 0
    cv2.circle(test_image, (500, 500), 50, (0, 0, 0), -1)  # Sector 8
    
    analyzer = GridAnalyzer()
    results = analyzer.analyze_images(golden_image, test_image)
    
    print(f"Overall verdict: {results['verdict']}")
    print(f"Anomaly count: {results['anomaly_count']} / 9 segments")
    print(f"Defect locations: {results['defect_locations']}")
    
    return results


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_gui():
    """Launch the integrated GUI application."""
    from modular_inspection_integrated.gui import InspectorApp
    print("Launching Integrated Inspector GUI...")
    app = InspectorApp()
    app.mainloop()


if __name__ == "__main__":
    # This is the MAIN entry point for the PCB Inspection application
    # Run: python hybrid_integrated.py [options]
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--demo":
            run_inspection_demo()
            run_grid_analysis_demo()
        elif arg == "--grid-demo":
            run_grid_analysis_demo()
        elif arg == "--help" or arg == "-h":
            print("=" * 60)
            print("INTEGRATED PCB INSPECTION - hybrid_integrated.py")
            print("=" * 60)
            print("")
            print("Usage:")
            print("  python hybrid_integrated.py              Launch GUI (default)")
            print("  python hybrid_integrated.py --demo       Run pipeline demo")
            print("  python hybrid_integrated.py --grid-demo  Run grid analysis demo")
            print("  python hybrid_integrated.py --help       Show this help message")
            print("")
            print("For programmatic use:")
            print("  from hybrid_integrated import run_inspection, run_inspection_with_config")
            print("  result = run_inspection(golden_image, test_image)")
            print("")
            print("  OR for GUI:")
            print("  from hybrid_integrated import run_gui")
            print("  run_gui()")
            print("")
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
    else:
        # Default: Launch GUI application
        run_gui()
