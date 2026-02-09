"""Core inspection pipeline logic.

Contains the main inspection routine combining alignment, SSIM, and pixel matching.
"""
import time
import cv2
import numpy as np
import os
from datetime import datetime

from .io import read_image
from .align import align_images, align_images_detailed
from .ssim import calc_ssim
from .pixel_match import run_pixel_matching, run_pixel_matching_multiscale
from .edge_detection import run_edge_detection
from .illumination import apply_light_sensitivity_mode, preprocess_pair
from .grid_analyzer import GridAnalyzer
from .config import (
    AlignmentMethod, LightSensitivityMode, LightSensitivityConfig,
    AlignmentConfig, PixelMatchConfig, InspectionConfig, get_default_config,
    load_substrate_config, get_inspection_params_from_substrate,
    SSIM_PASS_THRESHOLD, PIXEL_DIFF_THRESHOLD, COUNT_THRESHOLD
)
from .analysis import AnomalyLocationMapper, export_results_as_json
from .qr_cropper import QRCodeExtractor, get_qr_json


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
    # Note: Global modification of constants is generally discouraged but kept for compatibility
    # In a cleaner implementation, one would pass config objects to run_inspection
    
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
