"""Enhanced pixel-based anomaly detection.

Combines:
- V1: valid_area_mask, adaptive thresholding, multi-scale, confidence score, CLAHE
- V2: illumination normalization preprocessing
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple

from .illumination import preprocess_pair, equalize_histogram_gray


def run_pixel_matching(
    golden_image: np.ndarray, 
    aligned_image: np.ndarray, 
    pixel_thresh: int, 
    count_thresh: int,
    area_thresh: float = 20.0, 
    min_contour_area: int = 50, 
    mask: np.ndarray = None,
    valid_area_mask: np.ndarray = None,
    use_adaptive_threshold: bool = False,
    use_histogram_equalization: bool = True,
    normalize_lighting: bool = True,
    normalize_method: str = "match_histogram",
    kernel_size: Tuple[int, int] = (5, 5)
) -> dict:
    """Run pixel-based anomaly detection and return results dict.
    
    Enhanced to count TOTAL ANOMALOUS PIXELS and provide confidence score.

    Args:
        golden_image: Reference/golden image (BGR)
        aligned_image: Aligned test image (BGR)
        pixel_thresh: Threshold for pixel difference detection
        count_thresh: Maximum allowed anomalous pixel count
        area_thresh: Maximum allowed anomaly area percentage
        min_contour_area: Minimum contour area to consider as anomaly
        mask: Optional ROI mask (255=include, 0=exclude)
        valid_area_mask: Valid area mask from alignment (avoids border artifacts)
        use_adaptive_threshold: Use adaptive thresholding for uneven lighting
        use_histogram_equalization: Apply CLAHE before detection
        normalize_lighting: Apply illumination normalization (V2)
        normalize_method: Method for normalization (V2)
        kernel_size: Morphological kernel size

    Returns dict keys: 
        - area_score: Percentage of anomalous pixels
        - anomaly_count: Number of detected contours
        - anomalous_pixel_count: TOTAL number of anomalous pixels
        - count_thresh: Threshold used
        - verdict: "Anomaly" or "Normal"
        - heatmap: Visualization heatmap
        - contour_map: Image with bounding boxes
        - anomaly_mask: Binary mask of all anomalous pixels
        - confidence: Match quality score (0.0-1.0)
        - valid_pixel_count: Number of valid pixels analyzed
        - preprocessing_applied: Whether normalization was applied
    """
    # Apply illumination normalization (V2 feature)
    if normalize_lighting:
        golden_processed, aligned_processed = preprocess_pair(
            golden_image, aligned_image, method=normalize_method
        )
    else:
        golden_processed, aligned_processed = golden_image, aligned_image
    
    # Convert to grayscale
    golden_gray = cv2.cvtColor(golden_processed, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned_processed, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE histogram equalization (V1 feature)
    if use_histogram_equalization:
        golden_gray = equalize_histogram_gray(golden_gray)
        aligned_gray = equalize_histogram_gray(aligned_gray)
    
    # Compute absolute difference
    diff = cv2.absdiff(golden_gray, aligned_gray)
    diff = cv2.convertScaleAbs(diff)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)


    # Combine masks: ROI mask + valid area mask from alignment
    combined_mask = None
    if mask is not None:
        combined_mask = mask.copy()
    
    if valid_area_mask is not None:
        if combined_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, valid_area_mask)
        else:
            combined_mask = valid_area_mask
    
    # Apply combined mask
    if combined_mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    # Thresholding
    if use_adaptive_threshold:
        thresh = cv2.adaptiveThreshold(
            diff, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            blockSize=11,
            C=-5
        )
    else:
        thresh_val = int(pixel_thresh)
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closing, kernel, iterations=2)
    
    # Apply mask again
    if combined_mask is not None:
        dilated = cv2.bitwise_and(dilated, dilated, mask=combined_mask)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count anomalous pixels
    anomalous_pixel_count = np.count_nonzero(dilated)
    
    # Calculate valid pixel count
    if combined_mask is not None:
        valid_pixel_count = np.count_nonzero(combined_mask)
    else:
        valid_pixel_count = dilated.size

    # Draw contour bounding boxes
    contour_image = aligned_image.copy()
    anomaly_count = 0
    for c in contours:
        if cv2.contourArea(c) < min_contour_area:
            continue
        anomaly_count += 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Calculate area score
    area_score = (anomalous_pixel_count / max(valid_pixel_count, 1)) * 100.0
    
    # Verdict
    if area_score > area_thresh or anomalous_pixel_count > int(count_thresh):
        verdict = "Anomaly"
    else:
        verdict = "Normal"

    # Calculate confidence score
    anomaly_ratio = anomalous_pixel_count / max(valid_pixel_count, 1)
    confidence = max(0.0, 1.0 - (anomaly_ratio * 10))
    confidence = min(1.0, confidence)

    # Generate heatmap
    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)

    return {
        'area_score': float(area_score),
        'anomaly_count': int(anomaly_count),
        'anomalous_pixel_count': int(anomalous_pixel_count),
        'count_thresh': int(count_thresh),
        'verdict': verdict,
        'heatmap': heatmap,
        'contour_map': contour_image,
        'anomaly_mask': dilated,
        'confidence': float(confidence),
        'valid_pixel_count': int(valid_pixel_count),
        'preprocessing_applied': normalize_lighting,
        'preprocessing_method': normalize_method if normalize_lighting else None
    }


def run_pixel_matching_multiscale(
    golden_image: np.ndarray, 
    aligned_image: np.ndarray, 
    pixel_thresh: int, 
    count_thresh: int,
    area_thresh: float = 20.0, 
    min_contour_area: int = 50,
    mask: np.ndarray = None,
    valid_area_mask: np.ndarray = None,
    scales: List[Tuple[int, int]] = [(3, 3), (5, 5), (9, 9)],
    use_histogram_equalization: bool = True,
    normalize_lighting: bool = True,
    normalize_method: str = "match_histogram"
) -> dict:
    """Run multi-scale detection for different defect sizes (V1 feature).
    
    Detects both small defects and larger anomalies by running detection
    at multiple kernel scales and combining results.
    
    Args:
        golden_image: Reference/golden image (BGR)
        aligned_image: Aligned test image (BGR)
        pixel_thresh: Threshold for pixel difference detection
        count_thresh: Maximum allowed anomalous pixel count
        area_thresh: Maximum allowed anomaly area percentage
        min_contour_area: Minimum contour area to consider
        mask: Optional ROI mask
        valid_area_mask: Valid area mask from alignment
        scales: List of kernel sizes to use
        use_histogram_equalization: Apply CLAHE before detection
        normalize_lighting: Apply illumination normalization
        normalize_method: Normalization method
        
    Returns:
        Same dict structure as run_pixel_matching()
    """
    # Preprocessing
    if normalize_lighting:
        golden_processed, aligned_processed = preprocess_pair(
            golden_image, aligned_image, method=normalize_method
        )
    else:
        golden_processed, aligned_processed = golden_image, aligned_image
    
    golden_gray = cv2.cvtColor(golden_processed, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned_processed, cv2.COLOR_BGR2GRAY)
    
    if use_histogram_equalization:
        golden_gray = equalize_histogram_gray(golden_gray)
        aligned_gray = equalize_histogram_gray(aligned_gray)
    
    diff = cv2.absdiff(golden_gray, aligned_gray)
    
    # Combine masks
    combined_mask = None
    if mask is not None:
        combined_mask = mask.copy()
    if valid_area_mask is not None:
        if combined_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, valid_area_mask)
        else:
            combined_mask = valid_area_mask
    
    if combined_mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=combined_mask)
    
    # Multi-scale detection
    combined_anomaly_mask = None
    
    for kernel_size in scales:
        kernel = np.ones(kernel_size, np.uint8)
        
        _, thresh = cv2.threshold(diff, pixel_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        if combined_anomaly_mask is None:
            combined_anomaly_mask = closing
        else:
            combined_anomaly_mask = cv2.bitwise_or(combined_anomaly_mask, closing)
    
    # Final dilation
    final_kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(combined_anomaly_mask, final_kernel, iterations=1)
    
    if combined_mask is not None:
        dilated = cv2.bitwise_and(dilated, dilated, mask=combined_mask)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count
    anomalous_pixel_count = np.count_nonzero(dilated)
    valid_pixel_count = np.count_nonzero(combined_mask) if combined_mask is not None else dilated.size
    
    # Draw contours
    contour_image = aligned_image.copy()
    anomaly_count = 0
    for c in contours:
        if cv2.contourArea(c) < min_contour_area:
            continue
        anomaly_count += 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Calculations
    area_score = (anomalous_pixel_count / max(valid_pixel_count, 1)) * 100.0
    
    if area_score > area_thresh or anomalous_pixel_count > int(count_thresh):
        verdict = "Anomaly"
    else:
        verdict = "Normal"
    
    anomaly_ratio = anomalous_pixel_count / max(valid_pixel_count, 1)
    confidence = max(0.0, min(1.0, 1.0 - (anomaly_ratio * 10)))
    
    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)
    
    return {
        'area_score': float(area_score),
        'anomaly_count': int(anomaly_count),
        'anomalous_pixel_count': int(anomalous_pixel_count),
        'count_thresh': int(count_thresh),
        'verdict': verdict,
        'heatmap': heatmap,
        'contour_map': contour_image,
        'anomaly_mask': dilated,
        'confidence': float(confidence),
        'valid_pixel_count': int(valid_pixel_count),
        'scales_used': scales,
        'preprocessing_applied': normalize_lighting
    }
