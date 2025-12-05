"""PCB strip extraction from panel images."""
import cv2
import numpy as np
from typing import List, Tuple, Optional


def extract_strips(panel_image: np.ndarray, 
                  expected_count: int = 6,
                  min_aspect_ratio: float = 1.5,
                  max_aspect_ratio: float = 8.0) -> List[np.ndarray]:
    """Extract individual PCB strips from a panel image.
    
    Uses edge detection and contour analysis to identify rectangular strips.
    Handles both single-row and multi-row layouts.
    
    Args:
        panel_image: Input panel image (BGR format)
        expected_count: Expected number of strips (default: 6)
        min_aspect_ratio: Minimum aspect ratio (w/h or h/w) for valid strip
        max_aspect_ratio: Maximum aspect ratio for valid strip
        
    Returns:
        List of extracted strip images (cropped and sorted)
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding works better for varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # Morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: try Canny edge detection
        return _extract_strips_canny(panel_image, expected_count, 
                                    min_aspect_ratio, max_aspect_ratio)
    
    # Filter and validate contours
    valid_strips = []
    panel_height, panel_width = panel_image.shape[:2]
    
    # Calculate average dimensions from all contours to filter outliers
    all_areas = [cv2.contourArea(c) for c in contours]
    if all_areas:
        median_area = np.median(all_areas)
        min_area = median_area * 0.3  # At least 30% of median
    else:
        min_area = (panel_width * panel_height) * 0.01  # At least 1% of panel
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio (always > 1)
        aspect_ratio = max(w/h, h/w) if h > 0 else 0
        
        # Validate strip dimensions
        if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
            w > 50 and h > 50):  # Minimum size check
            
            valid_strips.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })
    
    # Sort strips: top-to-bottom, then left-to-right
    valid_strips.sort(key=lambda s: (s['bbox'][1] // 100, s['bbox'][0]))
    
    # Extract the strip images
    extracted_strips = []
    for strip_info in valid_strips[:expected_count]:  # Limit to expected count
        x, y, w, h = strip_info['bbox']
        strip_img = panel_image[y:y+h, x:x+w]
        extracted_strips.append(strip_img)
    
    return extracted_strips


def _extract_strips_canny(panel_image: np.ndarray,
                         expected_count: int,
                         min_aspect_ratio: float,
                         max_aspect_ratio: float) -> List[np.ndarray]:
    """Fallback extraction using Canny edge detection.
    
    Used when adaptive thresholding doesn't find enough contours.
    """
    gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Filter by size and aspect ratio
    valid_strips = []
    panel_height, panel_width = panel_image.shape[:2]
    min_area = (panel_width * panel_height) * 0.01
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w/h, h/w) if h > 0 else 0
        
        if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
            w > 50 and h > 50):
            
            valid_strips.append({
                'bbox': (x, y, w, h),
                'area': area
            })
    
    # Sort by position
    valid_strips.sort(key=lambda s: (s['bbox'][1] // 100, s['bbox'][0]))
    
    # Extract images
    extracted_strips = []
    for strip_info in valid_strips[:expected_count]:
        x, y, w, h = strip_info['bbox']
        strip_img = panel_image[y:y+h, x:x+w]
        extracted_strips.append(strip_img)
    
    return extracted_strips


def extract_strips_manual_roi(panel_image: np.ndarray,
                              roi_list: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """Extract strips using manually defined ROIs.
    
    Useful when automatic detection fails or for consistent extraction patterns.
    
    Args:
        panel_image: Input panel image
        roi_list: List of (x, y, w, h) tuples defining each strip location
        
    Returns:
        List of extracted strip images
    """
    extracted_strips = []
    
    for x, y, w, h in roi_list:
        # Validate ROI is within image bounds
        if (x >= 0 and y >= 0 and 
            x + w <= panel_image.shape[1] and 
            y + h <= panel_image.shape[0]):
            
            strip_img = panel_image[y:y+h, x:x+w]
            extracted_strips.append(strip_img)
    
    return extracted_strips


def visualize_strip_extraction(panel_image: np.ndarray,
                               strips: List[np.ndarray],
                               save_path: Optional[str] = None) -> np.ndarray:
    """Create visualization of extracted strips for debugging.
    
    Args:
        panel_image: Original panel image
        strips: List of extracted strips
        save_path: Optional path to save visualization
        
    Returns:
        Visualization image with numbered bounding boxes
    """
    vis_image = panel_image.copy()
    
    # We don't have original bounding boxes, so we'll re-extract them
    gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and labels
    for idx, contour in enumerate(contours[:len(strips)]):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(vis_image, f"Strip {idx+1}", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image
