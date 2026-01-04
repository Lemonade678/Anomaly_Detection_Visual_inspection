"""Edge detection for substrate extraction."""
import cv2
import numpy as np
import os
from typing import List, Optional


def run_edge_detection(
    image_or_path, 
    debug_output_dir: Optional[str] = None,
    min_width: int = 80,
    max_width: int = 240,
    min_height: int = 200,
    max_height: int = 800,
    canny_low: int = 40,
    canny_high: int = 125
) -> List[np.ndarray]:
    """Detect substrates using Canny edge detection.
    
    Args:
        image_or_path: Image array or path to image file
        debug_output_dir: Optional directory to save debug crops
        min_width, max_width: Width range for valid substrates
        min_height, max_height: Height range for valid substrates
        canny_low, canny_high: Canny edge detection thresholds
        
    Returns:
        List of cropped substrate images (sorted Top-Left -> Right)
    """
    # Load image
    if isinstance(image_or_path, str):
        if not os.path.exists(image_or_path):
            return []
        original_image = cv2.imread(image_or_path)
        if original_image is None:
            return []
        img_name = os.path.basename(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        original_image = image_or_path
        img_name = "memory_image.png"
    else:
        return []

    # Edge detection
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, canny_low, canny_high)
    
    # Create output directory if needed
    if debug_output_dir and not os.path.exists(debug_output_dir):
        os.makedirs(debug_output_dir)

    # Find contours
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []

    # Filter by size
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (min_width <= w <= max_width) and (min_height <= h <= max_height):
            valid_contours.append(contour)

    if not valid_contours:
        return []

    # Sort: Top-to-bottom, then Left-to-right
    def sort_key(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y // 50) * 10000 + x

    sorted_contours = sorted(valid_contours, key=sort_key)

    # Crop substrates
    cropped_substrates = []
    
    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        crop = original_image[y:y+h, x:x+w]
        cropped_substrates.append(crop)
        
        if debug_output_dir:
            output_filename = os.path.join(debug_output_dir, f'crop_{i}_{img_name}')
            cv2.imwrite(output_filename, crop)

    return cropped_substrates
