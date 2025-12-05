"""Image utility functions for display, cropping, and manipulation."""
import cv2
import numpy as np
import json
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


def show_image_with_zoom(image: np.ndarray, window_name: str = "Image Viewer",
                        max_width: int = 1200, max_height: int = 900) -> None:
    """Display image in a window with automatic sizing.
    
    Args:
        image: Image to display
        window_name: Window title
        max_width: Maximum window width
        max_height: Maximum window height
    """
    h, w = image.shape[:2]
    
    # Calculate scaling to fit within max dimensions
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = image
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)


def draw_roi_overlay(image: np.ndarray, roi: Tuple[int, int, int, int],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2, label: Optional[str] = None) -> np.ndarray:
    """Draw ROI rectangle overlay on image.
    
    Args:
        image: Input image
        roi: ROI as (x, y, w, h)
        color: BGR color for rectangle
        thickness: Line thickness
        label: Optional label text
        
    Returns:
        Image with ROI overlay
    """
    result = image.copy()
    x, y, w, h = roi
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label:
        # Add background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (x, y - text_size[1] - 10), 
                     (x + text_size[0] + 10, y), color, -1)
        cv2.putText(result, label, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def save_crop_pattern(roi: Tuple[int, int, int, int], 
                     filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save crop pattern to JSON file.
    
    Args:
        roi: ROI as (x, y, w, h)
        filepath: Output file path
        metadata: Optional metadata to include
    """
    pattern = {
        'roi': {'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3]},
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(pattern, f, indent=2)


def load_crop_pattern(filepath: str) -> Optional[Tuple[int, int, int, int]]:
    """Load crop pattern from JSON file.
    
    Args:
        filepath: Path to pattern file
        
    Returns:
        ROI as (x, y, w, h) or None if file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        return None
    
    with open(filepath, 'r') as f:
        pattern = json.load(f)
    
    roi_dict = pattern.get('roi', {})
    return (roi_dict.get('x'), roi_dict.get('y'), 
            roi_dict.get('w'), roi_dict.get('h'))


def apply_crop_pattern(image: np.ndarray, roi: Tuple[int, int, int, int],
                      validate: bool = True) -> Optional[np.ndarray]:
    """Apply crop pattern to image with validation.
    
    Args:
        image: Input image
        roi: ROI as (x, y, w, h)
        validate: Whether to validate ROI bounds
        
    Returns:
        Cropped image or None if ROI is invalid
    """
    h, w = image.shape[:2]
    x, y, crop_w, crop_h = roi
    
    if validate:
        # Check if ROI is within image bounds
        if x < 0 or y < 0 or x + crop_w > w or y + crop_h > h:
            return None
    else:
        # Clip ROI to image bounds
        x = max(0, x)
        y = max(0, y)
        crop_w = min(crop_w, w - x)
        crop_h = min(crop_h, h - y)
    
    return image[y:y+crop_h, x:x+crop_w]


def preview_comparison(image1: np.ndarray, image2: np.ndarray,
                      title1: str = "Image 1", title2: str = "Image 2",
                      window_name: str = "Comparison") -> None:
    """Show side-by-side comparison of two images.
    
    Args:
        image1: First image
        image2: Second image
        title1: Label for first image
        title2: Label for second image
        window_name: Window title
    """
    # Resize images to same size
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    target_h = max(h1, h2)
    target_w = max(w1, w2)
    
    # Resize both to same dimensions
    img1_resized = cv2.resize(image1, (target_w, target_h))
    img2_resized = cv2.resize(image2, (target_w, target_h))
    
    # Add labels
    cv2.putText(img1_resized, title1, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img2_resized, title2, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Concatenate horizontally
    comparison = np.hstack([img1_resized, img2_resized])
    
    # Display
    show_image_with_zoom(comparison, window_name)


def resize_with_aspect_ratio(image: np.ndarray, target_width: int = None,
                             target_height: int = None) -> np.ndarray:
    """Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_width: Target width (optional)
        target_height: Target height (optional)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if target_width and target_height:
        # Fit within both dimensions
        scale = min(target_width / w, target_height / h)
    elif target_width:
        scale = target_width / w
    elif target_height:
        scale = target_height / h
    else:
        return image
    
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """Create thumbnail with padding to maintain aspect ratio.
    
    Args:
        image: Input image
        size: Target size (width, height)
        
    Returns:
        Thumbnail image with padding
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    
    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with padding
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def add_text_overlay(image: np.ndarray, text: str, 
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 0.7,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2,
                    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0)) -> np.ndarray:
    """Add text overlay with optional background.
    
    Args:
        image: Input image
        text: Text to display
        position: Text position (x, y)
        font_scale: Font size scale
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Background color (BGR), None for no background
        
    Returns:
        Image with text overlay
    """
    result = image.copy()
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                  font_scale, thickness)
    
    x, y = position
    
    # Draw background if specified
    if bg_color:
        padding = 5
        cv2.rectangle(result, 
                     (x - padding, y - text_h - padding),
                     (x + text_w + padding, y + baseline + padding),
                     bg_color, -1)
    
    # Draw text
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness, cv2.LINE_AA)
    
    return result


def validate_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
    """Validate if ROI is within image bounds.
    
    Args:
        image: Image to validate against
        roi: ROI as (x, y, w, h)
        
    Returns:
        True if valid, False otherwise
    """
    h, w = image.shape[:2]
    x, y, crop_w, crop_h = roi
    
    return (x >= 0 and y >= 0 and 
            x + crop_w <= w and y + crop_h <= h and
            crop_w > 0 and crop_h > 0)


def scale_roi(roi: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    """Scale ROI coordinates by a factor.
    
    Useful when applying ROI from one image size to another.
    
    Args:
        roi: Original ROI as (x, y, w, h)
        scale: Scale factor
        
    Returns:
        Scaled ROI
    """
    x, y, w, h = roi
    return (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
