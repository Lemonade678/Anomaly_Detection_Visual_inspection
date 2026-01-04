"""Image utility functions for display, cropping, and manipulation."""
import cv2
import numpy as np
import json
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


def show_image_with_zoom(image: np.ndarray, window_name: str = "Image Viewer",
                        max_width: int = 1200, max_height: int = 900) -> None:
    """Display image in a window with automatic sizing."""
    h, w = image.shape[:2]
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
    """Draw ROI rectangle overlay on image."""
    result = image.copy()
    x, y, w, h = roi
    
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    if label:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (x, y - text_size[1] - 10), 
                     (x + text_size[0] + 10, y), color, -1)
        cv2.putText(result, label, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def save_crop_pattern(roi: Tuple[int, int, int, int], 
                     filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save crop pattern to JSON file."""
    pattern = {
        'roi': {'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3]},
        'metadata': metadata or {}
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(pattern, f, indent=2)


def load_crop_pattern(filepath: str) -> Optional[Tuple[int, int, int, int]]:
    """Load crop pattern from JSON file."""
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
    """Apply crop pattern to image with validation."""
    h, w = image.shape[:2]
    x, y, crop_w, crop_h = roi
    
    if validate:
        if x < 0 or y < 0 or x + crop_w > w or y + crop_h > h:
            return None
    else:
        x = max(0, x)
        y = max(0, y)
        crop_w = min(crop_w, w - x)
        crop_h = min(crop_h, h - y)
    
    return image[y:y+crop_h, x:x+crop_w]


def resize_with_aspect_ratio(image: np.ndarray, target_width: int = None,
                             target_height: int = None) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    if target_width and target_height:
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
    """Create thumbnail with padding to maintain aspect ratio."""
    target_w, target_h = size
    h, w = image.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
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
    """Add text overlay with optional background."""
    result = image.copy()
    
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                  font_scale, thickness)
    
    x, y = position
    
    if bg_color:
        padding = 5
        cv2.rectangle(result, 
                     (x - padding, y - text_h - padding),
                     (x + text_w + padding, y + baseline + padding),
                     bg_color, -1)
    
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness, cv2.LINE_AA)
    
    return result


def validate_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
    """Validate if ROI is within image bounds."""
    h, w = image.shape[:2]
    x, y, crop_w, crop_h = roi
    
    return (x >= 0 and y >= 0 and 
            x + crop_w <= w and y + crop_h <= h and
            crop_w > 0 and crop_h > 0)


def scale_roi(roi: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    """Scale ROI coordinates by a factor."""
    x, y, w, h = roi
    return (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
