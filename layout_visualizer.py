"""Defect layout visualization for batch inspection results."""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def create_defect_layout(strip_results: List[Dict[str, Any]],
                        strip_images: Optional[List[np.ndarray]] = None,
                        layout_type: str = "grid",
                        title: str = "Inspection Results") -> np.ndarray:
    """Create visual defect layout showing all strips with verdicts.
    
    Generates a layout matching workflow Steps 3 & 8, showing:
    - All strips in a grid layout
    - Color-coded borders (Green=NORMAL/PASS, Red=ANOMALY/FAIL)
    - Strip numbers and verdicts
    - Summary statistics
    
    Args:
        strip_results: List of strip inspection results (from BatchInspector)
        strip_images: Optional list of actual strip images to display
        layout_type: "grid" (2x3) or "row" (1x6) layout
        title: Title text for the layout
        
    Returns:
        Layout visualization image (BGR format)
    """
    num_strips = len(strip_results)
    
    if num_strips == 0:
        return _create_empty_layout(title)
    
    # Define layout dimensions
    if layout_type == "grid":
        grid_rows, grid_cols = 2, 3  # 2x3 grid for 6 strips
        if num_strips <= 4:
            grid_rows, grid_cols = 2, 2
    else:  # row layout
        grid_rows, grid_cols = 1, num_strips
    
    # Cell dimensions
    cell_width = 300
    cell_height = 400
    border_thickness = 10
    padding = 20
    
    # Calculate canvas size
    canvas_width = grid_cols * (cell_width + padding) + padding
    canvas_height = (grid_rows * (cell_height + padding) + 
                    padding + 100)  # Extra space for title and summary
    
    # Create canvas (light gray background)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    # Draw title
    cv2.putText(canvas, title, (20, 50), 
               cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 3)
    
    # Draw timestamp
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(canvas, timestamp, (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Draw each strip cell
    y_offset = 120
    
    for idx, result in enumerate(strip_results):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Calculate cell position
        x = padding + col * (cell_width + padding)
        y = y_offset + row * (cell_height + padding)
        
        # Draw cell
        _draw_strip_cell(canvas, result, strip_images[idx] if strip_images else None,
                        x, y, cell_width, cell_height, border_thickness)
    
    # Draw summary at bottom
    _draw_summary(canvas, strip_results, canvas_width, canvas_height)
    
    return canvas


def _draw_strip_cell(canvas: np.ndarray,
                    result: Dict[str, Any],
                    strip_image: Optional[np.ndarray],
                    x: int, y: int, 
                    width: int, height: int,
                    border_thickness: int):
    """Draw a single strip cell in the layout."""
    
    # Determine color based on verdict
    verdict = result.get('verdict', 'UNKNOWN')
    
    if verdict == 'NORMAL' or verdict == 'PASS':
        border_color = (0, 255, 0)  # Green
        text_color = (0, 180, 0)
        bg_color = (240, 255, 240)  # Light green
    elif verdict == 'ANOMALY' or verdict == 'FAIL':
        border_color = (0, 0, 255)  # Red
        text_color = (0, 0, 200)
        bg_color = (255, 240, 240)  # Light red
    else:  # ERROR
        border_color = (0, 165, 255)  # Orange
        text_color = (0, 140, 200)
        bg_color = (255, 250, 240)  # Light orange
    
    # Draw background
    cv2.rectangle(canvas, (x, y), (x + width, y + height), bg_color, -1)
    
    # Draw border
    cv2.rectangle(canvas, (x, y), (x + width, y + height), 
                 border_color, border_thickness)
    
    # Draw strip number
    strip_num = result.get('strip_number', '?')
    cv2.putText(canvas, f"Strip #{strip_num}", (x + 10, y + 35),
               cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    
    # Draw strip image if available
    if strip_image is not None:
        img_height = height - 120
        img_width = width - 20
        
        # Resize strip image to fit
        strip_resized = _resize_with_aspect_ratio(strip_image, img_width, img_height)
        
        # Center the image in the cell
        img_h, img_w = strip_resized.shape[:2]
        img_x = x + (width - img_w) // 2
        img_y = y + 50
        
        canvas[img_y:img_y+img_h, img_x:img_x+img_w] = strip_resized
    
    # Draw verdict text
    cv2.putText(canvas, verdict, (x + 10, y + height - 70),
               cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, text_color, 2)
    
    # Draw SSIM score if available
    ssim = result.get('ssim_score', 0)
    cv2.putText(canvas, f"SSIM: {ssim:.3f}", (x + 10, y + height - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    
    # Draw anomaly count if applicable
    if verdict in ['ANOMALY', 'FAIL']:
        anomaly_count = result.get('anomaly_count', 0)
        cv2.putText(canvas, f"Defects: {anomaly_count}", (x + 10, y + height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)


def _draw_summary(canvas: np.ndarray, 
                 results: List[Dict[str, Any]],
                 canvas_width: int,
                 canvas_height: int):
    """Draw summary statistics at the bottom of the layout."""
    
    total = len(results)
    normal_count = sum(1 for r in results if r.get('verdict') in ['NORMAL', 'PASS'])
    anomaly_count = sum(1 for r in results if r.get('verdict') in ['ANOMALY', 'FAIL'])
    error_count = sum(1 for r in results if r.get('verdict') == 'ERROR')
    
    # Summary box position
    summary_y = canvas_height - 50
    
    summary_text = (f"Total: {total}  |  "
                   f"PASS: {normal_count}  |  "
                   f"FAIL: {anomaly_count}  |  "
                   f"ERROR: {error_count}")
    
    cv2.putText(canvas, summary_text, (20, summary_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Overall verdict
    if anomaly_count > 0:
        overall_verdict = "LOT: REJECT"
        color = (0, 0, 255)
    elif error_count > 0:
        overall_verdict = "LOT: CHECK"
        color = (0, 165, 255)
    else:
        overall_verdict = "LOT: ACCEPT"
        color = (0, 255, 0)
    
    cv2.putText(canvas, overall_verdict, (canvas_width - 300, summary_y),
               cv2.FONT_HERSHEY_COMPLEX, 1, color, 3)


def _resize_with_aspect_ratio(image: np.ndarray, 
                              target_width: int, 
                              target_height: int) -> np.ndarray:
    """Resize image maintaining aspect ratio to fit within target dimensions."""
    h, w = image.shape[:2]
    
    if h == 0 or w == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate scaling factor
    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w == 0 or new_h == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def _create_empty_layout(title: str) -> np.ndarray:
    """Create empty layout when no results are available."""
    canvas = np.ones((400, 800, 3), dtype=np.uint8) * 240
    
    cv2.putText(canvas, title, (20, 50),
               cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 3)
    
    cv2.putText(canvas, "No inspection results available", (200, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    return canvas


def create_comparison_layout(master_strips: List[np.ndarray],
                            test_strips: List[np.ndarray],
                            results: List[Dict[str, Any]],
                            title: str = "Master vs Test Comparison") -> np.ndarray:
    """Create side-by-side comparison layout of master and test strips.
    
    Shows master on left, test on right, with verdict overlay.
    
    Args:
        master_strips: List of master strip images
        test_strips: List of test strip images
        results: List of inspection results
        title: Layout title
        
    Returns:
        Comparison layout image
    """
    num_strips = min(len(master_strips), len(test_strips))
    
    if num_strips == 0:
        return _create_empty_layout(title)
    
    # Layout dimensions
    strip_width = 250
    strip_height = 300
    padding = 20
    
    canvas_width = 2 * strip_width + 3 * padding + 100  # Master + Test + gap
    canvas_height = num_strips * (strip_height + padding) + padding + 80
    
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    # Title
    cv2.putText(canvas, title, (20, 50),
               cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
    
    # Column headers
    cv2.putText(canvas, "MASTER", (padding + 80, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
    cv2.putText(canvas, "TEST", (2*padding + strip_width + 90, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)
    
    # Draw each pair
    for idx in range(num_strips):
        y = 100 + idx * (strip_height + padding)
        
        # Master strip
        master_resized = _resize_with_aspect_ratio(master_strips[idx], 
                                                   strip_width, strip_height)
        m_h, m_w = master_resized.shape[:2]
        canvas[y:y+m_h, padding:padding+m_w] = master_resized
        
        # Test strip
        test_resized = _resize_with_aspect_ratio(test_strips[idx],
                                                strip_width, strip_height)
        t_h, t_w = test_resized.shape[:2]
        x_test = 2*padding + strip_width
        canvas[y:y+t_h, x_test:x_test+t_w] = test_resized
        
        # Verdict indicator
        if idx < len(results):
            verdict = results[idx].get('verdict', 'UNKNOWN')
            color = (0, 255, 0) if verdict in ['NORMAL', 'PASS'] else (0, 0, 255)
            
            x_indicator = canvas_width - 80
            y_indicator = y + strip_height // 2
            
            cv2.circle(canvas, (x_indicator, y_indicator), 20, color, -1)
            cv2.putText(canvas, "OK" if color == (0, 255, 0) else "NG",
                       (x_indicator - 15, y_indicator + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas
