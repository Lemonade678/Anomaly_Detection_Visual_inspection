"""
Template Matching Module for PCB Inspection.

Provides template-based inspection that is more robust to camera position
variance than pixel-wise comparison. Divides golden image into regions and
finds matching regions in the test image using cv2.matchTemplate().
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TemplateRegion:
    """Represents a template region from the golden image."""
    id: int
    x: int
    y: int
    width: int
    height: int
    template: np.ndarray
    
    # Match results (filled after matching)
    match_x: int = -1
    match_y: int = -1
    match_score: float = 0.0
    matched: bool = False


@dataclass
class TemplateMatchConfig:
    """Configuration for template matching inspection."""
    grid_cols: int = 4           # Number of columns in grid
    grid_rows: int = 4           # Number of rows in grid
    match_threshold: float = 0.7  # Minimum correlation for a match
    search_margin: int = 50      # Extra pixels to search around expected position
    diff_threshold: int = 30     # Pixel difference threshold for anomaly
    min_anomaly_area: int = 100  # Minimum anomaly area in pixels


def divide_into_templates(image: np.ndarray, 
                          grid_cols: int = 4, 
                          grid_rows: int = 4) -> List[TemplateRegion]:
    """Divide an image into a grid of template regions.
    
    Args:
        image: Input image (golden reference)
        grid_cols: Number of columns
        grid_rows: Number of rows
        
    Returns:
        List of TemplateRegion objects
    """
    h, w = image.shape[:2]
    cell_w = w // grid_cols
    cell_h = h // grid_rows
    
    templates = []
    idx = 0
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * cell_w
            y = row * cell_h
            
            # Handle edge cells (may be slightly larger)
            w_actual = cell_w if col < grid_cols - 1 else w - x
            h_actual = cell_h if row < grid_rows - 1 else h - y
            
            template_img = image[y:y+h_actual, x:x+w_actual].copy()
            
            templates.append(TemplateRegion(
                id=idx,
                x=x,
                y=y,
                width=w_actual,
                height=h_actual,
                template=template_img
            ))
            idx += 1
    
    return templates


def find_template_in_image(template: TemplateRegion,
                           test_image: np.ndarray,
                           search_margin: int = 50,
                           match_threshold: float = 0.7) -> TemplateRegion:
    """Find a template region in the test image.
    
    Uses cv2.matchTemplate with normalized cross-correlation.
    Searches around the expected position with a margin.
    
    Args:
        template: Template region to find
        test_image: Test image to search in
        search_margin: Extra pixels to search around expected position
        match_threshold: Minimum correlation score for valid match
        
    Returns:
        Updated TemplateRegion with match results
    """
    h, w = test_image.shape[:2]
    
    # Define search region (around expected position)
    search_x1 = max(0, template.x - search_margin)
    search_y1 = max(0, template.y - search_margin)
    search_x2 = min(w, template.x + template.width + search_margin)
    search_y2 = min(h, template.y + template.height + search_margin)
    
    search_region = test_image[search_y1:search_y2, search_x1:search_x2]
    
    # Check if search region is large enough
    if search_region.shape[0] < template.height or search_region.shape[1] < template.width:
        template.matched = False
        return template
    
    # Convert to grayscale for matching
    if len(template.template.shape) == 3:
        template_gray = cv2.cvtColor(template.template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template.template
        
    if len(search_region.shape) == 3:
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_region
    
    # Perform template matching
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= match_threshold:
        # Convert local coords back to global
        template.match_x = search_x1 + max_loc[0]
        template.match_y = search_y1 + max_loc[1]
        template.match_score = max_val
        template.matched = True
    else:
        template.matched = False
        template.match_score = max_val
    
    return template


def compare_matched_regions(template: TemplateRegion,
                            golden_image: np.ndarray,
                            test_image: np.ndarray,
                            diff_threshold: int = 30) -> Tuple[np.ndarray, int]:
    """Compare a matched template region for anomalies.
    
    Args:
        template: Matched template region
        golden_image: Golden reference image
        test_image: Test image
        diff_threshold: Pixel difference threshold
        
    Returns:
        Tuple of (difference mask, anomaly pixel count)
    """
    if not template.matched:
        return np.zeros((template.height, template.width), dtype=np.uint8), 0
    
    # Extract golden region
    golden_region = golden_image[
        template.y:template.y+template.height,
        template.x:template.x+template.width
    ]
    
    # Extract matched test region
    test_region = test_image[
        template.match_y:template.match_y+template.height,
        template.match_x:template.match_x+template.width
    ]
    
    # Handle size mismatch at edges
    min_h = min(golden_region.shape[0], test_region.shape[0])
    min_w = min(golden_region.shape[1], test_region.shape[1])
    
    golden_region = golden_region[:min_h, :min_w]
    test_region = test_region[:min_h, :min_w]
    
    # Convert to grayscale
    if len(golden_region.shape) == 3:
        golden_gray = cv2.cvtColor(golden_region, cv2.COLOR_BGR2GRAY)
    else:
        golden_gray = golden_region
        
    if len(test_region.shape) == 3:
        test_gray = cv2.cvtColor(test_region, cv2.COLOR_BGR2GRAY)
    else:
        test_gray = test_region
    
    # Calculate difference
    diff = cv2.absdiff(golden_gray, test_gray)
    _, diff_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    
    anomaly_count = cv2.countNonZero(diff_mask)
    
    return diff_mask, anomaly_count


def run_template_inspection(golden_image: np.ndarray,
                            test_image: np.ndarray,
                            config: Optional[TemplateMatchConfig] = None) -> Dict:
    """Run template-based inspection.
    
    Main function that performs the complete template matching inspection.
    
    Args:
        golden_image: Golden/reference image
        test_image: Test image to inspect
        config: Template matching configuration
        
    Returns:
        Dictionary with results:
        - verdict: 'Normal' or 'Anomaly'
        - templates: List of matched templates
        - anomaly_mask: Combined anomaly mask
        - anomaly_count: Total anomaly pixels
        - match_rate: Percentage of templates matched
        - heatmap: Visual heatmap
        - annotated_image: Test image with annotations
    """
    if config is None:
        config = TemplateMatchConfig()
    
    # Ensure same size
    h1, w1 = golden_image.shape[:2]
    h2, w2 = test_image.shape[:2]
    
    if (h1, w1) != (h2, w2):
        test_image = cv2.resize(test_image, (w1, h1))
    
    # Step 1: Divide golden image into templates
    templates = divide_into_templates(golden_image, config.grid_cols, config.grid_rows)
    
    # Step 2: Find each template in test image
    for i, template in enumerate(templates):
        templates[i] = find_template_in_image(
            template, test_image, 
            config.search_margin, 
            config.match_threshold
        )
    
    # Step 3: Compare matched regions
    total_anomaly_count = 0
    anomaly_mask = np.zeros(golden_image.shape[:2], dtype=np.uint8)
    
    for template in templates:
        if template.matched:
            diff_mask, count = compare_matched_regions(
                template, golden_image, test_image, config.diff_threshold
            )
            total_anomaly_count += count
            
            # Place diff mask in global anomaly mask
            y, x = template.y, template.x
            h, w = diff_mask.shape[:2]
            anomaly_mask[y:y+h, x:x+w] = cv2.bitwise_or(
                anomaly_mask[y:y+h, x:x+w], diff_mask
            )
    
    # Calculate match rate
    matched_count = sum(1 for t in templates if t.matched)
    match_rate = matched_count / len(templates) * 100
    
    # Create heatmap visualization
    heatmap = cv2.applyColorMap(anomaly_mask, cv2.COLORMAP_JET)
    
    # Create annotated image
    annotated = test_image.copy()
    for template in templates:
        color = (0, 255, 0) if template.matched else (0, 0, 255)
        thickness = 1 if template.matched else 2
        
        if template.matched:
            # Draw matched position
            cv2.rectangle(annotated,
                         (template.match_x, template.match_y),
                         (template.match_x + template.width, template.match_y + template.height),
                         color, thickness)
        else:
            # Draw expected position for unmatched
            cv2.rectangle(annotated,
                         (template.x, template.y),
                         (template.x + template.width, template.y + template.height),
                         color, thickness)
    
    # Highlight anomalies in red
    anomaly_overlay = annotated.copy()
    anomaly_overlay[anomaly_mask > 0] = [0, 0, 255]
    annotated = cv2.addWeighted(annotated, 0.7, anomaly_overlay, 0.3, 0)
    
    # Determine verdict
    verdict = 'Anomaly' if total_anomaly_count >= config.min_anomaly_area else 'Normal'
    
    return {
        'verdict': verdict,
        'templates': templates,
        'anomaly_mask': anomaly_mask,
        'anomaly_count': total_anomaly_count,
        'area_score': total_anomaly_count / (h1 * w1) * 100,
        'match_rate': match_rate,
        'matched_templates': matched_count,
        'total_templates': len(templates),
        'heatmap': heatmap,
        'annotated_image': annotated,
        'confidence': match_rate / 100
    }
