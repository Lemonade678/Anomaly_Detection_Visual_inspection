"""Analysis utilities for inspection results.

Contains classes and functions for:
- Mapping anomaly locations to grid/sectors
- Generating annotated images with defect highlights
- Exporting results to JSON
"""
import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class AnomalyLocationMapper:
    """Maps and reports anomaly locations with pixel coordinates."""
    
    def __init__(self):
        self.anomaly_regions = []
        self.centroids = []
    
    def analyze_mask(self, anomaly_mask: np.ndarray, min_area: int = 50) -> dict:
        """Analyze anomaly mask to extract location information."""
        self.anomaly_regions = []
        self.centroids = []
        
        # Ensure mask is uint8
        if anomaly_mask.dtype != np.uint8:
            anomaly_mask = anomaly_mask.astype(np.uint8)

        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = anomaly_mask.shape[:2]
        
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
            
            region_info = {
                'id': len(self.anomaly_regions) + 1,
                'bbox': (x, y, bw, bh),
                'centroid': (cx, cy),
                'area_px': int(area),
                'sector': f"{sector_y}-{sector_x}",
                'relative_pos': (round(cx/w * 100, 1), round(cy/h * 100, 1))
            }
            
            self.anomaly_regions.append(region_info)
            self.centroids.append((cx, cy))
        
        total_anomalous_pixels = np.count_nonzero(anomaly_mask)
        
        return {
            'regions': self.anomaly_regions,
            'total_regions': len(self.anomaly_regions),
            'total_anomalous_pixels': total_anomalous_pixels,
            'coverage_percent': round(total_anomalous_pixels / (h * w) * 100, 3) if h*w > 0 else 0,
            'image_size': (w, h)
        }
    
    def create_annotated_image(self, base_image: np.ndarray, 
                                show_centroids: bool = True,
                                show_labels: bool = True,
                                show_grid: bool = True) -> np.ndarray:
        """Create an annotated image with anomaly locations."""
        if base_image is None:
            return None
            
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
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
            
            if show_centroids:
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(annotated, (cx, cy), 7, (0, 0, 255), 2)
            
            if show_labels:
                label = f"#{region_id}"
                # Draw background for text for better visibility
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x, y-lh-4), (x+lw, y), (0,0,0), -1)
                cv2.putText(annotated, label, (x, y-2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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


def export_results_as_json(result: dict, output_path: str = None) -> str:
    """Export inspection results to JSON file.
    
    Args:
        result: Result dict from inspection
        output_path: Optional output file path (auto-generates if None)
        
    Returns:
        Path to saved JSON file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inspection_result_{timestamp}.json"
    
    # Convert numpy arrays, datetime objects and non-serializable types
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    # Simplified result for JSON (exclude heavy image data)
    json_result = {
        'timestamp': datetime.now().isoformat(),
        'verdict': result.get('verdict', 'Unknown'),
        'method': result.get('method', 'Unknown'),
        'metrics': {
            'ssim_score': result.get('ssim_score'),
            'area_score': result.get('area_score'),
            'anomaly_count': result.get('anomaly_count'),
            'confidence': result.get('confidence'),
            'processing_time': result.get('processing_time')
        }
    }
    
    if 'location_data' in result:
        json_result['location_data'] = result['location_data']
    
    if 'location_summary' in result:
        json_result['location_summary'] = result['location_summary']
        
    if 'error' in result:
        json_result['error'] = result['error']
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=default_serializer)
        return output_path
    except Exception as e:
        print(f"Failed to export JSON: {e}")
        return ""
