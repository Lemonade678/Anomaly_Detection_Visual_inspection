"""Pixel-difference based matching/analysis."""
import cv2
import numpy as np


def run_pixel_matching(golden_image: np.ndarray, aligned_image: np.ndarray, pixel_thresh: int, count_thresh: int,
                       area_thresh: float = 20.0, min_contour_area: int = 50):
    """Run pixel-based anomaly detection and return results dict.
    
    Now counts TOTAL ANOMALOUS PIXELS instead of just contour count.

    Returns dict keys: 
        - area_score: Percentage of anomalous pixels
        - anomaly_count: Number of detected contours (legacy)
        - anomalous_pixel_count: TOTAL number of anomalous pixels
        - count_thresh: Threshold used
        - verdict: "Anomaly" or "Normal"
        - heatmap: Visualization heatmap
        - contour_map: Image with bounding boxes
        - anomaly_mask: Binary mask of all anomalous pixels
    """
    golden_gray = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(golden_gray, aligned_gray)

    thresh_val = int(pixel_thresh)
    _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count total anomalous pixels
    anomalous_pixel_count = np.count_nonzero(dilated)

    contour_image = aligned_image.copy()
    anomaly_count = 0
    for c in contours:
        if cv2.contourArea(c) < min_contour_area:
            continue
        anomaly_count += 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    area_score = (anomalous_pixel_count / dilated.size) * 100.0
    
    # Verdict based on PIXEL COUNT (not contour count)
    if area_score > area_thresh or anomalous_pixel_count > int(count_thresh):
        verdict = "Anomaly"
    else:
        verdict = "Normal"

    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)

    return {
        'area_score': float(area_score),
        'anomaly_count': int(anomaly_count),  # Legacy: contour count
        'anomalous_pixel_count': int(anomalous_pixel_count),  # NEW: Total anomalous pixels
        'count_thresh': int(count_thresh),
        'verdict': verdict,
        'heatmap': heatmap,
        'contour_map': contour_image,
        'anomaly_mask': dilated  # NEW: Binary mask of anomalies
    }
