"""SSIM wrapper for the inspection pipeline."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def calc_ssim(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """Compute SSIM between two images (assumes BGR arrays).

    Ensures images are same size by resizing image2 to image1 if necessary.
    
    Args:
        image1: First image (BGR)
        image2: Second image (BGR)
        
    Returns:
        Tuple of (score, heatmap):
        - score: float in range [-1, 1], where 1.0 means identical
        - heatmap: BGR image visualizing the difference
    """
    if image1 is None or image2 is None:
        return 0.0, np.zeros((100, 100, 3), dtype=np.uint8)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if (h1, w1) != (h2, w2):
        image2 = cv2.resize(image2, (w1, h1))

    # structural_similarity with full=True returns the SSIM map
    score, diff = structural_similarity(image1, image2, full=True, channel_axis=-1)
    
    # Average across channels if multichannel
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)
        
    # Normalize to 0-1, clip just in case
    diff = np.clip(diff, 0, 1)
    
    # Invert so anomalies (low SSIM) are high values (bright)
    anomaly_map = 1.0 - diff
    
    # Convert to uint8 and apply colormap
    anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    
    return float(score), heatmap
