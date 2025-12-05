"""SSIM wrapper used by the inspection pipeline."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def calc_ssim(image1: np.ndarray, image2: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute SSIM between two images (assumes BGR arrays).

    Ensures images are same size by resizing image2 to image1 if necessary.
    Returns:
        score: float in range [-1, 1], where 1.0 means identical.
        diff_map: heatmap image (BGR) visualizing the difference.
    """
    if image1 is None or image2 is None:
        return 0.0, np.zeros((100, 100, 3), dtype=np.uint8)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if (h1, w1) != (h2, w2):
        image2 = cv2.resize(image2, (w1, h1))

    # structural_similarity expects channel_axis for color images
    # full=True returns the full structural similarity image
    score, diff = structural_similarity(image1, image2, full=True, channel_axis=-1)
    
    # diff is in range [-1, 1] (roughly), or [0, 1] depending on input range.
    # Since input is uint8 [0, 255], skimage converts to float.
    # The diff map values: 1.0 means identical, lower means different.
    
    # Create anomaly map: 1.0 - diff (so 0 is good, 1 is bad)
    # Average across channels if it's multichannel
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)
        
    # Normalize to 0-1 for visualization
    # SSIM map can be negative, but usually [0, 1] for images.
    # Let's clip to [0, 1] just in case
    diff = np.clip(diff, 0, 1)
    
    # Invert so that anomalies (low SSIM) are high values (bright)
    anomaly_map = 1.0 - diff
    
    # Convert to uint8
    anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
    
    # Apply colormap (JET) for better visualization
    heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    
    return float(score), heatmap
