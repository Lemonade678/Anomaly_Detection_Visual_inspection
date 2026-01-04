"""Illumination normalization and light sensitivity processing.

Combines V2's comprehensive light handling with preprocessing for robust comparison.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from .config import LightSensitivityMode, LightSensitivityConfig


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to an image.
    
    Args:
        image: Input BGR image
        gamma: Gamma value (>1 brightens, <1 darkens)
        
    Returns:
        Gamma-corrected BGR image
    """
    if gamma == 1.0:
        return image
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in range(256)]).astype(np.uint8)
    
    return cv2.LUT(image, table)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_normalized = clahe.apply(l_channel)
    
    lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)


def highlight_recovery(image: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Recover detail from overexposed/highlight regions.
    
    Args:
        image: Input BGR image
        threshold: Pixel value above which is considered overexposed
        
    Returns:
        Image with recovered highlights
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    highlight_mask = l_channel > threshold
    l_channel[highlight_mask] = threshold + (l_channel[highlight_mask] - threshold) * 0.3
    
    lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def hdr_processing(image: np.ndarray, 
                   sigma_spatial: float = 10.0,
                   sigma_range: float = 0.1) -> np.ndarray:
    """Apply HDR-like processing for high dynamic range scenes.
    
    Args:
        image: Input BGR image
        sigma_spatial: Spatial sigma for bilateral filter
        sigma_range: Range sigma for bilateral filter
        
    Returns:
        HDR-processed BGR image
    """
    img_float = image.astype(np.float32) / 255.0
    
    base = cv2.bilateralFilter(img_float, -1, sigma_range, sigma_spatial)
    detail = img_float - base
    
    base_compressed = np.power(base + 0.001, 0.5)
    detail_enhanced = detail * 1.5
    
    result = base_compressed + detail_enhanced
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def gold_pad_hsv_filter(image: np.ndarray, 
                        hue_low: int = 15, hue_high: int = 40,
                        sat_low: int = 50, sat_high: int = 255,
                        val_low: int = 80, val_high: int = 255,
                        enhance_contrast: bool = True,
                        return_mask: bool = False) -> np.ndarray:
    """Apply HSV filtering specifically for gold bonding pads.
    
    Gold bonding pads typically appear in yellow-golden colors.
    This function isolates and enhances those regions for better inspection.
    
    Args:
        image: Input BGR image
        hue_low: Lower hue bound (0-179 in OpenCV, gold is around 15-40)
        hue_high: Upper hue bound
        sat_low: Lower saturation bound (0-255)
        sat_high: Upper saturation bound
        val_low: Lower value/brightness bound (0-255)
        val_high: Upper value/brightness bound
        enhance_contrast: Apply CLAHE enhancement to gold regions
        return_mask: If True, return the mask; if False, return enhanced image
        
    Returns:
        Enhanced BGR image with gold regions highlighted, or the mask
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for gold color range
    lower_gold = np.array([hue_low, sat_low, val_low])
    upper_gold = np.array([hue_high, sat_high, val_high])
    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel)
    
    if return_mask:
        return gold_mask
    
    # Create enhanced version
    result = image.copy()
    
    if enhance_contrast:
        # Apply CLAHE to enhance contrast in gold regions
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Apply enhancement only in gold regions using mask
        mask_3ch = cv2.merge([gold_mask, gold_mask, gold_mask])
        
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Blend enhanced regions with original
        result = np.where(mask_3ch > 0, enhanced_bgr, image)
    
    # Optionally boost saturation in gold regions for visibility
    hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Boost saturation slightly in gold regions
    hsv_result[:, :, 1] = np.where(
        gold_mask > 0,
        np.clip(hsv_result[:, :, 1] * 1.2, 0, 255),
        hsv_result[:, :, 1]
    )
    
    result = cv2.cvtColor(hsv_result.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result


def get_gold_pad_mask(image: np.ndarray, config: Optional[LightSensitivityConfig] = None) -> np.ndarray:
    """Get a binary mask of gold pad regions.
    
    Useful for focusing inspection only on gold bonding pad areas.
    
    Args:
        image: Input BGR image
        config: Light sensitivity config with gold pad HSV parameters
        
    Returns:
        Binary mask with gold regions as white (255)
    """
    if config is None:
        config = LightSensitivityConfig()
    
    return gold_pad_hsv_filter(
        image,
        hue_low=config.gold_hue_low,
        hue_high=config.gold_hue_high,
        sat_low=config.gold_saturation_low,
        sat_high=config.gold_saturation_high,
        val_low=config.gold_value_low,
        val_high=config.gold_value_high,
        return_mask=True
    )


def detect_light_condition(image: np.ndarray) -> str:
    """Analyze image to detect lighting condition.
    
    Args:
        image: Input BGR image
        
    Returns:
        'low', 'high', 'hdr', or 'standard'
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    if std_val > 70:
        return 'hdr'
    if mean_val < 60:
        return 'low'
    if mean_val > 200:
        return 'high'
    
    return 'standard'


def apply_light_sensitivity_mode(image: np.ndarray, 
                                  mode: LightSensitivityMode,
                                  config: Optional[LightSensitivityConfig] = None) -> np.ndarray:
    """Apply light sensitivity processing based on mode.
    
    Args:
        image: Input BGR image
        mode: Light sensitivity mode to apply
        config: Configuration parameters (uses defaults if None)
        
    Returns:
        Processed BGR image
    """
    if config is None:
        config = LightSensitivityConfig()
    
    # Auto-detect if mode is AUTO
    if mode == LightSensitivityMode.AUTO:
        detected = detect_light_condition(image)
        mode = LightSensitivityMode(detected)
    
    if mode == LightSensitivityMode.LOW_LIGHT:
        result = gamma_correction(image, config.low_light_gamma)
        result = apply_clahe(result, clip_limit=config.low_light_clahe_clip)
        return result
        
    elif mode == LightSensitivityMode.HIGH_LIGHT:
        result = highlight_recovery(image, config.highlight_threshold)
        result = gamma_correction(result, config.high_light_gamma)
        return result
        
    elif mode == LightSensitivityMode.HDR:
        return hdr_processing(image, config.hdr_sigma_spatial, config.hdr_sigma_range)
    
    elif mode == LightSensitivityMode.GOLD_PAD:
        # Apply gold bonding pad HSV filter
        return gold_pad_hsv_filter(
            image,
            hue_low=config.gold_hue_low,
            hue_high=config.gold_hue_high,
            sat_low=config.gold_saturation_low,
            sat_high=config.gold_saturation_high,
            val_low=config.gold_value_low,
            val_high=config.gold_value_high,
            enhance_contrast=config.gold_enhance_contrast
        )
        
    else:  # STANDARD
        if config.gamma != 1.0:
            return gamma_correction(image, config.gamma)
        return image


def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of source image to reference image.
    
    Best method for master vs test comparison.
    
    Args:
        source: Source image (test image to be transformed)
        reference: Reference image (master/golden image)
        
    Returns:
        Source image with histogram matched to reference
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    
    matched_lab = np.zeros_like(source_lab)
    
    for i in range(3):
        matched_lab[:, :, i] = _match_channel_histogram(
            source_lab[:, :, i], 
            reference_lab[:, :, i]
        )
    
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def _match_channel_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of a single channel."""
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.searchsorted(ref_cdf, src_cdf[i])
        lookup[i] = min(j, 255)
    
    return lookup[source]


def preprocess_pair(master: np.ndarray, test: np.ndarray, 
                    method: str = "match_histogram") -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess master and test image pair for comparison.
    
    Args:
        master: Master/golden image (BGR)
        test: Test image (BGR)
        method: Preprocessing method:
            - "match_histogram": Match test histogram to master (RECOMMENDED)
            - "clahe_both": Apply CLAHE to both images
            - "normalize_both": Apply mean-std normalization to both
            - "none": No preprocessing
            
    Returns:
        Tuple of (preprocessed_master, preprocessed_test)
    """
    if method == "match_histogram":
        return master, match_histograms(test, master)
    elif method == "clahe_both":
        return apply_clahe(master), apply_clahe(test)
    elif method == "normalize_both":
        return mean_std_normalization(master), mean_std_normalization(test)
    elif method == "none":
        return master, test
    else:
        return master, match_histograms(test, master)


def mean_std_normalization(image: np.ndarray, 
                           target_mean: float = 128.0,
                           target_std: float = 64.0) -> np.ndarray:
    """Normalize image to have target mean and standard deviation.
    
    Args:
        image: Input BGR image
        target_mean: Target mean intensity
        target_std: Target standard deviation
        
    Returns:
        Normalized BGR image
    """
    img_float = image.astype(np.float32)
    mean = np.mean(img_float, axis=(0, 1), keepdims=True)
    std = np.std(img_float, axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    
    normalized = (img_float - mean) / std * target_std + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)


def equalize_histogram_gray(image: np.ndarray) -> np.ndarray:
    """Equalize histogram of grayscale image using CLAHE (V1 method)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
