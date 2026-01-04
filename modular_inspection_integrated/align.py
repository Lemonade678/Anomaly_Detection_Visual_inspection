"""Multi-method image alignment combining V1 and V2 features.

Features:
- Multiple alignment methods: Phase Correlation, ORB, SIFT, ECC, Auto
- CLAHE preprocessing before feature detection (V1)
- Returns valid_area_mask to avoid border artifacts (V1)
- Detailed alignment info for debugging (V2)
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from .config import AlignmentMethod, AlignmentConfig


# Configuration constants
MAX_NUM_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15
MIN_MATCH_COUNT = 10


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Equalize histogram using CLAHE (from V1)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def align_images(
    golden_image: np.ndarray, 
    test_image: np.ndarray,
    method: AlignmentMethod = AlignmentMethod.AUTO,
    config: Optional[AlignmentConfig] = None,
    border_value: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, Tuple[float, float], float, np.ndarray]:
    """Align test_image to golden_image using specified method.
    
    Args:
        golden_image: Reference/master image (BGR)
        test_image: Image to align (BGR)
        method: Alignment method to use
        config: Alignment configuration
        border_value: Color for border pixels after warping
        
    Returns:
        Tuple of:
        - aligned_image: Test image aligned to match golden image
        - shift: (dx, dy) translation component
        - response: Match quality score (0.0 to 1.0)
        - valid_mask: Binary mask (255=valid, 0=border) for post-alignment
    """
    if config is None:
        config = AlignmentConfig()
    
    # Resize test to match golden
    h_golden, w_golden = golden_image.shape[:2]
    test_resized = cv2.resize(test_image, (w_golden, h_golden))
    
    # Full mask for fallback
    full_mask = np.ones((h_golden, w_golden), dtype=np.uint8) * 255
    
    # Try methods in order for AUTO mode
    if method == AlignmentMethod.AUTO:
        for fallback_method in config.fallback_order:
            result = _try_alignment(golden_image, test_resized, fallback_method, 
                                   config, border_value)
            if result[2] >= config.phase_min_response:  # response threshold
                return result
        # All failed, return last attempt
        return result
    else:
        return _try_alignment(golden_image, test_resized, method, config, border_value)


def _try_alignment(
    golden_image: np.ndarray,
    test_image: np.ndarray,
    method: AlignmentMethod,
    config: AlignmentConfig,
    border_value: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[float, float], float, np.ndarray]:
    """Try a specific alignment method."""
    
    h, w = golden_image.shape[:2]
    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    if method == AlignmentMethod.PHASE_CORRELATION:
        return _align_phase_correlation(golden_image, test_image, config)
    elif method == AlignmentMethod.ORB_HOMOGRAPHY:
        return _align_orb(golden_image, test_image, config, border_value)
    elif method == AlignmentMethod.SIFT_HOMOGRAPHY:
        return _align_sift(golden_image, test_image, config, border_value)
    elif method == AlignmentMethod.ECC:
        return _align_ecc(golden_image, test_image, config)
    else:
        return test_image, (0.0, 0.0), 0.0, full_mask


def _align_phase_correlation(
    golden: np.ndarray, 
    test: np.ndarray,
    config: AlignmentConfig
) -> Tuple[np.ndarray, Tuple[float, float], float, np.ndarray]:
    """Phase correlation alignment (translation only, fast)."""
    h, w = golden.shape[:2]
    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    
    # Phase correlation
    (dx, dy), response = cv2.phaseCorrelate(
        golden_gray.astype(np.float64),
        test_gray.astype(np.float64)
    )
    
    if response < config.phase_min_response:
        return test, (0.0, 0.0), float(response), full_mask
    
    # Apply translation
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(test, M, (w, h), borderValue=(0, 0, 0))
    
    # Valid mask
    valid_mask = cv2.warpAffine(full_mask, M, (w, h), borderValue=0)
    
    return aligned, (float(dx), float(dy)), float(response), valid_mask


def _align_orb(
    golden: np.ndarray, 
    test: np.ndarray,
    config: AlignmentConfig,
    border_value: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[float, float], float, np.ndarray]:
    """ORB feature matching alignment (handles rotation/scale)."""
    h, w = golden.shape[:2]
    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Convert to grayscale and apply CLAHE (V1 improvement)
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    golden_eq = equalize_histogram(golden_gray)
    test_eq = equalize_histogram(test_gray)
    
    # Detect ORB features
    orb = cv2.ORB_create(config.orb_max_features)
    kp_golden, desc_golden = orb.detectAndCompute(golden_eq, None)
    kp_test, desc_test = orb.detectAndCompute(test_eq, None)
    
    if desc_golden is None or desc_test is None:
        return test, (0.0, 0.0), 0.0, full_mask
    
    if len(kp_golden) < config.orb_min_matches or len(kp_test) < config.orb_min_matches:
        return test, (0.0, 0.0), 0.1, full_mask
    
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc_test, desc_golden, None)
    
    if len(matches) < config.orb_min_matches:
        return test, (0.0, 0.0), 0.1, full_mask
    
    # Sort by distance and keep best
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(config.orb_min_matches, int(len(matches) * config.orb_good_match_percent))
    good_matches = matches[:num_good]
    
    # Calculate response
    avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
    response = 1.0 - (avg_distance / 256)
    response = max(0.0, min(1.0, response))
    
    # Extract points
    pts_test = np.float32([kp_test[m.queryIdx].pt for m in good_matches])
    pts_golden = np.float32([kp_golden[m.trainIdx].pt for m in good_matches])
    
    # Find homography
    H, mask = cv2.findHomography(pts_test, pts_golden, cv2.RANSAC, 
                                  config.ransac_reproj_threshold)
    
    if H is None:
        return test, (0.0, 0.0), 0.1, full_mask
    
    # Extract translation
    dx, dy = H[0, 2], H[1, 2]
    
    # Apply homography
    aligned = cv2.warpPerspective(test, H, (w, h), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=border_value)
    
    # Create valid mask
    dummy = np.ones((h, w), dtype=np.uint8) * 255
    valid_mask = cv2.warpPerspective(dummy, H, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
    
    # Adjust response by inlier ratio
    if mask is not None:
        inlier_ratio = np.sum(mask) / len(mask)
        response *= inlier_ratio
    
    return aligned, (float(dx), float(dy)), float(response), valid_mask


def _align_sift(
    golden: np.ndarray, 
    test: np.ndarray,
    config: AlignmentConfig,
    border_value: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[float, float], float, np.ndarray]:
    """SIFT feature matching alignment (most accurate)."""
    h, w = golden.shape[:2]
    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    
    # Detect SIFT features
    sift = cv2.SIFT_create(config.sift_max_features)
    kp_golden, desc_golden = sift.detectAndCompute(golden_gray, None)
    kp_test, desc_test = sift.detectAndCompute(test_gray, None)
    
    if desc_golden is None or desc_test is None:
        return test, (0.0, 0.0), 0.0, full_mask
    
    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc_test, desc_golden, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < config.sift_ratio_thresh * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < config.sift_min_matches:
        return test, (0.0, 0.0), 0.1, full_mask
    
    # Calculate response
    avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
    response = 1.0 - min(avg_distance / 300, 1.0)
    
    # Extract points
    pts_test = np.float32([kp_test[m.queryIdx].pt for m in good_matches])
    pts_golden = np.float32([kp_golden[m.trainIdx].pt for m in good_matches])
    
    # Find homography
    H, mask = cv2.findHomography(pts_test, pts_golden, cv2.RANSAC,
                                  config.ransac_reproj_threshold)
    
    if H is None:
        return test, (0.0, 0.0), 0.1, full_mask
    
    dx, dy = H[0, 2], H[1, 2]
    
    aligned = cv2.warpPerspective(test, H, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=border_value)
    
    dummy = np.ones((h, w), dtype=np.uint8) * 255
    valid_mask = cv2.warpPerspective(dummy, H, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderValue=0)
    
    if mask is not None:
        response *= np.sum(mask) / len(mask)
    
    return aligned, (float(dx), float(dy)), float(response), valid_mask


def _align_ecc(
    golden: np.ndarray, 
    test: np.ndarray,
    config: AlignmentConfig
) -> Tuple[np.ndarray, Tuple[float, float], float, np.ndarray]:
    """ECC alignment (sub-pixel precision)."""
    h, w = golden.shape[:2]
    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    
    # Motion type
    if config.ecc_motion_type == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif config.ecc_motion_type == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif config.ecc_motion_type == "affine":
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:  # homography
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                config.ecc_max_iterations, config.ecc_epsilon)
    
    try:
        cc, warp_matrix = cv2.findTransformECC(
            golden_gray.astype(np.float32),
            test_gray.astype(np.float32),
            warp_matrix, warp_mode, criteria
        )
        
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(test, warp_matrix, (w, h))
            valid_mask = cv2.warpPerspective(full_mask, warp_matrix, (w, h), borderValue=0)
            dx, dy = warp_matrix[0, 2], warp_matrix[1, 2]
        else:
            aligned = cv2.warpAffine(test, warp_matrix, (w, h))
            valid_mask = cv2.warpAffine(full_mask, warp_matrix, (w, h), borderValue=0)
            dx, dy = warp_matrix[0, 2], warp_matrix[1, 2]
        
        return aligned, (float(dx), float(dy)), float(cc), valid_mask
        
    except cv2.error:
        return test, (0.0, 0.0), 0.0, full_mask


def align_images_detailed(
    golden_image: np.ndarray,
    test_image: np.ndarray,
    config: Optional[AlignmentConfig] = None
) -> dict:
    """Align images and return detailed information (V2 feature).
    
    Args:
        golden_image: Reference image
        test_image: Test image
        config: Alignment configuration
        
    Returns:
        Dictionary with alignment details
    """
    if config is None:
        config = AlignmentConfig()
    
    aligned, shift, response, valid_mask = align_images(
        golden_image, test_image, config.method, config
    )
    
    return {
        'aligned_image': aligned,
        'shift': shift,
        'response': response,
        'valid_mask': valid_mask,
        'method_used': config.method.value,
        'success': response >= config.phase_min_response
    }
