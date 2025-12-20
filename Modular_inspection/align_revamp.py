"""Image alignment using ORB feature matching and homography transformation.

This module provides robust image alignment that can handle:
- Translation
- Rotation  
- Scale changes
- Perspective distortion

Better suited for real-world inspection scenarios compared to phase correlation.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

#add function to equalize the histogram of the image by using CLAHE
def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Equalize the histogram of the image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

# Configuration constants
MAX_NUM_FEATURES = 2000  # Increased for better matching
GOOD_MATCH_PERCENT = 0.15  # Top 15% of matches
MIN_MATCH_COUNT = 10  # Minimum matches required for reliable alignment


def align_images(golden_image: np.ndarray, test_image: np.ndarray, 
                 border_value: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Align test_image to golden_image using ORB feature matching + Homography.
    
    This method is more robust than phase correlation for:
    - Images with rotation
    - Images with perspective distortion
    - Images with scale differences
    
    Args:
        golden_image: Reference/master image (BGR)
        test_image: Image to align (BGR)
        border_value: Color for border pixels after warping
        
    Returns:
        aligned_image: Test image aligned to match golden image
        shift: Approximate (dx, dy) translation component
        response: Match quality score (0.0 to 1.0, higher is better)
        valid_mask: Binary mask (255=valid, 0=border) indicating valid image area
    """
    # Ensure same size - resize test to match golden
    h_golden, w_golden = golden_image.shape[:2]
    test_resized = cv2.resize(test_image, (w_golden, h_golden))
    
    # Convert to grayscale for feature detection
    golden_gray = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)

                   
    #equalization of histogram
    Equalized_golden_gray = equalize_histogram(golden_gray)
    Equalized_test_gray = equalize_histogram(test_gray)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints_golden, descriptors_golden = orb.detectAndCompute(Equalized_golden_gray, None)
    keypoints_test, descriptors_test = orb.detectAndCompute(Equalized_test_gray, None)

    # Handle edge cases where no features are detected
    full_mask = np.ones(test_resized.shape[:2], dtype=np.uint8) * 255
    if descriptors_golden is None or descriptors_test is None:
        return test_resized, (0.0, 0.0), 0.0 ,full_mask
    
    if len(keypoints_golden) < MIN_MATCH_COUNT or len(keypoints_test) < MIN_MATCH_COUNT:
        return test_resized, (0.0, 0.0), 0.1 , full_mask

    # Match features using Brute-Force Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_test, descriptors_golden, None)

    if len(matches) < MIN_MATCH_COUNT:
        return test_resized, (0.0, 0.0), 0.1, full_mask

    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the best matches
    num_good_matches = max(MIN_MATCH_COUNT, int(len(matches) * GOOD_MATCH_PERCENT))
    good_matches = matches[:num_good_matches]

    # Calculate match quality score (response)
    # Based on average match distance normalized to [0, 1]
    avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
    max_hamming_dist = 256  # Maximum Hamming distance for ORB (256-bit descriptors)
    response = 1.0 - (avg_distance / max_hamming_dist)
    response = max(0.0, min(1.0, response))  # Clamp to [0, 1]

    # Extract matched point locations
    points_test = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_golden = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points_test[i, :] = keypoints_test[match.queryIdx].pt
        points_golden[i, :] = keypoints_golden[match.trainIdx].pt

    # Find homography matrix using RANSAC to reject outliers
    h_matrix, mask = cv2.findHomography(points_test, points_golden, cv2.RANSAC, 5.0)

    if h_matrix is None:
        return test_resized, (0.0, 0.0), 0.1, full_mask

    # Extract approximate translation from homography matrix
    # H = [[a, b, tx], [c, d, ty], [e, f, 1]]
    dx = h_matrix[0, 2]
    dy = h_matrix[1, 2]

    # Apply homography to warp test image to align with golden
    height, width = golden_image.shape[:2]
    aligned_image = cv2.warpPerspective(
        test_resized, 
        h_matrix, 
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

        #create a mask of the aligned image
    dummy_mask = np.ones((test_resized.shape[0], test_resized.shape[1]), dtype=np.uint8) * 255

    valid_area_mask = cv2.warpPerspective(
        dummy_mask, 
        h_matrix, 
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    

    # Adjust response based on inlier ratio (how many matches survived RANSAC)
    if mask is not None:
        inlier_ratio = np.sum(mask) / len(mask)
        response = response * inlier_ratio  # Penalize if many outliers

    return aligned_image, (float(dx), float(dy)), float(response), valid_area_mask


def align_images_with_debug(golden_image: np.ndarray, test_image: np.ndarray,
                            save_debug: bool = False, 
                            debug_path: str = "alignment_debug.jpg") -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Align images with optional debug visualization.
    
    Same as align_images but can save a visualization of the matches.
    
    Args:
        golden_image: Reference/master image
        test_image: Image to align
        save_debug: If True, saves match visualization
        debug_path: Path to save debug image
        
    Returns:
        Same as align_images()
    """
    # Resize test to match golden
    h_golden, w_golden = golden_image.shape[:2]
    test_resized = cv2.resize(test_image, (w_golden, h_golden))
    
    # Feature detection
    golden_gray = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints_golden, descriptors_golden = orb.detectAndCompute(golden_gray, None)
    keypoints_test, descriptors_test = orb.detectAndCompute(test_gray, None)
    
    if descriptors_golden is None or descriptors_test is None:
        return test_resized, (0.0, 0.0), 0.0
    
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = sorted(matcher.match(descriptors_test, descriptors_golden), key=lambda x: x.distance)
    
    num_good_matches = max(MIN_MATCH_COUNT, int(len(matches) * GOOD_MATCH_PERCENT))
    good_matches = matches[:num_good_matches]
    
    # Save debug visualization if requested
    if save_debug and len(good_matches) >= MIN_MATCH_COUNT:
        match_img = cv2.drawMatches(
            test_resized, keypoints_test,
            golden_image, keypoints_golden,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(debug_path, match_img)
    
    # Continue with alignment (call main function)

    return align_images(golden_image, test_image)
