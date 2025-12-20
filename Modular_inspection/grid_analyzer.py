"""
Grid-based image analyzer for 9-part defect detection.

Divides images into 3x3 grid and compares corresponding segments
using SSIM and pixel matching for detailed anomaly detection.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from .align_revamp import align_images
from .ssim import calc_ssim
from .pixel_match import run_pixel_matching


class GridAnalyzer:
    """Analyzes images by dividing them into a 3x3 grid."""

    def __init__(self, grid_size: int = 3, pixel_diff_threshold: int = 40, count_threshold: int = 1000):
        """
        Initialize the grid analyzer.

        Args:
            grid_size: Size of the grid (3 for 3x3)
            pixel_diff_threshold: Pixel difference threshold for anomaly detection
            count_threshold: Minimum anomalous pixel count threshold
        """
        self.grid_size = grid_size
        self.pixel_diff_threshold = pixel_diff_threshold
        self.count_threshold = count_threshold
        self.ssim_threshold = 0.975

    def divide_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Divide image into 3x3 grid segments.

        Args:
            image: Input image array

        Returns:
            List of 9 image segments in order (left-to-right, top-to-bottom)
        """
        height, width = image.shape[:2]
        segment_height = height // self.grid_size
        segment_width = width // self.grid_size

        segments = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_start = i * segment_height
                y_end = (i + 1) * segment_height if i < self.grid_size - 1 else height
                x_start = j * segment_width
                x_end = (j + 1) * segment_width if j < self.grid_size - 1 else width

                segment = image[y_start:y_end, x_start:x_end]
                segments.append(segment)

        return segments

    def get_segment_bounds(self, image: np.ndarray, segment_index: int) -> Tuple[int, int, int, int]:
        """
        Get pixel coordinates for a specific segment.

        Args:
            image: Input image array
            segment_index: Index of segment (0-8)

        Returns:
            Tuple of (x_start, y_start, x_end, y_end)
        """
        height, width = image.shape[:2]
        segment_height = height // self.grid_size
        segment_width = width // self.grid_size

        row = segment_index // self.grid_size
        col = segment_index % self.grid_size

        y_start = row * segment_height
        y_end = (row + 1) * segment_height if row < self.grid_size - 1 else height
        x_start = col * segment_width
        x_end = (col + 1) * segment_width if col < self.grid_size - 1 else width

        return (x_start, y_start, x_end, y_end)

    def analyze_segment_pair(self, golden_segment: np.ndarray, test_segment: np.ndarray) -> Dict:
        """
        Analyze a pair of image segments.

        Args:
            golden_segment: Golden/master segment
            test_segment: Test segment to compare

        Returns:
            Dictionary with analysis results
        """
        result = {
            'ssim_score': 0.0,
            'pixel_diff_score': 0.0,
            'anomaly_detected': False,
            'confidence': 0.0,
            'verdict': 'Normal'
        }

        if golden_segment.size == 0 or test_segment.size == 0:
            return result

        try:
            aligned_segment, (dx, dy), confidence = align_images(golden_segment, test_segment)

            if confidence < 0.1:
                result['confidence'] = confidence
                return result

            ssim_score, _ = calc_ssim(golden_segment, aligned_segment)
            result['ssim_score'] = float(ssim_score)

            if ssim_score > self.ssim_threshold:
                result['confidence'] = ssim_score
                return result

            pixel_result = run_pixel_matching(
                golden_segment,
                aligned_segment,
                pixel_thresh=self.pixel_diff_threshold,
                count_thresh=self.count_threshold
            )

            result['pixel_diff_score'] = pixel_result['area_score']
            result['anomaly_detected'] = pixel_result['verdict'] == 'Anomaly'
            result['verdict'] = pixel_result['verdict']
            result['confidence'] = pixel_result['area_score'] / 100.0

        except Exception as e:
            result['error'] = str(e)

        return result

    def analyze_images(self, golden_image: np.ndarray, test_image: np.ndarray) -> Dict:
        """
        Perform full 9-part grid analysis on image pair.

        Args:
            golden_image: Master/golden image
            test_image: Test image to inspect

        Returns:
            Dictionary with results for all 9 segments and overall verdict
        """
        golden_segments = self.divide_image(golden_image)
        test_segments = self.divide_image(test_image)

        if len(golden_segments) != len(test_segments):
            return {
                'error': 'Segment count mismatch',
                'verdict': 'Error'
            }

        segment_results = []
        anomaly_count = 0
        total_confidence = 0.0

        for idx in range(9):
            segment_result = self.analyze_segment_pair(golden_segments[idx], test_segments[idx])
            segment_result['segment_index'] = idx
            segment_results.append(segment_result)

            if segment_result['anomaly_detected']:
                anomaly_count += 1

            total_confidence += segment_result['confidence']

        overall_defect_score = (anomaly_count / 9) * 100
        average_confidence = total_confidence / 9

        final_verdict = 'Anomaly' if anomaly_count >= 2 else 'Normal'

        return {
            'segments': segment_results,
            'anomaly_count': anomaly_count,
            'overall_defect_score': overall_defect_score,
            'average_confidence': average_confidence,
            'verdict': final_verdict,
            'defect_locations': [i for i, s in enumerate(segment_results) if s['anomaly_detected']]
        }

    def visualize_results(self, image: np.ndarray, analysis_results: Dict) -> np.ndarray:
        """
        Create a visualization of analysis results on the image.

        Args:
            image: Original image
            analysis_results: Results from analyze_images()

        Returns:
            Annotated image with grid and defect highlights
        """
        height, width = image.shape[:2]
        segment_height = height // self.grid_size
        segment_width = width // self.grid_size

        viz_image = image.copy()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x_start = j * segment_width
                y_start = i * segment_height
                x_end = (j + 1) * segment_width
                y_end = (i + 1) * segment_height

                segment_idx = i * self.grid_size + j

                if segment_idx < len(analysis_results.get('segments', [])):
                    segment = analysis_results['segments'][segment_idx]
                    if segment['anomaly_detected']:
                        cv2.rectangle(viz_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)
                    else:
                        cv2.rectangle(viz_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                else:
                    cv2.rectangle(viz_image, (x_start, y_start), (x_end, y_end), (128, 128, 128), 1)

        return viz_image
