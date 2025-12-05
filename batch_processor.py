"""Batch inspection workflow processor for PCB panels with multiple strips."""
import os
import time
import csv
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from inspection.io import read_image
from inspection.align_revamp import align_images
from inspection.ssim import calc_ssim
from inspection.pixel_match import run_pixel_matching
from inspection.strip_extractor import extract_strips


class BatchInspector:
    """Manages batch inspection of multiple PCB strips against a master image.
    
    Follows the workflow:
    1. Load master image (golden template without defects)
    2. Extract strips from master image
    3. For each test image:
        - Extract strips from test image
        - Compare each strip against corresponding master strip
        - Aggregate defect results
    4. Generate summary and defect layout
    """
    
    def __init__(self, master_image_path: str, 
                 pixel_thresh: int = 30, 
                 count_thresh: int = 5,
                 ssim_threshold: float = 0.97,
                 num_strips: int = 6):
        """Initialize batch inspector.
        
        Args:
            master_image_path: Path to master/golden image
            pixel_thresh: Pixel difference threshold for anomaly detection
            count_thresh: Anomaly count threshold
            ssim_threshold: SSIM score threshold for quick pass
            num_strips: Expected number of strips per panel
        """
        self.master_image_path = master_image_path
        self.pixel_thresh = pixel_thresh
        self.count_thresh = count_thresh
        self.ssim_threshold = ssim_threshold
        self.num_strips = num_strips
        
        # Load and extract master strips
        self.master_image = read_image(master_image_path)
        self.master_strips = extract_strips(self.master_image, expected_count=num_strips)
        
        if len(self.master_strips) == 0:
            raise ValueError(f"No strips detected in master image: {master_image_path}")
        
        # Results storage
        self.batch_results: List[Dict[str, Any]] = []
        
    def inspect_image(self, test_image_path: str, 
                     progress_callback=None) -> Dict[str, Any]:
        """Inspect a single test image with multiple strips.
        
        Args:
            test_image_path: Path to test image
            progress_callback: Optional callback(strip_num, total_strips, status_msg)
            
        Returns:
            Dictionary containing:
                - test_image_path: str
                - total_strips: int
                - strip_results: List[Dict] - results for each strip
                - defect_count: int - number of defective strips
                - verdict: str - "PASS" or "FAIL"
                - processing_time: float
        """
        start_time = time.time()
        
        # Load test image
        test_image = read_image(test_image_path)
        
        # Extract strips from test image
        test_strips = extract_strips(test_image, expected_count=self.num_strips)
        
        if len(test_strips) == 0:
            return {
                'test_image_path': test_image_path,
                'total_strips': 0,
                'strip_results': [],
                'defect_count': 0,
                'verdict': 'ERROR',
                'error': 'No strips detected',
                'processing_time': time.time() - start_time
            }
        
        # Compare each strip
        strip_results = []
        defect_count = 0
        
        num_strips_to_check = min(len(self.master_strips), len(test_strips))
        
        for i in range(num_strips_to_check):
            if progress_callback:
                progress_callback(i + 1, num_strips_to_check, 
                                f"Inspecting strip {i+1}/{num_strips_to_check}")
            
            result = self._inspect_single_strip(
                self.master_strips[i], 
                test_strips[i], 
                strip_number=i+1
            )
            
            strip_results.append(result)
            
            if result['verdict'] == 'ANOMALY':
                defect_count += 1
        
        # Overall verdict
        overall_verdict = "FAIL" if defect_count > 0 else "PASS"
        
        result_summary = {
            'test_image_path': test_image_path,
            'total_strips': num_strips_to_check,
            'strip_results': strip_results,
            'defect_count': defect_count,
            'verdict': overall_verdict,
            'processing_time': time.time() - start_time
        }
        
        self.batch_results.append(result_summary)
        
        return result_summary
    
    def _inspect_single_strip(self, master_strip: np.ndarray, 
                            test_strip: np.ndarray,
                            strip_number: int) -> Dict[str, Any]:
        """Compare a single test strip against master strip.
        
        Args:
            master_strip: Master/golden strip image
            test_strip: Test strip image
            strip_number: Strip number (1-based)
            
        Returns:
            Dictionary with inspection results for this strip
        """
        start_time = time.time()
        
        # Align the test strip to master
        try:
            aligned_test, (dx, dy), response = align_images(master_strip, test_strip)
            
            if response < 0.1:
                return {
                    'strip_number': strip_number,
                    'verdict': 'ERROR',
                    'error': f'Alignment failed (response: {response:.2f})',
                    'ssim_score': 0.0,
                    'area_score': 0.0,
                    'anomaly_count': 0,
                    'processing_time': time.time() - start_time
                }
        except Exception as e:
            return {
                'strip_number': strip_number,
                'verdict': 'ERROR',
                'error': str(e),
                'ssim_score': 0.0,
                'area_score': 0.0,
                'anomaly_count': 0,
                'processing_time': time.time() - start_time
            }
        
        # Stage 1: SSIM check
        ssim_score, _ = calc_ssim(master_strip, aligned_test)
        
        if ssim_score > self.ssim_threshold:
            return {
                'strip_number': strip_number,
                'verdict': 'NORMAL',
                'ssim_score': float(ssim_score),
                'area_score': 0.0,
                'anomaly_count': 0,
                'method': 'SSIM',
                'processing_time': time.time() - start_time
            }
        
        # Stage 2: Pixel matching
        pixel_res = run_pixel_matching(
            master_strip, 
            aligned_test,
            self.pixel_thresh, 
            self.count_thresh
        )
        
        verdict = 'ANOMALY' if pixel_res['verdict'] == 'Anomaly' else 'NORMAL'
        
        return {
            'strip_number': strip_number,
            'verdict': verdict,
            'ssim_score': float(ssim_score),
            'area_score': float(pixel_res['area_score']),
            'anomaly_count': int(pixel_res['anomaly_count']),
            'method': 'PIXEL_MATCH',
            'processing_time': time.time() - start_time
        }
    
    def inspect_batch(self, test_image_paths: List[str], 
                     progress_callback=None) -> List[Dict[str, Any]]:
        """Inspect multiple test images.
        
        Args:
            test_image_paths: List of paths to test images
            progress_callback: Optional callback(image_num, total_images, status_msg)
            
        Returns:
            List of result dictionaries, one per test image
        """
        results = []
        
        for idx, test_path in enumerate(test_image_paths):
            if progress_callback:
                progress_callback(idx + 1, len(test_image_paths),
                                f"Processing image {idx+1}/{len(test_image_paths)}")
            
            result = self.inspect_image(test_path, progress_callback=None)
            results.append(result)
        
        return results
    
    def export_results_csv(self, output_path: str):
        """Export batch results to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        if not self.batch_results:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Timestamp', 'Image', 'Strip_Number', 'Verdict',
                'SSIM_Score', 'Area_Score', 'Anomaly_Count',
                'Method', 'Processing_Time'
            ])
            
            # Data rows
            for result in self.batch_results:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_name = os.path.basename(result['test_image_path'])
                
                for strip_result in result['strip_results']:
                    writer.writerow([
                        timestamp,
                        image_name,
                        strip_result['strip_number'],
                        strip_result['verdict'],
                        f"{strip_result.get('ssim_score', 0):.4f}",
                        f"{strip_result.get('area_score', 0):.2f}",
                        strip_result.get('anomaly_count', 0),
                        strip_result.get('method', 'N/A'),
                        f"{strip_result['processing_time']:.2f}"
                    ])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of batch processing.
        
        Returns:
            Dictionary with summary stats
        """
        if not self.batch_results:
            return {
                'total_images': 0,
                'total_strips': 0,
                'defective_strips': 0,
                'pass_count': 0,
                'fail_count': 0
            }
        
        total_images = len(self.batch_results)
        total_strips = sum(r['total_strips'] for r in self.batch_results)
        defective_strips = sum(r['defect_count'] for r in self.batch_results)
        pass_count = sum(1 for r in self.batch_results if r['verdict'] == 'PASS')
        fail_count = sum(1 for r in self.batch_results if r['verdict'] == 'FAIL')
        
        return {
            'total_images': total_images,
            'total_strips': total_strips,
            'defective_strips': defective_strips,
            'pass_count': pass_count,
            'fail_count': fail_count,
            'defect_rate': f"{(defective_strips / total_strips * 100) if total_strips > 0 else 0:.2f}%"
        }
