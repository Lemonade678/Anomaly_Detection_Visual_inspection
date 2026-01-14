"""QR Code Detection, Reading, and Cropping Module.

Features:
- Detect QR codes in images (including PCB images)
- Read/decode QR code content
- Crop detected QR codes
- Save cropped QR codes to folder
- Export results as JSON
"""
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class QRCodeExtractor:
    """Extracts and crops QR codes from images."""
    
    def __init__(self, output_dir: str = "qrcode_extraction"):
        """Initialize QR code extractor.
        
        Args:
            output_dir: Directory to save extracted QR codes
        """
        self.output_dir = output_dir
        self.detector = cv2.QRCodeDetector()
        self.last_results = []
        self.last_json_path = None
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _preprocess_for_pcb(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate multiple preprocessed versions to improve detection.
        
        PCB images often have white QR codes on green background.
        This creates multiple versions to maximize detection chances.
        
        Args:
            image: BGR image
            
        Returns:
            List of preprocessed grayscale images
        """
        versions = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Version 1: Original grayscale
        versions.append(gray)
        
        # Version 2: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        versions.append(enhanced)
        
        # Version 3: Otsu threshold (binary)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        versions.append(binary)
        
        # Version 4: Inverted binary (for white-on-dark QR codes)
        versions.append(cv2.bitwise_not(binary))
        
        # Version 5: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        versions.append(adaptive)
        
        # Version 6: Green channel extraction (good for white-on-green PCB)
        if len(image.shape) == 3:
            # Extract green channel and enhance contrast
            green = image[:, :, 1]
            green_enhanced = clahe.apply(green)
            _, green_binary = cv2.threshold(green_enhanced, 0, 255, 
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            versions.append(green_binary)
        
        # Version 7: High contrast stretch
        min_val, max_val = gray.min(), gray.max()
        if max_val > min_val:
            stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            versions.append(stretched)
        
        return versions
    
    def detect_and_decode(self, image: np.ndarray) -> List[Dict]:
        """Detect and decode all QR codes in an image.
        
        Uses pyzbar as primary method (more reliable), OpenCV as fallback.
        Tries multiple preprocessing versions to maximize detection.
        
        Args:
            image: BGR image array
            
        Returns:
            List of dicts with 'data', 'bbox', 'center' for each QR code
        """
        self.last_results = []
        
        if image is None or image.size == 0:
            return []
        
        # Generate preprocessed versions
        preprocessed_versions = self._preprocess_for_pcb(image)
        
        results = []
        found_data = set()  # Track unique QR codes by data
        
        # Try pyzbar on all versions
        try:
            from pyzbar import pyzbar
            from pyzbar.pyzbar import ZBarSymbol
            
            for version in preprocessed_versions:
                # Try QR code specific detection
                decoded = pyzbar.decode(version, symbols=[ZBarSymbol.QRCODE])
                
                # Also try general detection
                if not decoded:
                    decoded = pyzbar.decode(version)
                
                for obj in decoded:
                    # Skip if already found this QR code
                    data_str = obj.data.decode('utf-8') if obj.data else ""
                    if data_str in found_data:
                        continue
                    found_data.add(data_str)
                    
                    x, y, w, h = obj.rect
                    
                    if obj.polygon:
                        points = [(p.x, p.y) for p in obj.polygon]
                    else:
                        points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                    
                    results.append({
                        'id': len(results) + 1,
                        'data': data_str if data_str else "[Unable to decode]",
                        'bbox': (x, y, w, h),
                        'points': points,
                        'center': (x + w // 2, y + h // 2),
                        'type': obj.type,
                        'method': 'pyzbar'
                    })
            
            if results:
                self.last_results = results
                return results
                
        except ImportError:
            print("Note: pyzbar not installed. Install with: pip install pyzbar")
            print("Falling back to OpenCV QR detector...")
        except Exception as e:
            print(f"pyzbar error: {e}, trying OpenCV...")
        
        # Fallback: OpenCV on all versions
        for version in preprocessed_versions:
            try:
                retval, decoded_info, points, _ = self.detector.detectAndDecodeMulti(version)
                if retval and points is not None:
                    for i, (data, pts) in enumerate(zip(decoded_info, points)):
                        # Skip duplicates
                        if data and data in found_data:
                            continue
                        if data:
                            found_data.add(data)
                        
                        if pts is not None and len(pts) >= 4:
                            pts = pts.astype(int)
                            x_coords = pts[:, 0]
                            y_coords = pts[:, 1]
                            
                            bbox = (
                                int(min(x_coords)),
                                int(min(y_coords)),
                                int(max(x_coords) - min(x_coords)),
                                int(max(y_coords) - min(y_coords))
                            )
                            
                            results.append({
                                'id': len(results) + 1,
                                'data': data if data else "[Unable to decode]",
                                'bbox': bbox,
                                'points': pts.tolist(),
                                'center': (int(np.mean(x_coords)), int(np.mean(y_coords))),
                                'type': 'QRCODE',
                                'method': 'opencv'
                            })
            except Exception:
                pass
            
            if results:
                break
        
        # Single QR detection fallback
        if not results:
            for version in preprocessed_versions:
                try:
                    data, points, _ = self.detector.detectAndDecode(version)
                    if points is not None and len(points) > 0:
                        pts = points[0].astype(int) if len(points.shape) == 3 else points.astype(int)
                        x_coords = pts[:, 0]
                        y_coords = pts[:, 1]
                        
                        bbox = (
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords) - min(x_coords)),
                            int(max(y_coords) - min(y_coords))
                        )
                        
                        results.append({
                            'id': 1,
                            'data': data if data else "[Unable to decode]",
                            'bbox': bbox,
                            'points': pts.tolist(),
                            'center': (int(np.mean(x_coords)), int(np.mean(y_coords))),
                            'type': 'QRCODE',
                            'method': 'opencv_single'
                        })
                        break
                except Exception:
                    pass
        
        self.last_results = results
        return results
    
    def crop_qr_codes(self, image: np.ndarray, padding: int = 10) -> List[Tuple[np.ndarray, Dict]]:
        """Crop all detected QR codes from image."""
        results = self.detect_and_decode(image)
        crops = []
        
        h, w = image.shape[:2]
        
        for qr_info in results:
            x, y, qw, qh = qr_info['bbox']
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + qw + padding)
            y2 = min(h, y + qh + padding)
            
            crop = image[y1:y2, x1:x2].copy()
            crops.append((crop, qr_info))
        
        return crops
    
    def save_cropped_qr(self, image: np.ndarray, prefix: str = "qr", 
                        save_json: bool = True) -> List[str]:
        """Detect, crop, and save all QR codes from image.
        
        Args:
            image: BGR image array
            prefix: Filename prefix for saved crops
            save_json: Whether to save results as JSON
            
        Returns:
            List of saved file paths
        """
        crops = self.crop_qr_codes(image)
        saved_paths = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build JSON output
        json_output = {
            'timestamp': timestamp,
            'total_qr_codes': len(crops),
            'qr_codes': []
        }
        
        for crop_img, qr_info in crops:
            qr_id = qr_info['id']
            # Clean data for filename
            data_clean = ''.join(c if c.isalnum() else '_' for c in qr_info['data'][:20])
            
            filename = f"{prefix}_{timestamp}_{qr_id}_{data_clean}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            cv2.imwrite(filepath, crop_img)
            saved_paths.append(filepath)
            
            # Add to JSON
            json_output['qr_codes'].append({
                'id': qr_info['id'],
                'data': qr_info['data'],
                'type': qr_info.get('type', 'QRCODE'),
                'bbox': {
                    'x': qr_info['bbox'][0],
                    'y': qr_info['bbox'][1],
                    'width': qr_info['bbox'][2],
                    'height': qr_info['bbox'][3]
                },
                'center': {
                    'x': qr_info['center'][0],
                    'y': qr_info['center'][1]
                },
                'method': qr_info['method'],
                'image_file': filename
            })
        
        # Save JSON file
        if save_json and crops:
            json_filename = f"{prefix}_{timestamp}_results.json"
            json_path = os.path.join(self.output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            self.last_json_path = json_path
            saved_paths.append(json_path)
        
        return saved_paths
    
    def get_results_as_json(self) -> str:
        """Get last detection results as JSON string."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_qr_codes': len(self.last_results),
            'qr_codes': []
        }
        
        for qr in self.last_results:
            output['qr_codes'].append({
                'id': qr['id'],
                'data': qr['data'],
                'type': qr.get('type', 'QRCODE'),
                'bbox': {
                    'x': qr['bbox'][0],
                    'y': qr['bbox'][1],
                    'width': qr['bbox'][2],
                    'height': qr['bbox'][3]
                },
                'center': {
                    'x': qr['center'][0],
                    'y': qr['center'][1]
                },
                'method': qr['method']
            })
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def annotate_image(self, image: np.ndarray, results: List[Dict] = None) -> np.ndarray:
        """Draw annotations on image showing detected QR codes."""
        if results is None:
            results = self.last_results
        
        annotated = image.copy()
        
        for qr_info in results:
            x, y, w, h = qr_info['bbox']
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            cx, cy = qr_info['center']
            cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)
            
            label = f"QR #{qr_info['id']}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            data_preview = qr_info['data'][:30] + "..." if len(qr_info['data']) > 30 else qr_info['data']
            cv2.putText(annotated, data_preview, (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_summary_text(self) -> str:
        """Get text summary of last detection results."""
        if not self.last_results:
            return "No QR codes detected."
        
        lines = [f"Found {len(self.last_results)} QR code(s):"]
        for qr in self.last_results:
            lines.append(f"  #{qr['id']}: {qr['data'][:50]}...")
        return "\n".join(lines)


# Convenience functions
def detect_qr_codes(image: np.ndarray) -> List[Dict]:
    """Detect QR codes in image."""
    extractor = QRCodeExtractor()
    return extractor.detect_and_decode(image)


def extract_and_save_qr_codes(image: np.ndarray, output_dir: str = "qrcode_extraction") -> List[str]:
    """Extract and save QR codes with JSON results."""
    extractor = QRCodeExtractor(output_dir)
    return extractor.save_cropped_qr(image, save_json=True)


def get_qr_json(image: np.ndarray) -> str:
    """Get QR detection results as JSON string."""
    extractor = QRCodeExtractor()
    extractor.detect_and_decode(image)
    return extractor.get_results_as_json()

