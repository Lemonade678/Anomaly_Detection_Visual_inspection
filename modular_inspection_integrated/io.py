"""Image I/O helpers.

Contains read_image(path) which tries rawpy, OpenCV and PIL fallbacks
and returns a BGR np.ndarray.
"""
import os
import cv2
import numpy as np
from PIL import Image

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf", ".sr2", ".pef"}


def read_image(path: str) -> np.ndarray:
    """Read an image from disk and return a BGR uint8 numpy array.

    Tries rawpy for RAW formats first, then OpenCV, then PIL as a final fallback.
    Raises RuntimeError if nothing succeeds.
    
    Args:
        path: Path to image file
        
    Returns:
        BGR uint8 numpy array
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in RAW_EXTS:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_auto_wb=True,
                    no_auto_bright=False,
                    gamma=(2.2, 4.5),
                    output_bps=8,
                )
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError as e:
            raise RuntimeError(
                "Detected RAW file but 'rawpy' is not installed. "
                "Install with: pip install rawpy"
            ) from e
        except Exception:
            pass  # Fall through to other readers

    # Try OpenCV
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # PIL fallback
    try:
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path} with available readers: {e}") from e


def save_image(image: np.ndarray, path: str) -> bool:
    """Save an image to disk.
    
    Args:
        image: BGR numpy array
        path: Output path
        
    Returns:
        True if successful
    """
    try:
        cv2.imwrite(path, image)
        return True
    except Exception:
        return False
