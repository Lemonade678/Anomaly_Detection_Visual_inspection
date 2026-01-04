"""Unified PCB Inspection Package - modular_inspection_integrated

Combines the best features from:
- Modular_inspection_1 (CLAHE, multi-scale, valid_area_mask, grid analyzer)
- modular_inspection2 (light sensitivity, multi-alignment, illumination normalization)

Main modules:
- config: Centralized configuration with dataclasses
- io: Image I/O with RAW support
- align: Multi-method image alignment (ORB/SIFT/ECC/Phase/Auto)
- ssim: Structural similarity check
- pixel_match: Enhanced anomaly detection
- illumination: Light sensitivity and normalization
- edge_detection: Substrate extraction
- grid_analyzer: 9-part grid analysis
- batch_processor: Multi-image workflow
- qr_cropper: QR code extraction
- layout_visualizer: Defect visualization
- image_utils: Utility functions
"""

# Core inspection modules
from .config import (
    AlignmentMethod,
    LightSensitivityMode,
    AlignmentConfig,
    LightSensitivityConfig,
    PixelMatchConfig,
    IlluminationConfig,
    SSIMConfig,
    InspectionConfig,
    get_default_config
)

from .io import read_image
from .ssim import calc_ssim
from .align import align_images, align_images_detailed
from .pixel_match import run_pixel_matching, run_pixel_matching_multiscale
from .illumination import (
    apply_light_sensitivity_mode,
    preprocess_pair,
    match_histograms,
    apply_clahe,
    gamma_correction,
    gold_pad_hsv_filter,
    get_gold_pad_mask
)
from .edge_detection import run_edge_detection
from .grid_analyzer import GridAnalyzer
from .json_config import load_config, get_inspection_parameters
__version__ = "1.0.0"
__all__ = [
    # Config
    'AlignmentMethod', 'LightSensitivityMode', 'AlignmentConfig',
    'LightSensitivityConfig', 'PixelMatchConfig', 'IlluminationConfig',
    'SSIMConfig', 'InspectionConfig', 'get_default_config',
    # Core
    'read_image', 'calc_ssim', 'align_images', 'align_images_detailed',
    'run_pixel_matching', 'run_pixel_matching_multiscale',
    # Illumination
    'apply_light_sensitivity_mode', 'preprocess_pair', 'match_histograms',
    'apply_clahe', 'gamma_correction', 'gold_pad_hsv_filter', 'get_gold_pad_mask',
    # Detection
    'run_edge_detection', 'GridAnalyzer',
    # GUI
    'InspectorApp'
    # JSON Config
    'load_config', 'get_inspection_parameters'
]

# Lazy import for GUI to avoid tkinter dependency when not needed
def __getattr__(name):
    if name == 'InspectorApp':
        from .gui import InspectorApp
        return InspectorApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
