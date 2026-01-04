"""Centralized configuration for the integrated PCB inspection system.

Combines dataclass configs from modular_inspection2 with additions for V1 features.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import json
import os


class AlignmentMethod(Enum):
    """Available image alignment methods."""
    PHASE_CORRELATION = "phase"      # Fast, translation-only
    ORB_HOMOGRAPHY = "orb"           # Rotation + scale, moderate speed
    SIFT_HOMOGRAPHY = "sift"         # Most accurate, slowest
    ECC = "ecc"                      # Sub-pixel precision
    AUTO = "auto"                    # Try methods in fallback order


class LightSensitivityMode(Enum):
    """Light sensitivity processing modes."""
    AUTO = "auto"           # Automatic adaptive selection
    LOW_LIGHT = "low"       # Enhanced for dark images
    HIGH_LIGHT = "high"     # For overexposed images  
    HDR = "hdr"             # High dynamic range processing
    STANDARD = "standard"   # Default balanced processing
    GOLD_PAD = "gold_pad"   # HSV filter for gold bonding pads


class InspectionMode(Enum):
    """Inspection comparison method."""
    PIXEL_WISE = "pixel"      # Traditional pixel-by-pixel comparison
    TEMPLATE_MATCH = "template"  # Template matching for camera position variance


@dataclass
class LightSensitivityConfig:
    """Configuration for light sensitivity processing."""
    
    mode: LightSensitivityMode = LightSensitivityMode.AUTO
    gamma: float = 1.0  # 1.0 = no change, <1 = darken, >1 = brighten
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # Low-light specific
    low_light_gamma: float = 1.5
    low_light_clahe_clip: float = 4.0
    
    # High-light specific  
    high_light_gamma: float = 0.7
    highlight_threshold: int = 240
    
    # HDR parameters
    hdr_sigma_spatial: float = 10.0
    hdr_sigma_range: float = 0.1
    
    # Gold Pad HSV filter parameters (for gold bonding pads)
    # HSV ranges for gold color detection
    gold_hue_low: int = 15          # Lower hue bound (gold/yellow starts around 15)
    gold_hue_high: int = 40         # Upper hue bound (gold/yellow ends around 40)
    gold_saturation_low: int = 50   # Minimum saturation to detect gold
    gold_saturation_high: int = 255 # Maximum saturation
    gold_value_low: int = 80        # Minimum brightness
    gold_value_high: int = 255      # Maximum brightness
    gold_enhance_contrast: bool = True  # Apply CLAHE to enhance gold regions


@dataclass
class AlignmentConfig:
    """Configuration for image alignment."""
    
    method: AlignmentMethod = AlignmentMethod.AUTO
    
    # Fallback order for AUTO mode
    fallback_order: List[AlignmentMethod] = field(default_factory=lambda: [
        AlignmentMethod.PHASE_CORRELATION,
        AlignmentMethod.ORB_HOMOGRAPHY,
        AlignmentMethod.SIFT_HOMOGRAPHY,
        AlignmentMethod.ECC
    ])
    
    # Phase Correlation
    phase_min_response: float = 0.1
    
    # ORB Feature Matching
    orb_max_features: int = 5000
    orb_good_match_percent: float = 0.15
    orb_min_matches: int = 10
    
    # SIFT Feature Matching
    sift_max_features: int = 5000
    sift_ratio_thresh: float = 0.75
    sift_min_matches: int = 10
    
    # ECC
    ecc_max_iterations: int = 5000
    ecc_epsilon: float = 1e-10
    ecc_motion_type: str = "euclidean"
    
    # RANSAC
    ransac_reproj_threshold: float = 5.0


@dataclass
class PixelMatchConfig:
    """Configuration for pixel-based anomaly detection."""
    
    pixel_threshold: int = 40           # Difference threshold (0-255)
    count_threshold: int = 5000         # Anomalous pixel count threshold
    area_threshold: float = 20.0        # Area percentage threshold
    min_contour_area: int = 50          # Minimum contour area
    max_contour_area: Optional[int] = None
    kernel_size: Tuple[int, int] = (5, 5)
    dilation_iterations: int = 2
    
    # V1 enhancements
    use_adaptive_threshold: bool = False
    use_histogram_equalization: bool = True
    multi_scale_enabled: bool = False
    multi_scale_kernels: List[Tuple[int, int]] = field(
        default_factory=lambda: [(3, 3), (5, 5), (9, 9)]
    )


@dataclass
class IlluminationConfig:
    """Configuration for illumination normalization."""
    
    enabled: bool = True
    method: str = "match_histogram"  # match_histogram, clahe_both, normalize_both, none
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    target_mean: float = 128.0
    target_std: float = 64.0


@dataclass
class SSIMConfig:
    """Configuration for SSIM pre-check."""
    
    pass_threshold: float = 0.975
    enabled: bool = True


@dataclass
class EdgeDetectionConfig:
    """Configuration for edge-based cropping."""
    
    canny_low: int = 40
    canny_high: int = 125
    min_width: int = 80
    max_width: int = 240
    min_height: int = 200
    max_height: int = 800
    blur_kernel_size: Tuple[int, int] = (5, 5)


@dataclass
class GridAnalyzerConfig:
    """Configuration for 9-part grid analysis (from V1)."""
    
    grid_size: int = 3
    pixel_diff_threshold: int = 40
    count_threshold: int = 1000
    ssim_threshold: float = 0.975
    min_anomaly_segments: int = 2  # Segments with anomaly to flag whole image


@dataclass
class InspectionConfig:
    """Master configuration combining all sub-configs."""
    
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    pixel_match: PixelMatchConfig = field(default_factory=PixelMatchConfig)
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
    ssim: SSIMConfig = field(default_factory=SSIMConfig)
    edge_detection: EdgeDetectionConfig = field(default_factory=EdgeDetectionConfig)
    light_sensitivity: LightSensitivityConfig = field(default_factory=LightSensitivityConfig)
    grid_analyzer: GridAnalyzerConfig = field(default_factory=GridAnalyzerConfig)


# Global default configuration
DEFAULT_CONFIG = InspectionConfig()


def get_default_config() -> InspectionConfig:
    """Get a copy of the default configuration."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def load_substrate_config(json_path: str) -> dict:
    """Load substrate-specific config from JSON file (V1 feature).
    
    Args:
        json_path: Path to ConfigtypeX.json file
        
    Returns:
        Dictionary with substrate configuration
    """
    if not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config
    except (json.JSONDecodeError, IOError):
        return {}


def get_inspection_params_from_substrate(substrate_config: dict) -> dict:
    """Extract inspection parameters from substrate config.
    
    Args:
        substrate_config: Dict from load_substrate_config()
        
    Returns:
        Dict with pixel_threshold, golden_sample_path, etc.
    """
    params = substrate_config.get('inspection_parameters', {})
    return {
        'pixel_threshold': params.get('defect_threshold_area', 40),
        'alignment_threshold': params.get('alignment_threshold', 0.8),
        'golden_sample_path': params.get('golden_sample_path', ''),
        'roi_mask_path': params.get('roi_mask_path', ''),
        'pixel_ratio_calibration': params.get('pixel_ratio_calibration', 1.0)
    }


__all__ = [
    'AlignmentMethod', 'LightSensitivityMode', 'InspectionMode',
    'AlignmentConfig', 'LightSensitivityConfig', 'PixelMatchConfig',
    'IlluminationConfig', 'SSIMConfig', 'EdgeDetectionConfig',
    'GridAnalyzerConfig', 'InspectionConfig',
    'DEFAULT_CONFIG', 'get_default_config',
    'load_substrate_config', 'get_inspection_params_from_substrate'
]
