"""JSON Configuration Manager for Substrate Types.

Provides easy loading, saving, and management of substrate-specific configuration
files (Configtype1.json, Configtype2.json, etc.)

This is the integrated version combining jsonConfig.py from V1.
"""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path


# Default configs directory (relative to this file)
CONFIGS_DIR = Path(__file__).parent / "Configs"


def load_config(file_path: str) -> Dict:
    """Load configuration from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with configuration data
    """
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config


def save_config(config: Dict, file_path: str) -> None:
    """Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save JSON file
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)


def update_config(config: Dict, key: str, value) -> Dict:
    """Update a configuration value.
    
    Args:
        config: Configuration dictionary
        key: Key to update
        value: New value
        
    Returns:
        Updated configuration
    """
    config[key] = value
    return config


def get_available_config_types() -> List[str]:
    """Get list of available substrate config types.
    
    Returns:
        List of config type numbers (e.g., ['1', '2', '3', '4', '5'])
    """
    types = []
    
    if not CONFIGS_DIR.exists():
        return types
    
    for file in CONFIGS_DIR.glob("*.json"):
        name = file.stem.lower()
        if name.startswith("configtype"):
            type_num = name.replace("configtype", "")
            types.append(type_num)
    
    return sorted(types)


def load_substrate_config(substrate_type: str) -> Dict:
    """Load configuration for a specific substrate type.
    
    Args:
        substrate_type: Type number (e.g., '1', '2', '3') or full path
        
    Returns:
        Configuration dictionary
    """
    # Check if it's a path
    if os.path.exists(substrate_type):
        return load_config(substrate_type)
    
    # Try to find config file
    patterns = [
        f"Configtype{substrate_type}.json",
        f"configtype{substrate_type}.json",
        f"ConfigType{substrate_type}.json"
    ]
    
    for pattern in patterns:
        config_path = CONFIGS_DIR / pattern
        if config_path.exists():
            return load_config(str(config_path))
    
    return {}


def get_inspection_parameters(substrate_type: str) -> Dict:
    """Get inspection parameters for a substrate type.
    
    Args:
        substrate_type: Type number or config path
        
    Returns:
        Dictionary with inspection parameters:
        - pixel_threshold
        - alignment_threshold
        - golden_sample_path
        - roi_mask_path
        - pixel_ratio_calibration
    """
    config = load_substrate_config(substrate_type)
    params = config.get('inspection_parameters', {})
    
    return {
        'pixel_threshold': params.get('defect_threshold_area', 40),
        'alignment_threshold': params.get('alignment_threshold', 0.8),
        'golden_sample_path': params.get('golden_sample_path', ''),
        'roi_mask_path': params.get('roi_mask_path', ''),
        'pixel_ratio_calibration': params.get('pixel_ratio_calibration', 1.0)
    }


def get_substrate_dimensions(substrate_type: str) -> Dict:
    """Get physical dimensions for a substrate type.
    
    Args:
        substrate_type: Type number or config path
        
    Returns:
        Dictionary with dimension data
    """
    config = load_substrate_config(substrate_type)
    
    overall = config.get('overall substrate dimension', {})
    piece = config.get('physical_dimensions(1pcs)', {})
    bonding = config.get('bonding_pad_specs(1pcs)', {})
    
    return {
        'substrate_type': config.get('substrate_type', ''),
        'part_number': config.get('part_number', ''),
        'overall_width_mm': (
            overall.get('width_mm_lower_bound', 0),
            overall.get('width_mm_upper_bound', 0)
        ),
        'overall_length_mm': (
            overall.get('length_mm_lower_bound', 0),
            overall.get('length_mm_upper_bound', 0)
        ),
        'thickness_mm': (
            overall.get('thickness_mm_lower_bound', 0),
            overall.get('thickness_mm_upper_bound', 0)
        ),
        'piece_width_mm': piece.get('width_mm', 0),
        'piece_length_mm': piece.get('length_mm', 0),
        'bonding_pad_width_um': bonding.get('width_um', 0),
        'bonding_pad_length_um': bonding.get('length_um', 0),
        'bonding_pad_tolerance_um': bonding.get('tolerance_um', 0)
    }


def list_all_configs() -> List[Dict]:
    """List all available substrate configurations.
    
    Returns:
        List of config summaries
    """
    configs = []
    
    for type_num in get_available_config_types():
        config = load_substrate_config(type_num)
        configs.append({
            'type': type_num,
            'substrate_type': config.get('substrate_type', f'Type {type_num}'),
            'part_number': config.get('part_number', 'Unknown')
        })
    
    return configs


def create_new_config(substrate_type: str, part_number: str, 
                      width_mm: float = 17.0, length_mm: float = 17.0) -> Dict:
    """Create a new substrate configuration template.
    
    Args:
        substrate_type: Type name (e.g., "Type 6")
        part_number: Part number
        width_mm: Piece width in mm
        length_mm: Piece length in mm
        
    Returns:
        New configuration dictionary
    """
    return {
        "substrate_type": substrate_type,
        "part_number": part_number,
        "overall substrate dimension": {
            "width_mm_lower_bound": 42.89,
            "width_mm_upper_bound": 43.11,
            "length_mm_lower_bound": 170.02,
            "length_mm_upper_bound": 170.28,
            "thickness_mm_upper_bound": 0.64,
            "thickness_mm_lower_bound": 0.58,
            "__remark__comment": "Substrate sheet dimensions"
        },
        "physical_dimensions(1pcs)": {
            "width_mm": width_mm,
            "length_mm": length_mm
        },
        "bonding_pad_specs(1pcs)": {
            "width_um": 85,
            "length_um": 200,
            "tolerance_um": 5
        },
        "inspection_parameters": {
            "pixel_ratio_calibration": 1.0,
            "alignment_threshold": 0.8,
            "defect_threshold_area": 10,
            "golden_sample_path": f"assets/golden_samples/{substrate_type.lower().replace(' ', '')}_golden.png",
            "roi_mask_path": f"assets/masks/{substrate_type.lower().replace(' ', '')}_mask.png"
        }
    }


# Convenience exports
__all__ = [
    'load_config', 'save_config', 'update_config',
    'get_available_config_types', 'load_substrate_config',
    'get_inspection_parameters', 'get_substrate_dimensions',
    'list_all_configs', 'create_new_config',
    'CONFIGS_DIR'
]


if __name__ == '__main__':
    # Demo
    print("Available substrate configurations:")
    for config in list_all_configs():
        print(f"  Type {config['type']}: {config['substrate_type']} ({config['part_number']})")
    
    print("\nType 2 inspection parameters:")
    params = get_inspection_parameters('1')
    for key, value in params.items():
        print(f"  {key}: {value}")