"""UI Theme and Styling Configuration.

Defines the color palette and font styles for the modern dark interface.
"""
from dataclasses import dataclass

@dataclass
class Theme:
    """Application Theme Colors (Cyberpunk/Dark Modern)."""
    
    # Backgrounds
    BG_MAIN: str = "#121212"      # Deep background
    BG_PANEL: str = "#1E1E1E"     # Panel background
    BG_CANVAS: str = "#000000"    # Canvas background
    
    # Foregrounds / Text
    FG_PRIMARY: str = "#E0E0E0"   # Main text
    FG_SECONDARY: str = "#A0A0A0" # Subtitles / labels
    FG_DISABLED: str = "#555555"  # Disabled text
    
    # Accents
    ACCENT_PRIMARY: str = "#00FF41"   # Matrix Green (Main Action)
    ACCENT_HOVER: str = "#00CC33"     # Hover state
    ACCENT_WARNING: str = "#FFB000"   # Warnings
    ACCENT_ERROR: str = "#FF4444"     # Errors / Defects
    ACCENT_INFO: str = "#00FFFF"      # Cyan info
    
    # Borders
    BORDER_MAIN: str = "#333333"
    BORDER_FOCUS: str = "#555555"
    
    # Fonts
    FONT_MAIN: tuple = ("Segoe UI", 10)
    FONT_BOLD: tuple = ("Segoe UI", 10, "bold")
    FONT_HEADER: tuple = ("Segoe UI", 14, "bold")
    FONT_MONO: tuple = ("Consolas", 10)

# Global theme instance
DARK_THEME = Theme()
