"""Integrated PCB Inspection GUI.

NOTE: For the main entry point, use hybrid_integrated.py instead of this file.
      Run: python hybrid_integrated.py

Full-featured GUI combining:
- V1 gui_1.py: Clean inspector interface
- V2 gui_2.py + hybrid_2.py: Advanced controls (light sensitivity, alignment methods)

This module provides the InspectorApp class for the GUI interface.
It is imported and used by hybrid_integrated.py.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import sys
import time
import csv
import shutil
import json
from PIL import Image, ImageTk

# Import from integrated package
from .io import read_image
from .align import align_images, AlignmentMethod
from .ssim import calc_ssim
from .pixel_match import run_pixel_matching
from .edge_detection import run_edge_detection
from .illumination import apply_light_sensitivity_mode, preprocess_pair, gold_pad_hsv_filter
from .config import (
    LightSensitivityMode, LightSensitivityConfig,
    AlignmentConfig, get_default_config, InspectionMode
)
from .template_match import run_template_inspection, TemplateMatchConfig


# ==============================================================================
# ANOMALY LOCATION MAPPER
# ==============================================================================


# ==============================================================================
# TOOLTIP CLASS
# ==============================================================================

class ToolTip(object):
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     # miliseconds
        self.wraplength = 180   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffdd", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

class AnomalyLocationMapper:
    """Maps and reports anomaly locations with pixel coordinates."""
    
    def __init__(self):
        self.anomaly_regions = []
        self.centroids = []
    
    def analyze_mask(self, anomaly_mask: np.ndarray, min_area: int = 50) -> dict:
        """Analyze anomaly mask to extract location information."""
        self.anomaly_regions = []
        self.centroids = []
        
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = anomaly_mask.shape
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw // 2, y + bh // 2
            
            sector_x = "Left" if cx < w/3 else ("Right" if cx > 2*w/3 else "Center")
            sector_y = "Top" if cy < h/3 else ("Bottom" if cy > 2*h/3 else "Middle")
            
            region_info = {
                'id': len(self.anomaly_regions) + 1,
                'bbox': (x, y, bw, bh),
                'centroid': (cx, cy),
                'area_px': int(area),
                'sector': f"{sector_y}-{sector_x}",
                'relative_pos': (round(cx/w * 100, 1), round(cy/h * 100, 1))
            }
            
            self.anomaly_regions.append(region_info)
            self.centroids.append((cx, cy))
        
        return {
            'regions': self.anomaly_regions,
            'total_regions': len(self.anomaly_regions),
            'total_anomalous_pixels': np.count_nonzero(anomaly_mask),
            'image_size': (w, h)
        }
    
    def create_annotated_image(self, base_image: np.ndarray) -> np.ndarray:
        """Create an annotated image with anomaly locations."""
        annotated = base_image.copy()
        h, w = annotated.shape[:2]
        
        # Draw grid
        grid_color = (40, 40, 40)
        cv2.line(annotated, (w//3, 0), (w//3, h), grid_color, 1)
        cv2.line(annotated, (2*w//3, 0), (2*w//3, h), grid_color, 1)
        cv2.line(annotated, (0, h//3), (w, h//3), grid_color, 1)
        cv2.line(annotated, (0, 2*h//3), (w, 2*h//3), grid_color, 1)
        
        for region in self.anomaly_regions:
            x, y, bw, bh = region['bbox']
            cx, cy = region['centroid']
            region_id = region['id']
            
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
            
            label = f"#{region_id}"
            cv2.putText(annotated, label, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated
    
    def get_summary_text(self) -> str:
        """Get formatted summary of anomaly locations."""
        if not self.anomaly_regions:
            return "No anomalies detected."
        
        lines = [f"Found {len(self.anomaly_regions)} anomaly region(s):"]
        for region in self.anomaly_regions:
            lines.append(f"  #{region['id']}: {region['sector']} @ {region['centroid']}")
        return "\n".join(lines)


# ==============================================================================
# MAIN GUI APPLICATION
# ==============================================================================

class PixelInspectionWindow(tk.Toplevel):
    """Integrated PCB Inspector GUI Application."""
    
    # Theme colors
    BG_COLOR = "#000000"
    FG_COLOR = "#00FF41"
    ACCENT_COLOR = "#00FF41"
    ERROR_COLOR = "#FF0000"
    FONT_FACE = "Consolas"
    
    SSIM_PASS_THRESHOLD = 0.975
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("pixel_inspection_system(hiatus)")
        self.geometry("1600x1000")
        self.state('zoomed')  # Start maximized
        self.configure(bg=self.BG_COLOR)
        
        # State variables
        self.golden_image = None
        self.test_image = None
        self.aligned_image = None
        self.ssim_score = 0.0
        self._golden_loaded = False
        self._test_loaded = False
        
        # Last inspection results (for saving)
        self.last_result = None
        self.last_heatmap = None
        self.last_contour_map = None
        self.last_ssim_heatmap = None
        self.last_golden_path = None
        self.last_test_path = None
        
        # Settings
        self.selected_alignment_method = AlignmentMethod.AUTO
        self.selected_light_mode = LightSensitivityMode.AUTO
        self.selected_inspection_mode = InspectionMode.PIXEL_WISE  # New: inspection mode
        self.selected_illum_method = "match_histogram"
        self.gamma_value = 1.0
        self.anomaly_mapper = AnomalyLocationMapper()
        
        # Canvas metadata for coordinate tracking: {canvas: (ratio, off_x, off_y, orig_w, orig_h)}
        self.canvas_meta = {}
        
        # Template matching settings
        self.template_grid_cols = 4
        self.template_grid_rows = 4
        self.template_search_margin = 50
        
        # Window management - track all open tool windows
        self.tool_windows = {}  # {window_name: window_instance}
        self.session_file = "inspector_session.json"
        
        # Log file
        self.log_file = "inspection_log.csv"
        self._init_log_file()
        
        # Build UI
        self._setup_styles()
        self._build_menu()
        self._build_ui()
        
        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # self.after(500, self._load_session) # Removed

    
    def _on_close(self):
        """Handle window closure."""
        try:
            self.display_canvas.unbind_all("<MouseWheel>")
        except:
            pass
        self.destroy()
    
    def _go_back_to_main(self):
        """Close this window and go back to main InspectorApp."""
        try:
            self.display_canvas.unbind_all("<MouseWheel>")
        except:
            pass
        self.destroy()
        # Main window is managed by InspectorApp, which should still be running
    
    def _init_log_file(self):
        """Initialize log file with headers."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Image', 'Verdict', 'AreaScore', 
                               'AnomalyCount', 'PixelVerdict', 'SSIM', 'Time'])
    
    def _build_menu(self):
        """Build menu bar with navigation and help."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Navigate menu
        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Navigate", menu=nav_menu)
        nav_menu.add_command(label="← Back to Main Window", command=self._go_back_to_main)
        nav_menu.add_separator()
        nav_menu.add_command(label="Close This Window", command=self._on_close)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="How to Use", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _show_help(self):
        """Show help dialog."""
        help_text = """Pixel Inspection System (Hiatus)

This tool compares a Golden (reference) image with a Test image to detect anomalies.

1. Load Golden Image: Select the reference/master image.
2. Load Test Image: Select the image to inspect.
3. Click 'Inspect' to run the analysis.
4. Results show SSIM score, pixel differences, and anomaly locations.

Settings:
- Alignment: Choose method for aligning images (Auto, Phase, ORB, etc.)
- Light Mode: Adjust for different lighting conditions.
- Inspection Mode: Choose between Pixel-Wise or Template matching."""
        messagebox.showinfo("Help - Pixel Inspection", help_text)
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About", "Pixel Inspection System (Hiatus)\n\nAdvanced PCB inspection with SSIM and pixel matching.\n\nPart of Integrated PCB Inspector.")
    
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TLabel", background=self.BG_COLOR, foreground=self.FG_COLOR,
                       font=(self.FONT_FACE, 10))
        style.configure("TButton", background="#333333", foreground=self.FG_COLOR,
                       font=(self.FONT_FACE, 10, 'bold'))
        style.configure("TFrame", background=self.BG_COLOR)
        style.configure("TCheckbutton", background=self.BG_COLOR, foreground=self.FG_COLOR)
    
    def _build_ui(self):
        """Build the main UI layout."""
        # Main container with controls on left
        outer_frame = ttk.Frame(self)
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls (fixed, not scrollable)
        controls_frame = tk.Frame(outer_frame, bg=self.BG_COLOR, width=320)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        controls_frame.pack_propagate(False)
        
        self._build_controls(controls_frame)
        
        # Right side: Scrollable display area
        display_container = tk.Frame(outer_frame, bg=self.BG_COLOR)
        display_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling the display area
        self.display_canvas = tk.Canvas(display_container, bg=self.BG_COLOR, highlightthickness=0)
        display_scrollbar = ttk.Scrollbar(display_container, orient=tk.VERTICAL, 
                                          command=self.display_canvas.yview)
        
        self.display_scrollable_frame = tk.Frame(self.display_canvas, bg=self.BG_COLOR)
        
        self.display_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.display_canvas.configure(scrollregion=self.display_canvas.bbox("all"))
        )
        
        self.display_canvas.create_window((0, 0), window=self.display_scrollable_frame, anchor="nw")
        self.display_canvas.configure(yscrollcommand=display_scrollbar.set)
        
        # Bind mousewheel for scrolling display area
        # Bind mousewheel for scrolling display area
        def _on_display_mousewheel(event):
            try:
                if self.winfo_exists():
                    self.display_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except (tk.TclError, AttributeError):
                pass
        
        self.display_canvas.bind_all("<MouseWheel>", _on_display_mousewheel)
        
        display_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.display_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._build_display_area(self.display_scrollable_frame)
        
        # Status bar
        self._build_status_bar()
    
    def _build_controls(self, parent):
        """Build control panel."""
        title_label = tk.Label(parent, text="[ INTEGRATED INSPECTOR ]",
                              font=(self.FONT_FACE, 14, 'bold'),
                              bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
        # Back to Main button
        back_btn = tk.Button(parent, text="← Back to Main", 
                            font=(self.FONT_FACE, 9),
                            bg="#333333", fg=self.FG_COLOR,
                            command=self._go_back_to_main)
        back_btn.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # === IMAGE LOADING ===
        load_frame = tk.Frame(parent, bg=self.BG_COLOR,
                             highlightbackground=self.FG_COLOR, highlightthickness=1)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(load_frame, text="< IMAGE LOADING >").pack(pady=5)
        
        ttk.Button(load_frame, text="Load Golden Image",
                  command=self._load_golden).pack(fill=tk.X, padx=5, pady=2)
        self.golden_status = ttk.Label(load_frame, text="Not loaded", foreground="#888888")
        self.golden_status.pack()
        
        ttk.Button(load_frame, text="Load Test Image",
                  command=self._load_test).pack(fill=tk.X, padx=5, pady=2)
        self.test_status = ttk.Label(load_frame, text="Not loaded", foreground="#888888")
        self.test_status.pack(pady=(0, 5))
        
        # === ADVANCED SETTINGS ===
        advanced_frame = tk.Frame(parent, bg=self.BG_COLOR,
                                 highlightbackground=self.FG_COLOR, highlightthickness=1)
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(advanced_frame, text="< ADVANCED SETTINGS >",
                 foreground=self.ACCENT_COLOR).pack(pady=5)
        
        # Alignment method
        align_row = tk.Frame(advanced_frame, bg=self.BG_COLOR)
        align_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(align_row, text="Alignment:").pack(side=tk.LEFT)
        
        self.alignment_var = tk.StringVar(value="Auto")
        alignment_combo = ttk.Combobox(align_row, textvariable=self.alignment_var,
                                       values=["Auto", "Phase", "ORB", "SIFT", "ECC"],
                                       state="readonly", width=12)
        alignment_combo.pack(side=tk.LEFT, padx=5)
        alignment_combo.bind("<<ComboboxSelected>>", self._on_alignment_change)
        
        # Light mode
        light_row = tk.Frame(advanced_frame, bg=self.BG_COLOR)
        light_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(light_row, text="Light Mode:").pack(side=tk.LEFT)
        
        self.light_mode_var = tk.StringVar(value="Auto")
        light_combo = ttk.Combobox(light_row, textvariable=self.light_mode_var,
                                   values=["Auto", "Low Light", "High Light", "HDR", "Standard", "Gold Pad"],
                                   state="readonly", width=12)
        light_combo.pack(side=tk.LEFT, padx=5)
        light_combo.bind("<<ComboboxSelected>>", self._on_light_mode_change)
        
        # Inspection Mode (NEW)
        inspect_row = tk.Frame(advanced_frame, bg=self.BG_COLOR)
        inspect_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(inspect_row, text="Inspect Mode:").pack(side=tk.LEFT)
        
        self.inspect_mode_var = tk.StringVar(value="Pixel-wise")
        inspect_combo = ttk.Combobox(inspect_row, textvariable=self.inspect_mode_var,
                                     values=["Pixel-wise", "Template Match"],
                                     state="readonly", width=12)
        inspect_combo.pack(side=tk.LEFT, padx=5)
        inspect_combo.bind("<<ComboboxSelected>>", self._on_inspect_mode_change)
        
        # Template grid size (only relevant for Template Match mode)
        template_row = tk.Frame(advanced_frame, bg=self.BG_COLOR)
        template_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(template_row, text="Template Grid:").pack(side=tk.LEFT)
        
        self.template_grid_var = tk.StringVar(value="4x4")
        template_combo = ttk.Combobox(template_row, textvariable=self.template_grid_var,
                                      values=["2x2", "3x3", "4x4", "5x5", "6x6", "8x8"],
                                      state="readonly", width=12)
        template_combo.pack(side=tk.LEFT, padx=5)
        template_combo.bind("<<ComboboxSelected>>", self._on_template_grid_change)
        
        # Gamma slider
        gamma_row = tk.Frame(advanced_frame, bg=self.BG_COLOR)
        gamma_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(gamma_row, text="Gamma:").pack(side=tk.LEFT)
        
        self.gamma_slider = ttk.Scale(gamma_row, from_=0.5, to=2.0, orient=tk.HORIZONTAL,
                                     command=self._on_gamma_change)
        self.gamma_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.gamma_label = ttk.Label(gamma_row, text="1.00", width=4)
        self.gamma_label.pack(side=tk.LEFT)
        self.gamma_slider.set(1.0)
        
        # Illumination normalization
        illum_row = tk.Frame(advanced_frame, bg=self.BG_COLOR)
        illum_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(illum_row, text="Normalize:").pack(side=tk.LEFT)
        
        self.illum_var = tk.StringVar(value="Match Histogram")
        illum_combo = ttk.Combobox(illum_row, textvariable=self.illum_var,
                                   values=["Match Histogram", "CLAHE Both", "Normalize Both", "None"],
                                   state="readonly", width=12)
        illum_combo.pack(side=tk.LEFT, padx=5)
        
        # Alignment confidence display
        self.align_confidence_label = ttk.Label(advanced_frame, 
                                                text="Align Confidence: --",
                                                foreground="#FFFF00")
        self.align_confidence_label.pack(fill=tk.X, padx=5, pady=5)
        
        # === DETECTION THRESHOLDS ===
        thresh_frame = tk.Frame(parent, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        thresh_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(thresh_frame, text="< DETECTION THRESHOLDS >").pack(pady=5)
        
        # Pixel threshold with entry
        pixel_row = tk.Frame(thresh_frame, bg=self.BG_COLOR)
        pixel_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(pixel_row, text="Pixel Diff:", width=12).pack(side=tk.LEFT)
        self.pixel_var = tk.IntVar(value=40)
        self.pixel_slider = ttk.Scale(pixel_row, from_=10, to=100, orient=tk.HORIZONTAL,
                                      variable=self.pixel_var, command=self._on_pixel_slider)
        self.pixel_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.pixel_entry = ttk.Entry(pixel_row, textvariable=self.pixel_var, width=5)
        self.pixel_entry.pack(side=tk.LEFT, padx=2)
        self.pixel_entry.bind('<Return>', self._on_pixel_entry)
        
        # Count threshold with entry
        count_row = tk.Frame(thresh_frame, bg=self.BG_COLOR)
        count_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(count_row, text="Count Thresh:", width=12).pack(side=tk.LEFT)
        self.count_var = tk.IntVar(value=5000)
        self.count_slider = ttk.Scale(count_row, from_=100, to=10000, orient=tk.HORIZONTAL,
                                      variable=self.count_var, command=self._on_count_slider)
        self.count_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.count_entry = ttk.Entry(count_row, textvariable=self.count_var, width=5)
        self.count_entry.pack(side=tk.LEFT, padx=2)
        self.count_entry.bind('<Return>', self._on_count_entry)
        
        # Noise Reduction slider with entry (NEW)
        noise_row = tk.Frame(thresh_frame, bg=self.BG_COLOR)
        noise_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(noise_row, text="Noise Reduce:", width=12).pack(side=tk.LEFT)
        self.noise_var = tk.IntVar(value=3)
        self.noise_slider = ttk.Scale(noise_row, from_=0, to=15, orient=tk.HORIZONTAL,
                                      variable=self.noise_var, command=self._on_noise_slider)
        self.noise_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.noise_entry = ttk.Entry(noise_row, textvariable=self.noise_var, width=5)
        self.noise_entry.pack(side=tk.LEFT, padx=2)
        self.noise_entry.bind('<Return>', self._on_noise_entry)
        
        # Auto-crop checkbox
        self.auto_crop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(thresh_frame, text="Auto-Crop Substrate",
                       variable=self.auto_crop_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Gold Focus Mode checkbox (NEW) - Focus on gold pads, filter out green
        self.gold_focus_var = tk.BooleanVar(value=True)  # Default ON for gold pad inspection
        gold_focus_cb = ttk.Checkbutton(thresh_frame, text="🔶 Gold Focus (filter green)",
                                        variable=self.gold_focus_var)
        gold_focus_cb.pack(fill=tk.X, padx=5, pady=2)
        
        # === RUN BUTTON ===
        run_btn = tk.Button(parent, text="▶ RUN INSPECTION",
                           font=(self.FONT_FACE, 14, 'bold'),
                           bg="#004400", fg=self.FG_COLOR,
                           activebackground="#006600",
                           command=self._run_inspection)
        run_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # === SAVE BUTTON ===
        save_btn = tk.Button(parent, text="💾 SAVE RESULTS",
                            font=(self.FONT_FACE, 12, 'bold'),
                            bg="#444400", fg="#FFFF00",
                            activebackground="#666600",
                            command=self._save_results)
        save_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # === RESULTS ===
        self.result_label = tk.Label(parent, text="RESULT: --",
                                    font=(self.FONT_FACE, 16, 'bold'),
                                    bg=self.BG_COLOR, fg="#888888")
        self.result_label.pack(pady=10)
        
        # Stats display
        self.stats_label = ttk.Label(parent, text="", justify=tk.LEFT)
        self.stats_label.pack(fill=tk.X, padx=10)
    def _build_display_area(self, parent):
        """Build image display area as 2x3 grid using Canvases."""
        # Row 1: Golden and Original Test
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Golden image (1,1)
        golden_frame = ttk.Frame(row1)
        golden_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(golden_frame, text="GOLDEN", font=('Consolas', 8)).pack()
        self.golden_canvas = tk.Canvas(golden_frame, bg="#111111", highlightthickness=0)
        self.golden_canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_canvas_events(self.golden_canvas)
        
        # Sample/Test image (1,2)
        test_frame = ttk.Frame(row1)
        test_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(test_frame, text="SAMPLE", font=('Consolas', 8)).pack()
        self.test_canvas = tk.Canvas(test_frame, bg="#111111", highlightthickness=0)
        self.test_canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_canvas_events(self.test_canvas)
        
        # Row 2: Aligned and Anomaly Heatmap (center)
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.BOTH, expand=True, pady=1)
        
        # Aligned image (2,1)
        aligned_frame = ttk.Frame(row2)
        aligned_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(aligned_frame, text="ALIGNED", font=('Consolas', 8)).pack()
        self.aligned_canvas = tk.Canvas(aligned_frame, bg="#111111", highlightthickness=0)
        self.aligned_canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_canvas_events(self.aligned_canvas)
        
        # ANOMALY HEATMAP (2,2)
        anomaly_frame = ttk.Frame(row2)
        anomaly_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        anomaly_title = tk.Label(anomaly_frame, text="⚠ ANOMALY ⚠",
                                font=(self.FONT_FACE, 8, 'bold'),
                                bg=self.BG_COLOR, fg="#FF4444")
        anomaly_title.pack()
        self.pixel_canvas = tk.Canvas(anomaly_frame, bg="#111111", highlightthickness=1, highlightbackground="#FF4444")
        self.pixel_canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_canvas_events(self.pixel_canvas)
        
        # Row 3: SSIM and Contour Map
        row3 = ttk.Frame(parent)
        row3.pack(fill=tk.BOTH, expand=True, pady=1)
        
        # SSIM heatmap (3,1)
        ssim_frame = ttk.Frame(row3)
        ssim_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(ssim_frame, text="SSIM", font=('Consolas', 8)).pack()
        self.ssim_canvas = tk.Canvas(ssim_frame, bg="#111111", highlightthickness=0)
        self.ssim_canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_canvas_events(self.ssim_canvas)
        
        # Contour Map (3,2)
        contour_frame = ttk.Frame(row3)
        contour_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(contour_frame, text="CONTOUR", font=('Consolas', 8)).pack()
        self.contour_canvas = tk.Canvas(contour_frame, bg="#111111", highlightthickness=0)
        self.contour_canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_canvas_events(self.contour_canvas)

    def _bind_canvas_events(self, canvas):
        canvas.bind("<Motion>", self._on_mouse_move)
        canvas.bind("<Leave>", self._on_mouse_leave)
    
    def _build_status_bar(self):
        # Stats display
        self.status_var = tk.StringVar(value="Ready")
        
        status_frame = tk.Frame(self, bg="#222222")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                              bg="#222222", fg=self.FG_COLOR,
                              font=(self.FONT_FACE, 9), anchor=tk.W)
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.lbl_coords = tk.Label(status_frame, text="XY: -",
                                 bg="#222222", fg="#00FFFF",
                                 font=(self.FONT_FACE, 9, 'bold'))
        self.lbl_coords.pack(side=tk.RIGHT, padx=10)
    
    # === EVENT HANDLERS ===
    
    def _on_alignment_change(self, event=None):
        """Handle alignment method change."""
        mapping = {
            "Auto": AlignmentMethod.AUTO,
            "Phase": AlignmentMethod.PHASE_CORRELATION,
            "ORB": AlignmentMethod.ORB_HOMOGRAPHY,
            "SIFT": AlignmentMethod.SIFT_HOMOGRAPHY,
            "ECC": AlignmentMethod.ECC
        }
        self.selected_alignment_method = mapping.get(self.alignment_var.get(), AlignmentMethod.AUTO)
    
    def _on_light_mode_change(self, event=None):
        """Handle light mode change."""
        mapping = {
            "Auto": LightSensitivityMode.AUTO,
            "Low Light": LightSensitivityMode.LOW_LIGHT,
            "High Light": LightSensitivityMode.HIGH_LIGHT,
            "HDR": LightSensitivityMode.HDR,
            "Standard": LightSensitivityMode.STANDARD,
            "Gold Pad": LightSensitivityMode.GOLD_PAD
        }
        self.selected_light_mode = mapping.get(self.light_mode_var.get(), LightSensitivityMode.AUTO)
    
    def _on_inspect_mode_change(self, event=None):
        """Handle inspection mode change."""
        mapping = {
            "Pixel-wise": InspectionMode.PIXEL_WISE,
            "Template Match": InspectionMode.TEMPLATE_MATCH
        }
        self.selected_inspection_mode = mapping.get(self.inspect_mode_var.get(), InspectionMode.PIXEL_WISE)
    
    def _on_template_grid_change(self, event=None):
        """Handle template grid size change."""
        grid_str = self.template_grid_var.get()
        try:
            cols, rows = map(int, grid_str.split('x'))
            self.template_grid_cols = cols
            self.template_grid_rows = rows
        except ValueError:
            self.template_grid_cols = 4
            self.template_grid_rows = 4
    
    def _on_gamma_change(self, value):
        """Handle gamma slider change."""
        self.gamma_value = float(value)
        self.gamma_label.config(text=f"{self.gamma_value:.2f}")
    
    def _on_pixel_slider(self, value):
        """Handle pixel slider change - update variable."""
        self.pixel_var.set(int(float(value)))
    
    def _on_pixel_entry(self, event=None):
        """Handle pixel entry - clamp value to valid range."""
        try:
            val = int(self.pixel_var.get())
            val = max(10, min(100, val))  # Clamp to 10-100
            self.pixel_var.set(val)
        except ValueError:
            self.pixel_var.set(40)  # Default
    
    def _on_count_slider(self, value):
        """Handle count slider change - update variable."""
        self.count_var.set(int(float(value)))
    
    def _on_count_entry(self, event=None):
        """Handle count entry - clamp value to valid range."""
        try:
            val = int(self.count_var.get())
            val = max(100, min(10000, val))  # Clamp to 100-10000
            self.count_var.set(val)
        except ValueError:
            self.count_var.set(5000)  # Default
    
    def _on_noise_slider(self, value):
        """Handle noise slider change - update variable."""
        self.noise_var.set(int(float(value)))
    
    def _on_noise_entry(self, event=None):
        """Handle noise entry - clamp value to valid range."""
        try:
            val = int(self.noise_var.get())
            val = max(0, min(15, val))  # Clamp to 0-15
            self.noise_var.set(val)
        except ValueError:
            self.noise_var.set(3)  # Default
    
    # === IMAGE LOADING ===
    
    def _load_golden(self):
        """Load golden/master image."""
        path = filedialog.askopenfilename(
            title="Select Golden Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                self.golden_image = read_image(path)
                self._golden_loaded = True
                self.last_golden_path = path
                self.golden_status.config(text=os.path.basename(path), foreground=self.FG_COLOR)
                self._display_on_canvas(self.golden_image, self.golden_canvas)
                self.status_var.set(f"Loaded golden: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _load_test(self):
        """Load test image."""
        path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                self.test_image = read_image(path)
                self._test_loaded = True
                self.last_test_path = path
                self.test_status.config(text=os.path.basename(path), foreground=self.FG_COLOR)
                self._display_on_canvas(self.test_image, self.test_canvas)  # Show test image immediately
                self.status_var.set(f"Loaded test: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    # === INSPECTION ===
    
    def _run_inspection(self):
        """Run the inspection pipeline."""
        if not self._golden_loaded or self.golden_image is None:
            messagebox.showwarning("Missing Image", "Please load a Golden Image first.")
            return
        if not self._test_loaded or self.test_image is None:
            messagebox.showwarning("Missing Image", "Please load a Test Image first.")
            return
        
        start_time = time.time()
        self.status_var.set("Running inspection...")
        self.update_idletasks()
        
        try:
            # Apply light sensitivity
            light_config = LightSensitivityConfig(mode=self.selected_light_mode, gamma=self.gamma_value)
            golden_proc = apply_light_sensitivity_mode(self.golden_image, self.selected_light_mode, light_config)
            test_proc = apply_light_sensitivity_mode(self.test_image, self.selected_light_mode, light_config)
            
            # Auto-crop if enabled
            if self.auto_crop_var.get():
                golden_crops = run_edge_detection(golden_proc)
                if golden_crops:
                    golden_proc = golden_crops[0]
                test_crops = run_edge_detection(test_proc)
                if test_crops:
                    test_proc = test_crops[0]
            
            # Resize test to match golden
            h, w = golden_proc.shape[:2]
            test_resized = cv2.resize(test_proc, (w, h))
            
            # Alignment
            self.status_var.set(f"Aligning with {self.selected_alignment_method.value}...")
            self.update_idletasks()
            
            aligned, (dx, dy), confidence, valid_mask = align_images(
                golden_proc, test_resized,
                method=self.selected_alignment_method
            )
            
            self.aligned_image = aligned
            self.align_confidence_label.config(
                text=f"Align Confidence: {confidence:.3f} ({self.selected_alignment_method.value})"
            )
            
            if confidence < 0.1:
                self.result_label.config(text="RESULT: ALIGN ERROR", fg=self.ERROR_COLOR)
                self.status_var.set("Alignment failed - low confidence")
                return
            
            self._display_on_canvas(aligned, self.aligned_canvas)
            
            # SSIM check
            self.status_var.set("Running SSIM check...")
            self.update_idletasks()
            
            self.ssim_score, ssim_heatmap = calc_ssim(golden_proc, aligned)
            self._display_on_canvas(ssim_heatmap, self.ssim_canvas)
            
            if self.ssim_score > self.SSIM_PASS_THRESHOLD:
                # SSIM pass
                processing_time = time.time() - start_time
                self.result_label.config(text="RESULT: NORMAL (SSIM)", fg=self.FG_COLOR)
                self.stats_label.config(
                    text=f"SSIM Score: {self.ssim_score:.4f}\nThreshold: {self.SSIM_PASS_THRESHOLD}\nTime: {processing_time:.2f}s"
                )
                self.status_var.set(f"Complete: NORMAL (SSIM Pass) in {processing_time:.2f}s")
                
                # Store results for saving
                self.last_result = {
                    'verdict': 'Normal',
                    'method': 'SSIM',
                    'ssim_score': self.ssim_score,
                    'processing_time': processing_time
                }
                self.last_ssim_heatmap = ssim_heatmap
                self.last_heatmap = None
                self.last_contour_map = None
                
                # Clear pixel heatmap
                blank = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.putText(blank, "N/A", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                self._display_on_canvas(blank, self.pixel_canvas)
                return
            
            self.status_var.set("Running analysis...")
            self.update_idletasks()
            
            illum_method_map = {
                "Match Histogram": "match_histogram",
                "CLAHE Both": "clahe_both",
                "Normalize Both": "normalize_both",
                "None": "none"
            }
            normalize_method = illum_method_map.get(self.illum_var.get(), "match_histogram")
            
            # Apply noise reduction if enabled
            noise_level = self.noise_var.get()
            golden_for_match = golden_proc
            aligned_for_match = aligned
            
            if noise_level > 0:
                # Apply Gaussian blur for noise reduction
                kernel_size = noise_level * 2 + 1  # Ensure odd kernel size
                golden_for_match = cv2.GaussianBlur(golden_proc, (kernel_size, kernel_size), 0)
                aligned_for_match = cv2.GaussianBlur(aligned, (kernel_size, kernel_size), 0)
            
            # Gold Focus Mode - Create mask to focus on gold regions only
            gold_mask = None
            if self.gold_focus_var.get():
                self.status_var.set("Creating gold region mask (filtering green)...")
                self.update_idletasks()
                
                # Get gold mask from golden image (HSV filter for gold color)
                gold_mask = gold_pad_hsv_filter(golden_for_match, 
                                                 hue_low=10, hue_high=45,  # Wider range for gold/yellow
                                                 sat_low=30, sat_high=255,
                                                 val_low=100, val_high=255,
                                                 return_mask=True)
                
                # Clean up mask with morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)
                gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel)
                
                # Combine with valid_mask if exists
                if valid_mask is not None:
                    valid_mask = cv2.bitwise_and(valid_mask, gold_mask)
                else:
                    valid_mask = gold_mask
            
            # Branch based on inspection mode
            if self.selected_inspection_mode == InspectionMode.TEMPLATE_MATCH:
                # Template matching mode
                self.status_var.set(f"Running template matching ({self.template_grid_cols}x{self.template_grid_rows} grid)...")
                self.update_idletasks()
                
                template_config = TemplateMatchConfig(
                    grid_cols=self.template_grid_cols,
                    grid_rows=self.template_grid_rows,
                    search_margin=self.template_search_margin,
                    diff_threshold=self.pixel_var.get(),
                    min_anomaly_area=self.count_var.get()
                )
                
                pixel_result = run_template_inspection(
                    golden_for_match, aligned_for_match, template_config
                )
                
                # Map template result to common format
                pixel_result['contour_map'] = pixel_result['annotated_image']
                pixel_result['anomaly_mask'] = pixel_result.get('anomaly_mask', np.zeros_like(golden_proc[:,:,0]))
                
            else:
                # Pixel-wise mode (original)
                self.status_var.set("Running pixel analysis...")
                self.update_idletasks()
                
                pixel_result = run_pixel_matching(
                    golden_for_match, aligned_for_match,
                    pixel_thresh=self.pixel_var.get(),
                    count_thresh=self.count_var.get(),
                    valid_area_mask=valid_mask,
                    normalize_lighting=normalize_method != "none",
                    normalize_method=normalize_method
                )
            
            # Annotate with anomaly locations if anomaly detected
            if pixel_result['verdict'] == 'Anomaly':
                self.anomaly_mapper.analyze_mask(pixel_result['anomaly_mask'])
                contour_map = self.anomaly_mapper.create_annotated_image(pixel_result['contour_map'])
            else:
                contour_map = pixel_result['contour_map']
            
            self._display_on_canvas(pixel_result['heatmap'], self.pixel_canvas)
            self._display_on_canvas(contour_map, self.contour_canvas)  # Display contour map in dedicated panel
            
            # Final result
            processing_time = time.time() - start_time
            
            if pixel_result['verdict'] == 'Anomaly':
                self.result_label.config(text="RESULT: !! ANOMALY !!", fg=self.ERROR_COLOR)
                location_text = self.anomaly_mapper.get_summary_text()
            else:
                self.result_label.config(text="RESULT: NORMAL", fg=self.FG_COLOR)
                location_text = ""
            
            stats_text = (
                f"Verdict: {pixel_result['verdict']}\n"
                f"SSIM: {self.ssim_score:.4f}\n"
                f"Area Score: {pixel_result['area_score']:.2f}%\n"
                f"Anomaly Count: {pixel_result['anomaly_count']}\n"
                f"Confidence: {pixel_result['confidence']:.3f}\n"
                f"Time: {processing_time:.2f}s"
            )
            if location_text:
                stats_text += f"\n\n{location_text}"
            
            self.stats_label.config(text=stats_text)
            self.status_var.set(f"Complete: {pixel_result['verdict']} in {processing_time:.2f}s")
            
            # Log result
            self._log_result(
                "test_image",
                pixel_result['verdict'],
                pixel_result['area_score'],
                pixel_result['anomaly_count'],
                pixel_result['verdict'],
                processing_time,
                self.ssim_score
            )
            
            # Store results for saving
            self.last_result = {
                'verdict': pixel_result['verdict'],
                'method': 'Pixel Matching',
                'ssim_score': self.ssim_score,
                'area_score': pixel_result['area_score'],
                'anomaly_count': pixel_result['anomaly_count'],
                'confidence': pixel_result['confidence'],
                'processing_time': processing_time,
                'location_summary': location_text if location_text else None
            }
            self.last_heatmap = pixel_result['heatmap']
            self.last_contour_map = contour_map
            self.last_ssim_heatmap = ssim_heatmap
            
        except Exception as e:
            self.result_label.config(text="RESULT: ERROR", fg=self.ERROR_COLOR)
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Inspection Error", str(e))
    
    def _display_on_canvas(self, cv2_image: np.ndarray, canvas: tk.Canvas, size=(450, 350)):
        """Display OpenCV image on tkinter Canvas with centering and metadata storage."""
        if cv2_image is None or cv2_image.size == 0:
            return
        
        h, w = cv2_image.shape[:2]
        if h == 0 or w == 0:
            return
        
        # 1. Calculate Resize Ratio
        # Canvas size might change, but let's assume fixed size passed or check widget size?
        # Better to check widget size if mapped, else use default 'size' param
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw > 10 and ch > 10:
            size = (cw, ch)

        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w <= 0 or new_h <= 0:
            return
        
        # 2. Resize
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        
        # 3. Calculate Offsets to Center
        off_x = (size[0] - new_w) // 2
        off_y = (size[1] - new_h) // 2
        
        # 4. Draw on Canvas
        canvas.delete("all") # Clear previous
        canvas.create_image(off_x + new_w//2, off_y + new_h//2, image=photo, anchor="center", tags="img")
        canvas.image = photo # Keep reference
        
        # 5. Store Metadata
        self.canvas_meta[canvas] = {
            "ratio": ratio,
            "off_x": off_x,
            "off_y": off_y,
            "orig_w": w,
            "orig_h": h,
            "disp_w": new_w,
            "disp_h": new_h
        }

    def _on_mouse_move(self, event):
        """Handle mouse movement on canvases to update coordinates and crosshairs."""
        canvas = event.widget
        if canvas not in self.canvas_meta:
            return
            
        meta = self.canvas_meta[canvas]
        
        # Get mouse pos relative to canvas
        mx, my = event.x, event.y
        
        # Convert to Image Coords
        # 1. Remove offset
        img_x = mx - meta["off_x"]
        img_y = my - meta["off_y"]
        
        # 2. Check bounds relative to displayed image
        if 0 <= img_x < meta["disp_w"] and 0 <= img_y < meta["disp_h"]:
             # 3. Scale back to original
            orig_x = int(img_x / meta["ratio"])
            orig_y = int(img_y / meta["ratio"])
            
            # Clamp
            orig_x = max(0, min(orig_x, meta["orig_w"] - 1))
            orig_y = max(0, min(orig_y, meta["orig_h"] - 1))
            
            self.lbl_coords.config(text=f"XY: {orig_x}, {orig_y}")
            
            # Draw Crosshair on THIS canvas
            self._draw_crosshair(canvas, mx, my)
            
            # OPTIONAL: Sync crosshairs on other canvases if they have same dimensions?
            # For now, just show on active canvas to avoid confusion with different sizes
        else:
            self.lbl_coords.config(text="XY: -")
            canvas.delete("crosshair")

    def _on_mouse_leave(self, event):
        """Clear coordinates and crosshair when leaving canvas."""
        self.lbl_coords.config(text="XY: -")
        event.widget.delete("crosshair")

    def _draw_crosshair(self, canvas, x, y):
        """Draw simple crosshair on canvas."""
        canvas.delete("crosshair")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        color = "#00FFFF" # Cyan
        
        # Horizontal
        canvas.create_line(0, y, w, y, fill=color, dash=(4, 2), tags="crosshair")
        # Vertical
        canvas.create_line(x, 0, x, h, fill=color, dash=(4, 2), tags="crosshair")
    
    def _log_result(self, test_path, verdict, p_score, p_count, p_verdict, p_time, ssim_score):
        """Log inspection result to CSV."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, os.path.basename(test_path), verdict,
                           f"{p_score:.4f}", p_count, p_verdict,
                           f"{ssim_score:.4f}", f"{p_time:.2f}"])
    
    def _save_results(self):
        """Save inspection results to a folder."""
        import json
        from datetime import datetime
        
        if self.last_result is None:
            messagebox.showwarning("No Results", "No inspection results to save. Run an inspection first.")
            return
        
        # Ask user for output folder
        output_dir = filedialog.askdirectory(title="Select Output Folder for Results")
        if not output_dir:
            return
        
        try:
            # Create timestamped subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_folder = os.path.join(output_dir, f"inspection_{timestamp}")
            os.makedirs(result_folder, exist_ok=True)
            
            self.status_var.set("Saving results...")
            self.update_idletasks()
            
            saved_files = []
            
            # Save aligned image
            if self.aligned_image is not None:
                aligned_path = os.path.join(result_folder, "aligned_test.png")
                cv2.imwrite(aligned_path, self.aligned_image)
                saved_files.append("aligned_test.png")
            
            # Save golden image
            if self.golden_image is not None:
                golden_path = os.path.join(result_folder, "golden_reference.png")
                cv2.imwrite(golden_path, self.golden_image)
                saved_files.append("golden_reference.png")
            
            # Save sample/test image (original, before alignment)
            if self.test_image is not None:
                sample_path = os.path.join(result_folder, "sample_test.png")
                cv2.imwrite(sample_path, self.test_image)
                saved_files.append("sample_test.png")
            
            # Save SSIM heatmap
            if self.last_ssim_heatmap is not None:
                ssim_path = os.path.join(result_folder, "ssim_heatmap.png")
                cv2.imwrite(ssim_path, self.last_ssim_heatmap)
                saved_files.append("ssim_heatmap.png")
            
            # Save anomaly heatmap
            if self.last_heatmap is not None:
                heatmap_path = os.path.join(result_folder, "anomaly_heatmap.png")
                cv2.imwrite(heatmap_path, self.last_heatmap)
                saved_files.append("anomaly_heatmap.png")
            
            # Save contour map with annotations
            if self.last_contour_map is not None:
                contour_path = os.path.join(result_folder, "contour_map.png")
                cv2.imwrite(contour_path, self.last_contour_map)
                saved_files.append("contour_map.png")
            
            # Save JSON results
            json_result = {
                'timestamp': datetime.now().isoformat(),
                'golden_image': os.path.basename(self.last_golden_path) if self.last_golden_path else None,
                'test_image': os.path.basename(self.last_test_path) if self.last_test_path else None,
                **self.last_result,
                'settings': {
                    'alignment_method': self.selected_alignment_method.value,
                    'light_mode': self.selected_light_mode.value,
                    'gamma': self.gamma_value,
                    'pixel_threshold': int(self.pixel_slider.get()),
                    'count_threshold': int(self.count_slider.get())
                },
                'saved_files': saved_files
            }
            
            json_path = os.path.join(result_folder, "inspection_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            saved_files.append("inspection_results.json")
            
            # Save summary text file
            summary_path = os.path.join(result_folder, "summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("INSPECTION RESULTS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {json_result['timestamp']}\n")
                f.write(f"Golden Image: {json_result['golden_image']}\n")
                f.write(f"Test Image: {json_result['test_image']}\n\n")
                f.write(f"Verdict: {self.last_result['verdict']}\n")
                f.write(f"Method: {self.last_result['method']}\n")
                f.write(f"SSIM Score: {self.last_result.get('ssim_score', 'N/A')}\n")
                if 'area_score' in self.last_result:
                    f.write(f"Area Score: {self.last_result['area_score']:.2f}%\n")
                    f.write(f"Anomaly Count: {self.last_result['anomaly_count']}\n")
                    f.write(f"Confidence: {self.last_result['confidence']:.3f}\n")
                f.write(f"Processing Time: {self.last_result['processing_time']:.2f}s\n")
                if self.last_result.get('location_summary'):
                    f.write(f"\nAnomaly Locations:\n{self.last_result['location_summary']}\n")
            saved_files.append("summary.txt")
            
            self.status_var.set(f"Results saved to: {result_folder}")
            messagebox.showinfo("Results Saved", 
                f"Inspection results saved successfully!\n\n"
                f"Location: {result_folder}\n\n"
                f"Files saved:\n" + "\n".join(f"• {f}" for f in saved_files))
                
        except Exception as e:
            self.status_var.set(f"Error saving: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")


# ==============================================================================
# QR CODE CROPPER WINDOW
# ==============================================================================

class QRCropperWindow(tk.Toplevel):
    """QR Code Detection and Cropping Window."""
    
    BG_COLOR = "#000000"
    FG_COLOR = "#00FF41"
    ACCENT_COLOR = "#00FF41"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("QR Code Extractor")
        self.geometry("1000x700")
        self.configure(bg=self.BG_COLOR)
        
        # Import QR extractor
        from .qr_cropper import QRCodeExtractor
        self.extractor = QRCodeExtractor()
        
        self.current_image = None
        self.last_results = []
        
        self._build_menu()
        self._build_ui()
    
    def _build_menu(self):
        """Build menu bar with navigation."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Navigate menu
        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Navigate", menu=nav_menu)
        nav_menu.add_command(label="🏠 Home (Inspector)", command=self._go_home)
        nav_menu.add_separator()
        nav_menu.add_command(label="✕ Close Window", command=self.destroy)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🔲 QR Code Extractor", command=lambda: self.lift())
        tools_menu.add_command(label="🔶 Gold Pad Extractor", command=lambda: self.parent._open_gold_pad_extractor())
        tools_menu.add_command(label="🔴 Red Pad Extractor", command=lambda: self.parent._open_red_pad_extractor())
        tools_menu.add_command(label="🔍 Simple Defect Detection", command=lambda: self.parent._open_simple_defect_detection())
        tools_menu.add_separator()
        tools_menu.add_command(label="📦 Batch Inspection", command=lambda: self.parent._run_batch_inspection())
        tools_menu.add_separator()
        tools_menu.add_command(label="📄 Open Log File", command=lambda: self.parent._open_log_file())
    
    def _go_home(self):
        """Bring main Inspector window to front."""
        if hasattr(self.parent, '_show_home'):
            self.parent._show_home()
        else:
            self.parent.deiconify()
            self.parent.lift()
            self.parent.focus_force()
    
    def _build_ui(self):
        """Build the QR Cropper UI."""
        # Main container
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=250)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(controls_frame, text="[ QR EXTRACTOR ]",
                              font=(self.FONT_FACE, 14, 'bold'),
                              bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
        # Load button
        load_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                             highlightbackground=self.FG_COLOR, highlightthickness=1)
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Image",
                  command=self._load_image).pack(fill=tk.X, padx=5, pady=5)
        
        self.image_status = ttk.Label(load_frame, text="No image loaded", 
                                      foreground="#888888")
        self.image_status.pack(pady=(0, 5))
        
        # Action buttons
        action_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(action_frame, text="< ACTIONS >").pack(pady=5)
        
        detect_btn = tk.Button(action_frame, text="▶ DETECT QR CODES",
                              font=(self.FONT_FACE, 11, 'bold'),
                              bg="#004400", fg=self.FG_COLOR,
                              command=self._detect_qr)
        detect_btn.pack(fill=tk.X, padx=5, pady=5)
        
        extract_btn = tk.Button(action_frame, text="📁 EXTRACT & SAVE",
                               font=(self.FONT_FACE, 11, 'bold'),
                               bg="#440044", fg="#FF88FF",
                               command=self._extract_and_save)
        extract_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Output folder
        folder_frame = tk.Frame(action_frame, bg=self.BG_COLOR)
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(folder_frame, text="Output Folder:").pack(anchor=tk.W)
        
        self.output_folder_var = tk.StringVar(value="qrcode_extraction")
        folder_entry = ttk.Entry(folder_frame, textvariable=self.output_folder_var)
        folder_entry.pack(fill=tk.X, pady=2)
        
        ttk.Button(folder_frame, text="Browse...",
                  command=self._browse_folder).pack(fill=tk.X, pady=2)
        
        # Results display
        results_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                                highlightbackground=self.FG_COLOR, highlightthickness=1)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(results_frame, text="< RESULTS >").pack(pady=5)
        
        self.results_text = tk.Text(results_frame, bg="#111111", fg=self.FG_COLOR,
                                   font=(self.FONT_FACE, 9), height=15, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Image display
        display_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(display_frame, text="IMAGE PREVIEW").pack()
        
        self.image_label = tk.Label(display_frame, bg="#111111")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to detect QR codes")
        status_bar = tk.Label(self, textvariable=self.status_var,
                             bg="#222222", fg=self.FG_COLOR,
                             font=(self.FONT_FACE, 9), anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _load_image(self):
        """Load an image for QR detection."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                self.image_status.config(text=os.path.basename(path), 
                                        foreground=self.FG_COLOR)
                self._display_image(self.current_image)
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded. Click 'DETECT QR CODES' to scan.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _detect_qr(self):
        """Detect QR codes in loaded image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.status_var.set("Detecting QR codes...")
        self.update_idletasks()
        
        try:
            results = self.extractor.detect_and_decode(self.current_image)
            self.last_results = results
            
            # Display annotated image
            annotated = self.extractor.annotate_image(self.current_image)
            self._display_image(annotated)
            
            # Show results
            self.results_text.delete(1.0, tk.END)
            
            if not results:
                self.results_text.insert(tk.END, "No QR codes detected.\n\n")
                self.results_text.insert(tk.END, "Tips:\n")
                self.results_text.insert(tk.END, "- Ensure QR code is clearly visible\n")
                self.results_text.insert(tk.END, "- Try improving lighting/contrast\n")
                self.results_text.insert(tk.END, "- QR code should not be rotated >45°\n")
                self.status_var.set("No QR codes found")
            else:
                self.results_text.insert(tk.END, f"Found {len(results)} QR code(s):\n\n")
                
                for qr in results:
                    self.results_text.insert(tk.END, f"═══ QR #{qr['id']} ═══\n")
                    self.results_text.insert(tk.END, f"Data: {qr['data']}\n")
                    self.results_text.insert(tk.END, f"Position: {qr['center']}\n")
                    self.results_text.insert(tk.END, f"Size: {qr['bbox'][2]}x{qr['bbox'][3]}\n")
                    self.results_text.insert(tk.END, f"Method: {qr['method']}\n\n")
                
                self.status_var.set(f"Found {len(results)} QR code(s)")
                
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def _extract_and_save(self):
        """Extract detected QR codes and save to folder."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if not self.last_results:
            # Try detecting first
            self._detect_qr()
            if not self.last_results:
                messagebox.showinfo("No QR Codes", "No QR codes were detected to extract.")
                return
        
        output_dir = self.output_folder_var.get()
        self.extractor.output_dir = output_dir
        
        self.status_var.set(f"Saving QR codes to {output_dir}...")
        self.update_idletasks()
        
        try:
            saved_paths = self.extractor.save_cropped_qr(self.current_image)
            
            if saved_paths:
                self.results_text.insert(tk.END, "\n═══ SAVED FILES ═══\n")
                for path in saved_paths:
                    self.results_text.insert(tk.END, f"✓ {os.path.basename(path)}\n")
                
                self.results_text.insert(tk.END, f"\nSaved to: {os.path.abspath(output_dir)}\n")
                self.status_var.set(f"Saved {len(saved_paths)} QR code(s) to {output_dir}")
                
                messagebox.showinfo("Success", 
                    f"Saved {len(saved_paths)} QR code(s) to:\n{os.path.abspath(output_dir)}")
            else:
                self.status_var.set("No QR codes to save")
                
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            self.status_var.set(f"Error: {str(e)}")
    
    def _browse_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
    
    def _display_image(self, cv2_image: np.ndarray, size=(600, 500)):
        """Display image in the label."""
        if cv2_image is None or cv2_image.size == 0:
            return
        
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w == 0 or new_h == 0:
            return
        
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        
        self.image_label.config(image=photo)
        self.image_label.image = photo


# ==============================================================================
# GOLD PAD EXTRACTOR WINDOW
# ==============================================================================

class GoldPadExtractorWindow(tk.Toplevel):
    """Gold Pad Extraction Tool - Extract gold circles from PCB strips using HSV filtering."""
    
    BG_COLOR = "#000000"
    FG_COLOR = "#FFD700"  # Gold color
    ACCENT_COLOR = "#FFD700"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("Gold Pad Extractor")
        self.geometry("1200x800")
        self.configure(bg=self.BG_COLOR)
        
        self.current_image = None
        self.detected_pads = []
        self.extracted_pads = []
        self.preview_image = None
        
        # Default HSV range for gold (adjustable)
        self.hue_low = tk.IntVar(value=15)
        self.hue_high = tk.IntVar(value=35)
        self.sat_low = tk.IntVar(value=50)
        self.sat_high = tk.IntVar(value=255)
        self.val_low = tk.IntVar(value=100)
        self.val_high = tk.IntVar(value=255)
        
        # Circle detection params
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=100)
        self.min_circularity = tk.DoubleVar(value=0.7)
        
        self._build_menu()
        self._build_ui()
    
    def _build_menu(self):
        """Build menu bar with navigation."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Navigate menu
        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Navigate", menu=nav_menu)
        nav_menu.add_command(label="🏠 Home (Inspector)", command=self._go_home)
        nav_menu.add_separator()
        nav_menu.add_command(label="✕ Close Window", command=self.destroy)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🔲 QR Code Extractor", command=lambda: self.parent._open_qr_cropper())
        tools_menu.add_command(label="🔶 Gold Pad Extractor", command=lambda: self.lift())
        tools_menu.add_command(label="🔴 Red Pad Extractor", command=lambda: self.parent._open_red_pad_extractor())
        tools_menu.add_command(label="🔍 Simple Defect Detection", command=lambda: self.parent._open_simple_defect_detection())
        tools_menu.add_separator()
        tools_menu.add_command(label="📦 Batch Inspection", command=lambda: self.parent._run_batch_inspection())
        tools_menu.add_separator()
        tools_menu.add_command(label="📄 Open Log File", command=lambda: self.parent._open_log_file())
    
    def _go_home(self):
        """Close this tool window and return focus to main."""
        self.destroy()
    
    def _build_ui(self):
        """Build the Gold Pad Extractor UI."""
        # Main container
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=300)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(controls_frame, text="[ GOLD PAD EXTRACTOR ]",
                              font=(self.FONT_FACE, 14, 'bold'),
                              bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
        # Load button
        load_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                             highlightbackground=self.FG_COLOR, highlightthickness=1)
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Image",
                  command=self._load_image).pack(fill=tk.X, padx=5, pady=5)
        
        self.image_status = ttk.Label(load_frame, text="No image loaded", 
                                      foreground="#888888")
        self.image_status.pack(pady=(0, 5))
        
        # HSV Settings
        hsv_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                            highlightbackground=self.FG_COLOR, highlightthickness=1)
        hsv_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(hsv_frame, text="< HSV GOLD FILTER >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        # Hue range
        self._add_slider(hsv_frame, "Hue Low:", self.hue_low, 0, 179)
        self._add_slider(hsv_frame, "Hue High:", self.hue_high, 0, 179)
        
        # Saturation range
        self._add_slider(hsv_frame, "Sat Low:", self.sat_low, 0, 255)
        self._add_slider(hsv_frame, "Sat High:", self.sat_high, 0, 255)
        
        # Value range
        self._add_slider(hsv_frame, "Val Low:", self.val_low, 0, 255)
        self._add_slider(hsv_frame, "Val High:", self.val_high, 0, 255)
        
        # Circle detection settings
        circle_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        circle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(circle_frame, text="< CIRCLE DETECTION >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        self._add_slider(circle_frame, "Min Radius:", self.min_radius, 5, 200)
        self._add_slider(circle_frame, "Max Radius:", self.max_radius, 10, 300)
        
        # Action buttons
        action_frame = tk.Frame(controls_frame, bg=self.BG_COLOR)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        
        preview_btn = tk.Button(action_frame, text="👁 PREVIEW DETECTION",
                               font=(self.FONT_FACE, 11, 'bold'),
                               bg="#004444", fg="#00FFFF",
                               command=self._preview_detection)
        preview_btn.pack(fill=tk.X, pady=3)
        
        extract_btn = tk.Button(action_frame, text="⭕ EXTRACT GOLD PADS",
                               font=(self.FONT_FACE, 11, 'bold'),
                               bg="#444400", fg=self.FG_COLOR,
                               command=self._extract_pads)
        extract_btn.pack(fill=tk.X, pady=3)
        
        save_btn = tk.Button(action_frame, text="💾 SAVE EXTRACTED PADS",
                            font=(self.FONT_FACE, 11, 'bold'),
                            bg="#440044", fg="#FF88FF",
                            command=self._save_pads)
        save_btn.pack(fill=tk.X, pady=3)
        
        # Results
        result_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(result_frame, text="< RESULTS >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        self.results_text = tk.Text(result_frame, bg="#111111", fg=self.FG_COLOR,
                                   font=(self.FONT_FACE, 9), height=10, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Image displays
        display_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Original/Preview
        preview_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(preview_frame, text="PREVIEW (Gold Mask / Detection)").pack()
        self.preview_label = tk.Label(preview_frame, bg="#111111")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Extracted pads gallery
        gallery_frame = tk.Frame(display_frame, bg=self.BG_COLOR, height=200)
        gallery_frame.pack(fill=tk.X, pady=10)
        gallery_frame.pack_propagate(False)
        
        ttk.Label(gallery_frame, text="EXTRACTED PADS").pack()
        
        self.gallery_canvas = tk.Canvas(gallery_frame, bg="#111111", height=150)
        self.gallery_canvas.pack(fill=tk.X, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to extract gold pads")
        status_bar = tk.Label(self, textvariable=self.status_var,
                             bg="#222222", fg=self.FG_COLOR,
                             font=(self.FONT_FACE, 9), anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _add_slider(self, parent, label, variable, from_, to):
        """Add a labeled slider."""
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        slider = ttk.Scale(row, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        value_label = ttk.Label(row, textvariable=variable, width=4)
        value_label.pack(side=tk.LEFT)
    
    def _load_image(self):
        """Load an image for gold pad extraction."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                self.image_status.config(text=os.path.basename(path), 
                                        foreground=self.FG_COLOR)
                self._display_image(self.current_image, self.preview_label)
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded.\n\nClick 'PREVIEW DETECTION' to see gold mask.\nAdjust HSV sliders if needed.\nThen click 'EXTRACT GOLD PADS'.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _get_gold_mask(self, image):
        """Create HSV mask for gold regions."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower = np.array([self.hue_low.get(), self.sat_low.get(), self.val_low.get()])
        upper = np.array([self.hue_high.get(), self.sat_high.get(), self.val_high.get()])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _detect_circles(self, mask):
        """Detect circular gold pads from the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        min_r = self.min_radius.get()
        max_r = self.max_radius.get()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
            
            # Fit minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if min_r <= radius <= max_r:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity >= self.min_circularity.get():
                        circles.append({
                            'center': (int(x), int(y)),
                            'radius': int(radius),
                            'area': area,
                            'circularity': circularity,
                            'contour': contour
                        })
        
        # Sort by y then x (top-left to bottom-right)
        circles.sort(key=lambda c: (c['center'][1] // 50, c['center'][0]))
        
        return circles
    
    def _preview_detection(self):
        """Preview the gold mask and detected circles."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.status_var.set("Detecting gold pads...")
        self.update_idletasks()
        
        # Get gold mask
        mask = self._get_gold_mask(self.current_image)
        
        # Detect circles
        self.detected_pads = self._detect_circles(mask)
        
        # Create preview image
        preview = self.current_image.copy()
        
        # Draw detected circles
        for i, pad in enumerate(self.detected_pads):
            cx, cy = pad['center']
            r = pad['radius']
            
            # Draw circle outline
            cv2.circle(preview, (cx, cy), r, (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(preview, (cx, cy), 3, (0, 0, 255), -1)
            
            # Draw ID
            cv2.putText(preview, str(i+1), (cx-10, cy-r-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Create side-by-side view (mask + detection)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 1] = np.where(mask > 0, 200, 0)  # Green tint for gold
        mask_colored[:, :, 0] = np.where(mask > 0, 50, 0)   # Blue tint
        
        h, w = preview.shape[:2]
        combined = np.hstack([
            cv2.resize(mask_colored, (w//2, h//2)),
            cv2.resize(preview, (w//2, h//2))
        ])
        
        self._display_image(combined, self.preview_label, size=(800, 500))
        
        # Update results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Found {len(self.detected_pads)} gold pads:\n\n")
        
        for i, pad in enumerate(self.detected_pads):
            self.results_text.insert(tk.END, 
                f"#{i+1}: Center ({pad['center'][0]}, {pad['center'][1]})\n"
                f"     Radius: {pad['radius']}px\n"
                f"     Circularity: {pad['circularity']:.2f}\n\n")
        
        self.status_var.set(f"Detected {len(self.detected_pads)} gold pads")
    
    def _extract_pads(self):
        """Extract individual gold pad images (masked only, no white background)."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if not self.detected_pads:
            # Run detection first
            self._preview_detection()
            if not self.detected_pads:
                messagebox.showinfo("No Pads", "No gold pads detected. Adjust HSV settings.")
                return
        
        self.status_var.set("Extracting gold pads...")
        self.update_idletasks()
        
        self.extracted_pads = []
        h, w = self.current_image.shape[:2]
        
        for i, pad in enumerate(self.detected_pads):
            cx, cy = pad['center']
            r = pad['radius']
            
            # Calculate bounding box with padding
            padding = 5
            x1 = max(0, cx - r - padding)
            y1 = max(0, cy - r - padding)
            x2 = min(w, cx + r + padding)
            y2 = min(h, cy + r + padding)
            
            # Extract region
            pad_image = self.current_image[y1:y2, x1:x2].copy()
            
            # Create circular mask
            mask = np.zeros(pad_image.shape[:2], dtype=np.uint8)
            local_cx = cx - x1
            local_cy = cy - y1
            cv2.circle(mask, (local_cx, local_cy), r, 255, -1)
            
            # Apply mask (transparent background)
            pad_masked = cv2.bitwise_and(pad_image, pad_image, mask=mask)
            
            self.extracted_pads.append({
                'id': i + 1,
                'image': pad_masked,  # Only masked version now
                'center': (cx, cy),
                'radius': r,
                'mask': mask
            })
        
        # Update gallery
        self._update_gallery()
        
        self.results_text.insert(tk.END, f"\n=== EXTRACTED {len(self.extracted_pads)} PADS ===\n")
        self.results_text.insert(tk.END, "Click 'SAVE EXTRACTED PADS' to save.\n")
        
        self.status_var.set(f"Extracted {len(self.extracted_pads)} gold pads")
    
    def _update_gallery(self):
        """Update the gallery canvas with extracted pads."""
        self.gallery_canvas.delete("all")
        
        if not self.extracted_pads:
            return
        
        # Display thumbnails
        x_offset = 10
        thumb_size = 80
        
        for pad in self.extracted_pads[:15]:  # Show first 15
            img = pad['image']
            
            # Resize for thumbnail
            h, w = img.shape[:2]
            scale = min(thumb_size/w, thumb_size/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            thumb = cv2.resize(img, (new_w, new_h))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            
            photo = ImageTk.PhotoImage(image=Image.fromarray(thumb_rgb))
            
            self.gallery_canvas.create_image(x_offset, 10, anchor=tk.NW, image=photo)
            self.gallery_canvas.create_text(x_offset + new_w//2, new_h + 20, 
                                           text=f"#{pad['id']}", fill=self.FG_COLOR)
            
            # Keep reference
            if not hasattr(self, 'gallery_photos'):
                self.gallery_photos = []
            self.gallery_photos.append(photo)
            
            x_offset += new_w + 15
    
    def _save_pads(self):
        """Save extracted gold pads (masked only with transparency)."""
        if not self.extracted_pads:
            messagebox.showwarning("No Pads", "No extracted pads to save. Run extraction first.")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pads_folder = os.path.join(output_dir, f"gold_pads_{timestamp}")
            os.makedirs(pads_folder, exist_ok=True)
            
            self.status_var.set("Saving extracted pads...")
            self.update_idletasks()
            
            saved_files = []
            
            for pad in self.extracted_pads:
                # Save only masked version with transparency (RGBA)
                filename = f"pad_{pad['id']:03d}.png"
                filepath = os.path.join(pads_folder, filename)
                
                # Create RGBA image with transparency
                b, g, r = cv2.split(pad['image'])
                alpha = pad['mask']
                rgba = cv2.merge([b, g, r, alpha])
                cv2.imwrite(filepath, rgba)
                saved_files.append(filename)
            
            # Save summary
            summary_path = os.path.join(pads_folder, "extraction_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("GOLD PAD EXTRACTION SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total pads extracted: {len(self.extracted_pads)}\n")
                f.write(f"HSV Range: H({self.hue_low.get()}-{self.hue_high.get()}), ")
                f.write(f"S({self.sat_low.get()}-{self.sat_high.get()}), ")
                f.write(f"V({self.val_low.get()}-{self.val_high.get()})\n\n")
                
                for pad in self.extracted_pads:
                    f.write(f"Pad #{pad['id']}: Center ({pad['center'][0]}, {pad['center'][1]}), R={pad['radius']}px\n")
            
            self.status_var.set(f"Saved {len(saved_files)} pads to: {pads_folder}")
            messagebox.showinfo("Saved", 
                f"Saved {len(self.extracted_pads)} gold pads!\n\n"
                f"Location: {pads_folder}\n\n"
                f"Each pad saved as:\n"
                f"• pad_XXX.png (transparent background)")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Save Error", str(e))
    
    def _display_image(self, cv2_image: np.ndarray, label: tk.Label, size=(600, 400)):
        """Display image in the label."""
        if cv2_image is None or cv2_image.size == 0:
            return
        
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w == 0 or new_h == 0:
            return
        
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        
        label.config(image=photo)
        label.image = photo


# ==============================================================================
# RED PAD EXTRACTOR WINDOW
# ==============================================================================

class RedPadExtractorWindow(tk.Toplevel):
    """Red Pad Extraction Tool - Extract red circles from PCB strips using HSV filtering.
    
    Red in HSV wraps around 0, so we use two ranges: 0-10 and 160-179.
    Output files are saved with '_red_padding' suffix.
    """
    
    BG_COLOR = "#000000"
    FG_COLOR = "#FF4444"  # Red color
    ACCENT_COLOR = "#FF4444"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("Red Pad Extractor")
        self.geometry("1200x800")
        self.configure(bg=self.BG_COLOR)
        
        self.current_image = None
        self.detected_pads = []
        self.extracted_pads = []
        self.preview_image = None
        
        # Default HSV range for red (two ranges: 0-10 and 160-179)
        self.hue_low1 = tk.IntVar(value=0)
        self.hue_high1 = tk.IntVar(value=10)
        self.hue_low2 = tk.IntVar(value=160)
        self.hue_high2 = tk.IntVar(value=179)
        self.sat_low = tk.IntVar(value=70)
        self.sat_high = tk.IntVar(value=255)
        self.val_low = tk.IntVar(value=50)
        self.val_high = tk.IntVar(value=255)
        
        # Circle detection params
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=100)
        self.min_circularity = tk.DoubleVar(value=0.7)
        
        self._build_menu()
        self._build_ui()
    
    def _build_menu(self):
        """Build menu bar with navigation."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Navigate menu
        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Navigate", menu=nav_menu)
        nav_menu.add_command(label="🏠 Home (Inspector)", command=self._go_home)
        nav_menu.add_separator()
        nav_menu.add_command(label="✕ Close Window", command=self.destroy)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🔲 QR Code Extractor", command=lambda: self.parent._open_qr_cropper())
        tools_menu.add_command(label="🔶 Gold Pad Extractor", command=lambda: self.parent._open_gold_pad_extractor())
        tools_menu.add_command(label="🔴 Red Pad Extractor", command=lambda: self.lift())
        tools_menu.add_command(label="🔍 Simple Defect Detection", command=lambda: self.parent._open_simple_defect_detection())
        tools_menu.add_separator()
        tools_menu.add_command(label="📦 Batch Inspection", command=lambda: self.parent._run_batch_inspection())
        tools_menu.add_separator()
        tools_menu.add_command(label="📄 Open Log File", command=lambda: self.parent._open_log_file())
    
    def _go_home(self):
        """Bring main Inspector window to front."""
        self.parent.lift()
        self.parent.focus_force()
    
    def _build_ui(self):
        """Build the Red Pad Extractor UI."""
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=320)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(controls_frame, text="[ RED PAD EXTRACTOR ]",
                              font=(self.FONT_FACE, 14, 'bold'),
                              bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
        # Load button
        load_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                             highlightbackground=self.FG_COLOR, highlightthickness=1)
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Image",
                  command=self._load_image).pack(fill=tk.X, padx=5, pady=5)
        
        self.image_status = ttk.Label(load_frame, text="No image loaded", 
                                      foreground="#888888")
        self.image_status.pack(pady=(0, 5))
        
        # HSV Settings - Red has TWO hue ranges
        hsv_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                            highlightbackground=self.FG_COLOR, highlightthickness=1)
        hsv_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(hsv_frame, text="< HSV RED FILTER >", foreground=self.ACCENT_COLOR).pack(pady=5)
        ttk.Label(hsv_frame, text="(Red uses dual hue ranges)", foreground="#888888").pack()
        
        # Hue range 1 (0-10)
        ttk.Label(hsv_frame, text="--- Range 1 ---", foreground="#AAAAAA").pack(pady=2)
        self._add_slider(hsv_frame, "Hue1 Low:", self.hue_low1, 0, 30)
        self._add_slider(hsv_frame, "Hue1 High:", self.hue_high1, 0, 30)
        
        # Hue range 2 (160-179)
        ttk.Label(hsv_frame, text="--- Range 2 ---", foreground="#AAAAAA").pack(pady=2)
        self._add_slider(hsv_frame, "Hue2 Low:", self.hue_low2, 150, 179)
        self._add_slider(hsv_frame, "Hue2 High:", self.hue_high2, 150, 179)
        
        # Saturation and Value
        ttk.Label(hsv_frame, text="--- Sat/Val ---", foreground="#AAAAAA").pack(pady=2)
        self._add_slider(hsv_frame, "Sat Low:", self.sat_low, 0, 255)
        self._add_slider(hsv_frame, "Sat High:", self.sat_high, 0, 255)
        self._add_slider(hsv_frame, "Val Low:", self.val_low, 0, 255)
        self._add_slider(hsv_frame, "Val High:", self.val_high, 0, 255)
        
        # Circle detection settings
        circle_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        circle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(circle_frame, text="< CIRCLE DETECTION >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        self._add_slider(circle_frame, "Min Radius:", self.min_radius, 5, 200)
        self._add_slider(circle_frame, "Max Radius:", self.max_radius, 10, 300)
        
        # Action buttons
        action_frame = tk.Frame(controls_frame, bg=self.BG_COLOR)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        
        preview_btn = tk.Button(action_frame, text="👁 PREVIEW DETECTION",
                               font=(self.FONT_FACE, 11, 'bold'),
                               bg="#440000", fg="#FF8888",
                               command=self._preview_detection)
        preview_btn.pack(fill=tk.X, pady=3)
        
        extract_btn = tk.Button(action_frame, text="⭕ EXTRACT RED PADS",
                               font=(self.FONT_FACE, 11, 'bold'),
                               bg="#440000", fg=self.FG_COLOR,
                               command=self._extract_pads)
        extract_btn.pack(fill=tk.X, pady=3)
        
        save_btn = tk.Button(action_frame, text="💾 SAVE EXTRACTED PADS",
                            font=(self.FONT_FACE, 11, 'bold'),
                            bg="#440044", fg="#FF88FF",
                            command=self._save_pads)
        save_btn.pack(fill=tk.X, pady=3)
        
        # Results
        result_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(result_frame, text="< RESULTS >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        self.results_text = tk.Text(result_frame, bg="#111111", fg=self.FG_COLOR,
                                   font=(self.FONT_FACE, 9), height=8, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Image displays
        display_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Original/Preview
        preview_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(preview_frame, text="PREVIEW (Red Mask / Detection)").pack()
        self.preview_label = tk.Label(preview_frame, bg="#111111")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Extracted pads gallery
        gallery_frame = tk.Frame(display_frame, bg=self.BG_COLOR, height=200)
        gallery_frame.pack(fill=tk.X, pady=10)
        gallery_frame.pack_propagate(False)
        
        ttk.Label(gallery_frame, text="EXTRACTED RED PADS").pack()
        
        self.gallery_canvas = tk.Canvas(gallery_frame, bg="#111111", height=150)
        self.gallery_canvas.pack(fill=tk.X, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to extract red pads")
        status_bar = tk.Label(self, textvariable=self.status_var,
                             bg="#222222", fg=self.FG_COLOR,
                             font=(self.FONT_FACE, 9), anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _add_slider(self, parent, label, variable, from_, to):
        """Add a labeled slider."""
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        slider = ttk.Scale(row, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        value_label = ttk.Label(row, textvariable=variable, width=4)
        value_label.pack(side=tk.LEFT)
    
    def _load_image(self):
        """Load an image for red pad extraction."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                self.image_status.config(text=os.path.basename(path), 
                                        foreground=self.FG_COLOR)
                self._display_image(self.current_image, self.preview_label)
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded.\n\nClick 'PREVIEW DETECTION' to see red mask.\nAdjust HSV sliders if needed.\nThen click 'EXTRACT RED PADS'.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _get_red_mask(self, image):
        """Create HSV mask for red regions (dual hue range)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Range 1: low reds (0-10)
        lower1 = np.array([self.hue_low1.get(), self.sat_low.get(), self.val_low.get()])
        upper1 = np.array([self.hue_high1.get(), self.sat_high.get(), self.val_high.get()])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Range 2: high reds (160-179)
        lower2 = np.array([self.hue_low2.get(), self.sat_low.get(), self.val_low.get()])
        upper2 = np.array([self.hue_high2.get(), self.sat_high.get(), self.val_high.get()])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Combine both ranges
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _detect_circles(self, mask):
        """Detect circular red pads from the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        min_r = self.min_radius.get()
        max_r = self.max_radius.get()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if min_r <= radius <= max_r:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity >= self.min_circularity.get():
                        circles.append({
                            'center': (int(x), int(y)),
                            'radius': int(radius),
                            'area': area,
                            'circularity': circularity,
                            'contour': contour
                        })
        
        circles.sort(key=lambda c: (c['center'][1] // 50, c['center'][0]))
        return circles
    
    def _preview_detection(self):
        """Preview the red mask and detected circles."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.status_var.set("Detecting red pads...")
        self.update_idletasks()
        
        mask = self._get_red_mask(self.current_image)
        self.detected_pads = self._detect_circles(mask)
        
        preview = self.current_image.copy()
        
        for i, pad in enumerate(self.detected_pads):
            cx, cy = pad['center']
            r = pad['radius']
            cv2.circle(preview, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(preview, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(preview, str(i+1), (cx-10, cy-r-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Side-by-side view
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 2] = np.where(mask > 0, 200, 0)  # Red tint
        mask_colored[:, :, 0] = np.where(mask > 0, 50, 0)
        
        h, w = preview.shape[:2]
        combined = np.hstack([
            cv2.resize(mask_colored, (w//2, h//2)),
            cv2.resize(preview, (w//2, h//2))
        ])
        
        self._display_image(combined, self.preview_label, size=(800, 500))
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Found {len(self.detected_pads)} red pads:\n\n")
        
        for i, pad in enumerate(self.detected_pads):
            self.results_text.insert(tk.END, 
                f"#{i+1}: Center ({pad['center'][0]}, {pad['center'][1]})\n"
                f"     Radius: {pad['radius']}px\n"
                f"     Circularity: {pad['circularity']:.2f}\n\n")
        
        self.status_var.set(f"Detected {len(self.detected_pads)} red pads")
    
    def _extract_pads(self):
        """Extract individual red pad images."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if not self.detected_pads:
            self._preview_detection()
            if not self.detected_pads:
                messagebox.showinfo("No Pads", "No red pads detected. Adjust HSV settings.")
                return
        
        self.status_var.set("Extracting red pads...")
        self.update_idletasks()
        
        self.extracted_pads = []
        h, w = self.current_image.shape[:2]
        
        for i, pad in enumerate(self.detected_pads):
            cx, cy = pad['center']
            r = pad['radius']
            
            padding = 5
            x1 = max(0, cx - r - padding)
            y1 = max(0, cy - r - padding)
            x2 = min(w, cx + r + padding)
            y2 = min(h, cy + r + padding)
            
            pad_image = self.current_image[y1:y2, x1:x2].copy()
            
            mask = np.zeros(pad_image.shape[:2], dtype=np.uint8)
            local_cx = cx - x1
            local_cy = cy - y1
            cv2.circle(mask, (local_cx, local_cy), r, 255, -1)
            
            pad_masked = cv2.bitwise_and(pad_image, pad_image, mask=mask)
            
            white_bg = np.ones_like(pad_image) * 255
            white_bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
            pad_on_white = cv2.add(pad_masked, white_bg)
            
            self.extracted_pads.append({
                'id': i + 1,
                'image': pad_on_white,
                'image_no_bg': pad_masked,
                'center': (cx, cy),
                'radius': r,
                'mask': mask
            })
        
        self._update_gallery()
        
        self.results_text.insert(tk.END, f"\n=== EXTRACTED {len(self.extracted_pads)} PADS ===\n")
        self.results_text.insert(tk.END, "Click 'SAVE EXTRACTED PADS' to save.\n")
        
        self.status_var.set(f"Extracted {len(self.extracted_pads)} red pads")
    
    def _update_gallery(self):
        """Update the gallery canvas with extracted pads."""
        self.gallery_canvas.delete("all")
        
        if not self.extracted_pads:
            return
        
        x_offset = 10
        thumb_size = 80
        
        for pad in self.extracted_pads[:15]:
            img = pad['image']
            
            h, w = img.shape[:2]
            scale = min(thumb_size/w, thumb_size/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            thumb = cv2.resize(img, (new_w, new_h))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            
            photo = ImageTk.PhotoImage(image=Image.fromarray(thumb_rgb))
            
            self.gallery_canvas.create_image(x_offset, 10, anchor=tk.NW, image=photo)
            self.gallery_canvas.create_text(x_offset + new_w//2, new_h + 20, 
                                           text=f"#{pad['id']}", fill=self.FG_COLOR)
            
            if not hasattr(self, 'gallery_photos'):
                self.gallery_photos = []
            self.gallery_photos.append(photo)
            
            x_offset += new_w + 15
    
    def _save_pads(self):
        """Save extracted red pads to a folder with _red_padding suffix."""
        if not self.extracted_pads:
            messagebox.showwarning("No Pads", "No extracted pads to save. Run extraction first.")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pads_folder = os.path.join(output_dir, f"red_pads_{timestamp}")
            os.makedirs(pads_folder, exist_ok=True)
            
            self.status_var.set("Saving extracted red pads...")
            self.update_idletasks()
            
            saved_files = []
            
            for pad in self.extracted_pads:
                # Save with white background - using _red_padding suffix
                filename = f"pad_{pad['id']:03d}_red_padding.png"
                filepath = os.path.join(pads_folder, filename)
                cv2.imwrite(filepath, pad['image'])
                saved_files.append(filename)
                
                # Also save circular crop (transparent background)
                filename_alpha = f"pad_{pad['id']:03d}_red_padding_masked.png"
                filepath_alpha = os.path.join(pads_folder, filename_alpha)
                
                b, g, r = cv2.split(pad['image_no_bg'])
                alpha = pad['mask']
                rgba = cv2.merge([b, g, r, alpha])
                cv2.imwrite(filepath_alpha, rgba)
            
            # Save summary
            summary_path = os.path.join(pads_folder, "extraction_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("RED PAD EXTRACTION SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total pads extracted: {len(self.extracted_pads)}\n")
                f.write(f"Hue Range 1: {self.hue_low1.get()}-{self.hue_high1.get()}\n")
                f.write(f"Hue Range 2: {self.hue_low2.get()}-{self.hue_high2.get()}\n")
                f.write(f"Saturation: {self.sat_low.get()}-{self.sat_high.get()}\n")
                f.write(f"Value: {self.val_low.get()}-{self.val_high.get()}\n\n")
                
                for pad in self.extracted_pads:
                    f.write(f"Pad #{pad['id']}: Center ({pad['center'][0]}, {pad['center'][1]}), R={pad['radius']}px\n")
            
            self.status_var.set(f"Saved {len(saved_files)} red pads to: {pads_folder}")
            messagebox.showinfo("Saved", 
                f"Saved {len(self.extracted_pads)} red pads!\n\n"
                f"Location: {pads_folder}\n\n"
                f"Each pad saved as:\n"
                f"• pad_XXX_red_padding.png (white background)\n"
                f"• pad_XXX_red_padding_masked.png (transparent)")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Save Error", str(e))
    
    def _display_image(self, cv2_image: np.ndarray, label: tk.Label, size=(600, 400)):
        """Display image in the label."""
        if cv2_image is None or cv2_image.size == 0:
            return
        
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w == 0 or new_h == 0:
            return
        
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        
        label.config(image=photo)
        label.image = photo


# ==============================================================================
# PRESET CONFIGURATION WINDOW
# ==============================================================================

class PresetConfigWindow(tk.Toplevel):
    """Popup window to configure defect presets."""
    
    def __init__(self, parent, presets, on_save_callback):
        super().__init__(parent)
        self.title("Configure Defect Presets")
        self.geometry("450x500")
        self.transient(parent)
        self.grab_set()
        
        self.parent = parent
        self.presets = presets.copy() # Work on a copy
        self.on_save = on_save_callback
        
        # Style
        self.configure(bg="#1e1e1e")
        self.ACCENT = "#00FFFF"
        
        # Variables
        self.current_preset = tk.StringVar(value="Scratches")
        self.sigma = tk.DoubleVar()
        self.thresh = tk.DoubleVar()
        self.block = tk.IntVar()
        self.c_val = tk.IntVar()
        
        self._build_ui()
        self._load_preset_values()
        
    def _build_ui(self):
        # Header
        tk.Label(self, text="Customize Preset Definitions", bg="#1e1e1e", fg=self.ACCENT,
                font=("Segoe UI", 12, "bold")).pack(pady=10)
        
        # Preset Selector
        sel_frame = tk.Frame(self, bg="#1e1e1e")
        sel_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(sel_frame, text="Select Preset to Edit:", bg="#1e1e1e", fg="white").pack(anchor=tk.W)
        
        # Exclude 'Custom' from being editable defined presets
        editable_presets = [k for k in self.presets.keys() if k != "Custom"]
        self.combo = ttk.Combobox(sel_frame, textvariable=self.current_preset, 
                                 values=editable_presets, state="readonly")
        self.combo.pack(fill=tk.X, pady=5)
        self.combo.bind("<<ComboboxSelected>>", self._load_preset_values)
        
        # Sliders Frame
        controls_frame = tk.LabelFrame(self, text="Parameters", bg="#1e1e1e", fg="#aaaaaa")
        controls_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Sigma
        self._add_slider(controls_frame, "Sigma (Blur)", self.sigma, 0.0, 5.0, 0.1)
        
        # Threshold
        self._add_slider(controls_frame, "Threshold (Global)", self.thresh, 0.0, 1.0, 0.05)
        
        # Block Size
        self._add_slider(controls_frame, "Block Size (Adaptive)", self.block, 3, 51, 2)
        
        # C Constant
        self._add_slider(controls_frame, "C Constant (Adaptive)", self.c_val, -10, 20, 1)
        
        # Buttons
        btn_frame = tk.Frame(self, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Button(btn_frame, text="Save Changes", command=self._save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self._reset_defaults).pack(side=tk.LEFT)

    def _add_slider(self, parent, label, variable, min_val, max_val, step):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        lbl_frame = tk.Frame(frame, bg="#1e1e1e")
        lbl_frame.pack(fill=tk.X)
        tk.Label(lbl_frame, text=label, bg="#1e1e1e", fg="#dddddd").pack(side=tk.LEFT)
        val_lbl = tk.Label(lbl_frame, textvariable=variable, bg="#1e1e1e", fg=self.ACCENT)
        val_lbl.pack(side=tk.RIGHT)
        
        s = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable)
        s.pack(fill=tk.X)
        
        # Enforce step/odd logic if needed
        if variable == self.block:
            s.configure(command=lambda v: self._snap_odd(v, variable))
        
    def _snap_odd(self, val, var):
        v = int(float(val))
        if v % 2 == 0: v += 1
        var.set(v)
        
    def _load_preset_values(self, event=None):
        name = self.current_preset.get()
        if name in self.presets:
            p = self.presets[name]
            self.sigma.set(p["sigma"])
            self.thresh.set(p["thresh"])
            self.block.set(p["block"])
            self.c_val.set(p["c"])
            
    def _save(self):
        # Update current working copy
        name = self.current_preset.get()
        self.presets[name] = {
            "sigma": round(self.sigma.get(), 2),
            "thresh": round(self.thresh.get(), 2),
            "block": int(self.block.get()),
            "c": int(self.c_val.get())
        }
        # Pass back to parent
        self.on_save(self.presets)
        self.destroy()
        
    def _reset_defaults(self):
        # Reset current preset hardcoded defaults (simplified for now)
        from tkinter import messagebox
        if messagebox.askyesno("Reset", "Reset this preset to original factory settings?"):
            # Need access to original factory defaults. 
            # Ideally passed in or static. For now, just a placeholder or minimal logic.
            pass


# ==============================================================================
# SIMPLE DEFECT DETECTION WINDOW
# ==============================================================================

class InspectorApp(tk.Tk):
    """Integrated PCB Inspector (based on Simple Defect Detection)."""
    

    
    BG_COLOR = "#000000"
    FG_COLOR = "#00FFFF"  # Cyan
    ACCENT_COLOR = "#00FFFF"
    FONT_FACE = "Consolas"
    
    # Preset Definitions (now includes noise and defect_pct)
    DEFECT_PRESETS = {
        "General":   {"sigma": 1.2, "thresh": 0.65, "block": 11, "c": 2, "noise": 3, "defect_pct": 10.0},
        "Scratches": {"sigma": 0.8, "thresh": 0.60, "block": 7,  "c": 2, "noise": 2, "defect_pct": 5.0},
        "Stains":    {"sigma": 2.0, "thresh": 0.70, "block": 25, "c": 4, "noise": 4, "defect_pct": 15.0},
        "Pinholes":  {"sigma": 0.8, "thresh": 0.60, "block": 9,  "c": 5, "noise": 2, "defect_pct": 3.0}
    }
    
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".bitmap", ".dib")
    
    def __init__(self):
        super().__init__()
        
        self.title("Integrated PCB Inspector")
        self.geometry("1600x1000")
        self.state('zoomed')
        self.configure(bg=self.BG_COLOR)
        
        # State
        self.input_folder = tk.StringVar(value="")
        self.output_folder = tk.StringVar(value="")
        self.sigma = tk.DoubleVar(value=1.2)
        self.thresh = tk.DoubleVar(value=0.65)
        self.use_otsu = tk.BooleanVar(value=True)  # Auto Otsu mode ON by default
        self.use_adaptive = tk.BooleanVar(value=False)  # Adaptive thresholding
        self.adaptive_block_size = tk.IntVar(value=11)  # Block size for adaptive
        self.adaptive_c = tk.IntVar(value=2)  # C constant for adaptive
        self.black_defect_pct = tk.DoubleVar(value=10.0)
        self.noise_level = tk.IntVar(value=3)  # Morphological noise reduction (0-10)
        
        # New Feature State
        self.filter_method = tk.StringVar(value="Gaussian")
        self.use_clahe = tk.BooleanVar(value=False)
        self.show_overlay = tk.BooleanVar(value=True)
        self.show_auto_in_list = tk.BooleanVar(value=False)
        self.auto_defects = []
        self.manual_labels = []
        self.idx = 0
        self.results = []  # Batch results
        
        self.current_image = None
        self.current_alpha = None
        self.current_bw = None
        
        # Canvas metadata for coordinate tracking: {canvas: (ratio, off_x, off_y, orig_w, orig_h)}
        self.canvas_meta = {}
        
        self.tool_windows = {} # Track open tool windows
        self.session_file = "inspector_session.json"
        
        self._setup_styles()
        self._build_menu()
        self._build_ui()
    
    def _setup_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Base styles
        style.configure("TLabel", background=self.BG_COLOR, foreground=self.FG_COLOR,
                       font=(self.FONT_FACE, 10))
        style.configure("TButton", background="#333333", foreground=self.FG_COLOR,
                       font=(self.FONT_FACE, 10, 'bold'))
        style.configure("TFrame", background=self.BG_COLOR)
        style.configure("TCheckbutton", background=self.BG_COLOR, foreground=self.FG_COLOR)
        
        # Blue (Cyan) button style for folder selection
        style.configure("Blue.TButton", 
                       background="#0088AA", foreground="#FFFFFF",
                       font=(self.FONT_FACE, 10, 'bold'))
        style.map("Blue.TButton",
                 background=[("active", "#00AACC"), ("pressed", "#006688")])
    
    def _build_menu(self):
        """Build menu bar with navigation."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="📊 Pixel Inspection (Hiatus)", command=self._open_pixel_inspection_window)
        tools_menu.add_separator()
        tools_menu.add_command(label="🔲 QR Code Extractor", command=self._open_qr_cropper)
        tools_menu.add_command(label="🔶 Gold Pad Extractor", command=self._open_gold_pad_extractor)
        tools_menu.add_command(label="🔴 Red Pad Extractor", command=self._open_red_pad_extractor)
        tools_menu.add_separator()
        tools_menu.add_command(label="🌀 Texture Analysis (FFT)", command=self._open_texture_analysis)
        tools_menu.add_separator()
        tools_menu.add_command(label="📄 Open Log File", command=self._open_log_file)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _open_tool_window(self, name: str, WindowClass):
        """Open or focus a tool window (singleton pattern)."""
        if name in self.tool_windows:
            win = self.tool_windows[name]
            if win.winfo_exists():
                win.deiconify()
                win.lift()
                win.focus_force()
                return win
            else:
                del self.tool_windows[name]
        
        win = WindowClass(self)
        self.tool_windows[name] = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._on_tool_close(name))
        return win
    
    def _on_tool_close(self, name):
        """Handle tool window closure."""
        if name in self.tool_windows:
            win = self.tool_windows[name]
            win.destroy()
            del self.tool_windows[name]
            
    def _open_pixel_inspection_window(self):
        self._open_tool_window("Pixel Inspection", PixelInspectionWindow)

    def _open_qr_cropper(self):
        self._open_tool_window("QR Extractor", QRCropperWindow)

    def _open_gold_pad_extractor(self):
        self._open_tool_window("Gold Pad", GoldPadExtractorWindow)

    def _open_red_pad_extractor(self):
        self._open_tool_window("Red Pad", RedPadExtractorWindow)
    
    def _open_texture_analysis(self):
        self._open_tool_window("Texture Analysis", TextureAnalysisWindow)
        
    def _open_log_file(self):
        import subprocess
        log_file = "inspection_log.csv" # Should ideally be shared constant
        if os.path.exists(log_file):
            subprocess.run(['explorer', '/select,', os.path.abspath(log_file)])
            
    def _show_about(self):
        messagebox.showinfo("About", "Integrated PCB Inspector\n\nMain: Simple Defect Detection\nTools: Pixel Inspection, QR, etc.")
        
    def _on_close(self):
        """Handle application closure."""
        # Close all tools
        for win in list(self.tool_windows.values()):
            if win.winfo_exists():
                win.destroy()
        self.destroy()
        sys.exit(0)

    
    def _go_home(self):
        """Bring main Inspector window to front."""
        self.parent._show_home()
    
    def _build_ui(self):
        """Build the Simple Defect Detection UI."""
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls (Scrollable Container)
        # Left panel - Controls (Scrollable Container)
        controls_container = tk.Frame(main_frame, bg=self.BG_COLOR, width=420)
        controls_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_container.pack_propagate(False)

        # Create Canvas and Scrollbar
        self.controls_canvas = tk.Canvas(controls_container, bg=self.BG_COLOR, highlightthickness=0)
        self.controls_scrollbar = ttk.Scrollbar(controls_container, orient="vertical", command=self.controls_canvas.yview)
        
        # Scrollable Frame inside Canvas
        self.scroll_content = tk.Frame(self.controls_canvas, bg=self.BG_COLOR)
        
        # Configure Scrollbar
        self.controls_canvas.configure(yscrollcommand=self.controls_scrollbar.set)
        
        # Packing for Scrollbale Area
        self.controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Window in Canvas
        self.canvas_window = self.controls_canvas.create_window((0, 0), window=self.scroll_content, anchor="nw")
        
        # Bindings for resizing and scrolling
        self.scroll_content.bind("<Configure>", self._on_frame_configure)
        self.controls_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Mousewheel scrolling (bind to canvas in this area)
        self.controls_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Pointer to where we actually add widgets (renamed for clarity in diff, but using old var name to minimize changes below would be easier... 
        # actually let's re-assign controls_frame to point to our new inner frame so the rest of the code works with minimal changes)
        controls_frame = self.scroll_content
        
        # Title
        title_label = tk.Label(controls_frame, text="[ SIMPLE DEFECT DETECTION ]", 
                              font=(self.FONT_FACE, 13, 'bold'),
                              bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
        # Folder selection
        folder_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                               highlightbackground=self.FG_COLOR, highlightthickness=1)
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(folder_frame, text="< FOLDERS >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        ttk.Button(folder_frame, text="Select Input Folder", style="Blue.TButton",
                  command=self._select_input).pack(fill=tk.X, padx=5, pady=2)
        self.input_label = ttk.Label(folder_frame, text="Not selected", foreground="#888888")
        self.input_label.pack(pady=(0, 5))
        
        ttk.Button(folder_frame, text="Select Output Folder", style="Blue.TButton",
                  command=self._select_output).pack(fill=tk.X, padx=5, pady=2)
        self.output_label = ttk.Label(folder_frame, text="Auto (input/BW_out)", foreground="#888888")
        self.output_label.pack(pady=(0, 5))
        
        if self.current_image is not None and self.output_label.cget("text") == "Auto (input/BW_out)":
             self._get_output_dir()
            
        # Load presets
        self.presets_file = "defect_presets.json"
        self._load_presets()

        # Processing settings
        settings_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                                 highlightbackground=self.FG_COLOR, highlightthickness=1)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="< PROCESSING SETTINGS >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        # Defect Presets (NEW)
        self.preset_var = tk.StringVar(value="General")
        preset_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        preset_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(preset_row, text="Preset", width=12).pack(side=tk.LEFT)
        
        self.preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var, 
                                        values=list(self.DEFECT_PRESETS.keys()) + ["Custom"], 
                                        state="readonly", width=15)
        self.preset_combo.pack(side=tk.LEFT, padx=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)
        
        # Config Button (Gear)
        ttk.Button(preset_row, text="⚙", width=3, command=self._open_preset_config).pack(side=tk.LEFT)
        
        # Auto Otsu checkbox
        otsu_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        otsu_row.pack(fill=tk.X, padx=5, pady=3)
        self.otsu_check = ttk.Checkbutton(otsu_row, text="🔄 Auto (Triangle) - Global threshold",
                                          variable=self.use_otsu, command=self._on_threshold_mode_change)
        self.otsu_check.pack(anchor=tk.W)
        
        # Adaptive thresholding checkbox
        adaptive_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        adaptive_row.pack(fill=tk.X, padx=5, pady=3)
        self.adaptive_check = ttk.Checkbutton(adaptive_row, text="📐 Adaptive - Local threshold",
                                               variable=self.use_adaptive, command=self._on_threshold_mode_change)
        self.adaptive_check.pack(anchor=tk.W)
        
        # Adaptive settings (block size and C)
        adaptive_settings = tk.Frame(settings_frame, bg=self.BG_COLOR)
        adaptive_settings.pack(fill=tk.X, padx=5, pady=2)
        
        block_row = tk.Frame(adaptive_settings, bg=self.BG_COLOR)
        block_row.pack(fill=tk.X, pady=1)
        ttk.Label(block_row, text="Block Size:", width=10).pack(side=tk.LEFT)
        self.block_scale = ttk.Scale(block_row, from_=3, to=51, variable=self.adaptive_block_size,
                                     command=self._on_manual_change)
        self.block_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.block_label = ttk.Label(block_row, text="11", width=4)
        self.block_label.pack(side=tk.LEFT)
        
        c_row = tk.Frame(adaptive_settings, bg=self.BG_COLOR)
        c_row.pack(fill=tk.X, pady=1)
        ttk.Label(c_row, text="C Constant:", width=10).pack(side=tk.LEFT)
        self.c_scale = ttk.Scale(c_row, from_=-10, to=20, variable=self.adaptive_c,
                                 command=self._on_manual_change)
        self.c_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.c_label = ttk.Label(c_row, text="2", width=4)
        self.c_label.pack(side=tk.LEFT)
        
        # Filter and CLAHE
        filter_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        filter_row.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(filter_row, text="Filter:", width=10).pack(side=tk.LEFT)
        filter_cb = ttk.Combobox(filter_row, textvariable=self.filter_method, 
                             values=["Gaussian", "Bilateral", "Median", "None"], 
                             state="readonly", width=10)
        filter_cb.pack(side=tk.LEFT, padx=(0, 5))
        filter_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        
        ttk.Checkbutton(filter_row, text="CLAHE", variable=self.use_clahe, 
                       command=self._refresh_preview).pack(side=tk.LEFT)

        # Sigma slider
        sigma_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        sigma_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sigma_row, text="Sigma:", width=10).pack(side=tk.LEFT)
        self.sigma_scale = ttk.Scale(sigma_row, from_=0.0, to=8.0, variable=self.sigma,
                                     command=self._on_manual_change)
        self.sigma_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sigma_label = ttk.Label(sigma_row, text="1.20", width=5)
        self.sigma_label.pack(side=tk.LEFT)
        
        # Threshold slider
        thresh_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        thresh_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(thresh_row, text="Threshold:", width=10).pack(side=tk.LEFT)
        self.thresh_scale = ttk.Scale(thresh_row, from_=0.0, to=1.0, variable=self.thresh,
                                      command=self._on_manual_change)
        self.thresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.thresh_label = ttk.Label(thresh_row, text="0.65", width=5)
        self.thresh_label.pack(side=tk.LEFT)

        # Noise Reduction slider
        noise_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        noise_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(noise_row, text="Stabilization:", width=12).pack(side=tk.LEFT)
        self.noise_scale = ttk.Scale(noise_row, from_=0, to=10, variable=self.noise_level,
                                     command=self._on_manual_change)
        self.noise_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.noise_label = ttk.Label(noise_row, textvariable=self.noise_level, width=3)
        self.noise_label.pack(side=tk.LEFT)
        
        
        
        # [NEW] Color Filter Frame (Collapsible)
        self.color_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                                   highlightbackground=self.FG_COLOR, highlightthickness=1)
        self.color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header with Checkbox to Enable
        header_frame = tk.Frame(self.color_frame, bg=self.BG_COLOR)
        header_frame.pack(fill=tk.X, padx=2, pady=2)
        
        self.use_hsv = tk.BooleanVar(value=False)
        cb_hsv = ttk.Checkbutton(header_frame, text="COLOR FILTER (Gold Focus)", 
                                variable=self.use_hsv, command=self._on_hsv_toggle)
        cb_hsv.pack(side=tk.LEFT)
        
        # Preview Mask Button (small)
        self.btn_mask = tk.Button(header_frame, text="👁 Mask", 
                                 bg="#333333", fg="cyan", font=("Consolas", 8),
                                 command=self._preview_mask_only)
        self.btn_mask.pack(side=tk.RIGHT, padx=2)

        # Sliders container (Hidden by default until enabled)
        self.hsv_controls = tk.Frame(self.color_frame, bg=self.BG_COLOR)
        
        # Hue
        self.hue_min = tk.IntVar(value=10)
        self.hue_max = tk.IntVar(value=40)
        self._add_hsv_slider(self.hsv_controls, "Hue Min", self.hue_min, 0, 179)
        self._add_hsv_slider(self.hsv_controls, "Hue Max", self.hue_max, 0, 179)
        
        # Saturation
        self.sat_min = tk.IntVar(value=50)
        self.sat_max = tk.IntVar(value=255)
        self._add_hsv_slider(self.hsv_controls, "Sat Min", self.sat_min, 0, 255)
        
        # Value
        self.val_min = tk.IntVar(value=50)
        self.val_max = tk.IntVar(value=255)
        self._add_hsv_slider(self.hsv_controls, "Val Min", self.val_min, 0, 255)
        
        if self.use_hsv.get():
            self.hsv_controls.pack(fill=tk.X, padx=2)
        
        # Black% defect threshold

        defect_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        defect_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(defect_row, text="DEFECT if black% >").pack(side=tk.LEFT)
        self.defect_spin = ttk.Spinbox(defect_row, from_=0.0, to=100.0, increment=0.5,
                   textvariable=self.black_defect_pct, width=7)
        self.defect_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(defect_row, text="%").pack(side=tk.LEFT)
        
        # Apply initial state - MOVED TO END OF METHOD TO FIX CRASH
        # self._on_threshold_mode_change()
        
        # Navigation
        nav_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                            highlightbackground=self.FG_COLOR, highlightthickness=1)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(nav_frame, text="< PREVIEW >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        nav_btn_row = tk.Frame(nav_frame, bg=self.BG_COLOR)
        nav_btn_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(nav_btn_row, text="<< Prev", command=self._prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav_btn_row, text="Next >>", command=self._next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.nav_info = ttk.Label(nav_frame, text="No images loaded", foreground="#888888")
        self.nav_info.pack(pady=5)
        
        # Action buttons
        action_frame = tk.Frame(controls_frame, bg=self.BG_COLOR)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        
        process_btn = tk.Button(action_frame, text="▶ PROCESS ALL", 
                               font=(self.FONT_FACE, 12, 'bold'),
                               bg="#004400", fg="#00FF41",
                               command=self._process_all)
        process_btn.pack(fill=tk.X, pady=3)
        
        overview_btn = tk.Button(action_frame, text="📊 VIEW OVERVIEW", 
                                font=(self.FONT_FACE, 11, 'bold'),
                                bg="#440044", fg="#FF88FF",
                                command=self._open_overview)
        overview_btn.pack(fill=tk.X, pady=3)
        
        # Labeling and Visualization
        vis_frame = tk.Frame(controls_frame, bg=self.BG_COLOR,
                            highlightbackground=self.FG_COLOR, highlightthickness=1)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(vis_frame, text="< RESULTS & LABELS >", foreground=self.ACCENT_COLOR).pack(pady=5)
        
        # Result Verdict
        self.result_label = tk.Label(vis_frame, text="--", 
                                    font=(self.FONT_FACE, 14, 'bold'),
                                    bg=self.BG_COLOR, fg="#888888")
        self.result_label.pack(pady=5)
        
        # Labeling Buttons
        lbl_btn_row = tk.Frame(vis_frame, bg=self.BG_COLOR)
        lbl_btn_row.pack(fill=tk.X, padx=5)
        
        tk.Button(lbl_btn_row, text="🏷 Open Labeler", bg="#003300", fg="#00FF00",
                 command=self._open_labeler).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        vis_chk_row = tk.Frame(vis_frame, bg=self.BG_COLOR)
        vis_chk_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(vis_chk_row, text="Show Boxes", variable=self.show_overlay, 
                       command=self._refresh_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(vis_chk_row, text="List Auto", variable=self.show_auto_in_list, 
                       command=self._refresh_visualization).pack(side=tk.LEFT, padx=2)
        
        # Treeview for defects
        tree_frame = ttk.Frame(vis_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("ID", "Type", "Area")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8)
        self.tree.heading("ID", text="ID"); self.tree.column("ID", width=40)
        self.tree.heading("Type", text="Type"); self.tree.column("Type", width=80)
        self.tree.heading("Area", text="Area"); self.tree.column("Area", width=60)
        
        vbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Delete Button (User Request)
        tk.Button(vis_frame, text="Remove Selected Label", bg="#440000", fg="#FF0000",
                 command=self._remove_selected_label).pack(fill=tk.X, padx=5, pady=2)
        
        # Right Click Menu
        self.tree_menu = tk.Menu(self, tearoff=0)
        self.tree_menu.add_command(label="Remove Label", command=self._remove_selected_label)
        self.tree.bind("<Button-3>", lambda event: self.tree_menu.post(event.x_root, event.y_root))
        
        # Stats Text (Restored)
        self.stats_text = tk.Text(vis_frame, bg="#111111", fg=self.FG_COLOR,
                                 font=(self.FONT_FACE, 9), height=6, width=35)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Image display
        display_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Original image
        orig_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Label(orig_frame, text="ORIGINAL").pack()
        self.orig_canvas = tk.Canvas(orig_frame, bg="#111111", highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        self.orig_canvas.bind("<Motion>", self._on_mouse_move)
        self.orig_canvas.bind("<Leave>", self._on_mouse_leave)
        
        # Binary image
        bw_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        bw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Label(bw_frame, text="BINARY").pack()
        self.bw_canvas = tk.Canvas(bw_frame, bg="#111111", highlightthickness=0)
        self.bw_canvas.pack(fill=tk.BOTH, expand=True)
        self.bw_canvas.bind("<Motion>", self._on_mouse_move)
        self.bw_canvas.bind("<Leave>", self._on_mouse_leave)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select input folder to begin")
        
        status_frame = tk.Frame(self, bg="#222222")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_bar = tk.Label(status_frame, textvariable=self.status_var, 
                             bg="#222222", fg=self.FG_COLOR,
                             font=(self.FONT_FACE, 9), anchor=tk.W)
        status_bar.pack(side=tk.LEFT, padx=5)
        
        self.lbl_coords = tk.Label(status_frame, text="XY: -", 
                                  bg="#222222", fg="#FFFFFF",
                                  font=(self.FONT_FACE, 9))
        self.lbl_coords.pack(side=tk.RIGHT, padx=10)
        
        self._add_tooltips()

        # [Moved to end] Apply initial state after all widgets are created
        self._on_threshold_mode_change()
    
    def _add_tooltips(self):
        """Add tooltips to controls."""
        try:
            ToolTip(self.input_label, "Select folder containing images to inspect")
            ToolTip(self.otsu_check, "Automatically calculate global threshold using Otsu's method")
            ToolTip(self.adaptive_check, "Use local adaptive thresholding (better for uneven lighting)")
            ToolTip(self.block_scale, "Size of the local neighborhood for adaptive thresholding")
            ToolTip(self.c_scale, "Constant subtracted from the mean (Adaptive C)")
            ToolTip(self.sigma_scale, "Gaussian blur strength (Sigma) to reduce noise before thresholding")
            ToolTip(self.thresh_scale, "Manual global threshold value (0.0 - 1.0)")
            ToolTip(self.noise_scale, "Morphological Open/Close operations to remove small noise and fill gaps")
            ToolTip(self.preset_combo, "Quickly select parameter presets for different defect types")
            ToolTip(self.defect_spin, "Threshold percentage of black pixels to consider an image defective")
        except Exception as e:
            print(f"Error adding tooltips: {e}")

    def _on_frame_configure(self, event):
        """Update scroll region when content changes."""
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Ensure inner frame fills canvas width."""
        self.controls_canvas.itemconfig(self.canvas_window, width=event.width)
        
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        self.controls_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _add_hsv_slider(self, parent, label, variable, from_, to):
        """Add compact HSV slider."""
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, padx=2, pady=1)
        tk.Label(row, text=label, width=8, bg=self.BG_COLOR, fg="white", font=("Consolas", 8)).pack(side=tk.LEFT)
        s = ttk.Scale(row, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, command=self._on_hsv_change)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        l = ttk.Label(row, textvariable=variable, width=4, font=("Consolas", 8))
        l.pack(side=tk.LEFT)

    def _on_hsv_toggle(self):
        """Show/Hide HSV controls."""
        if self.use_hsv.get():
            self.hsv_controls.pack(fill=tk.X, padx=2)
        else:
            self.hsv_controls.pack_forget()
        self._refresh_preview()

    def _on_hsv_change(self, val):
        """Live update when sliding."""
        self._refresh_preview()

    def _preview_mask_only(self):
        """Show JUST the Gold Mask in the preview window for tuning."""
        if self.current_image is None: return
        
        hsv_mask = self._get_hsv_mask(self.current_image)
        
        # Show in Preview Canvas (Original side) temporarily
        self._display_on_canvas(hsv_mask, self.orig_canvas, is_gray=True)
        self.status_var.set("Showing HSV Mask (White = Keep, Black = Ignore). Move sliders to tune.")

    def _get_hsv_mask(self, rgb_img):
        """Compute the HSV mask based on current sliders."""
        # Convert RGB to BGR for OpenCV (if needed) or directly to HSV
        # Note: self.current_image is RGB from _read_image_with_alpha usually
        # But cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) works too.
        
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        
        lower = np.array([self.hue_min.get(), self.sat_min.get(), self.val_min.get()])
        upper = np.array([self.hue_max.get(), self.sat_max.get(), self.val_max.get()])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Optional cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask


    
    # === UTILITY FUNCTIONS (from pad_binary_gui.py) ===
    
    def _load_presets(self):
        """Load presets from JSON file if exists, else use defaults."""
        if os.path.exists(self.presets_file):
            try:
                import json
                with open(self.presets_file, 'r') as f:
                    saved = json.load(f)
                    # Merge valid saved presets into defaults
                    for k, v in saved.items():
                        if k in self.DEFECT_PRESETS:
                            self.DEFECT_PRESETS[k] = v
                print(f"Loaded presets from {self.presets_file}")
            except Exception as e:
                print(f"Error loading presets: {e}")
    
    def _save_presets(self, new_presets):
        """Save presets to JSON file and update current."""
        import json
        self.DEFECT_PRESETS = new_presets
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.DEFECT_PRESETS, f, indent=4)
            print(f"saved presets to {self.presets_file}")
        except Exception as e:
            print(f"Error saving presets: {e}")
            
        # Refresh current view if currently using a preset
        curr = self.preset_var.get()
        if curr != "Custom":
            self._on_preset_change()

    def _open_preset_config(self):
        """Open the preset configuration dialog."""
        PresetConfigWindow(self, self.DEFECT_PRESETS, self._save_presets)

    def _list_images(self, folder):
        """List all supported image files in folder."""
        import glob
        files = []
        for ext in self.SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
            files.extend(glob.glob(os.path.join(folder, "**", f"*{ext.upper()}"), recursive=True))
        return sorted(set(files))
    
    def _read_image_with_alpha(self, path):
        """Read image, return (rgb, alpha)."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            alpha = None
        elif img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            bgr = img[:, :, :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            alpha = None
        return rgb, alpha
    
    def _masked_gaussian_smooth(self, gray01, mask01, sigma):
        """Normalized masked Gaussian (avoid boundary bleeding)."""
        if sigma <= 0:
            out = gray01.copy()
            out[mask01 <= 0] = 0.0
            return out
        
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        k = max(k, 3)
        
        num = cv2.GaussianBlur(gray01 * mask01, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        den = cv2.GaussianBlur(mask01, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        out = num / (den + 1e-8)
        out[mask01 <= 0] = 0.0
        return out
    
    def _compute_otsu_threshold(self, gray_u8, mask_bool=None):
        """Compute Otsu's optimal threshold, normalized to 0-1."""
        if mask_bool is not None:
            masked_pixels = gray_u8[mask_bool]
            if len(masked_pixels) == 0:
                return 0.5
            otsu_thresh, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            otsu_thresh, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_thresh / 255.0
    
    def _compute_auto_sigma(self, gray_u8, mask_bool=None):
        """Compute auto sigma based on noise estimation."""
        if mask_bool is not None:
            masked_gray = gray_u8.copy()
            masked_gray[~mask_bool] = 0
        else:
            masked_gray = gray_u8
        
        laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
        noise_var = laplacian.var()
        
        if noise_var < 50:
            sigma = 0.5
        elif noise_var > 500:
            sigma = 5.0
        else:
            sigma = 0.5 + (noise_var - 50) / 450 * 4.5
        return round(sigma, 2)
    
    def _make_binary(self, rgb, alpha, sigma=1.2, thresh=0.65, use_otsu=False, 
                      use_adaptive=False, block_size=11, c_value=2, noise_level=0,
                      use_hsv=False):
        """RGB -> Gray -> Filter (Gaussian/Bilateral/Median) -> Threshold -> BW."""
        
        # 0. HSV Masking (Gold Focus)
        # If enabled, we create a mask of "valid" areas (Gold) and "invalid" (Green Background)
        # We will use this to zero out the background AFTER thresholding.
        gold_mask = None
        if use_hsv:
            # We need to call generic helper or reimplement
            # Since _get_hsv_mask is instance method, we can't call it if this is static
            # But this is an instance method, so we can!
            # BUT: _make_binary is called by _quick_process/batch which might NOT be on 'self' context cleanly
            # Wait, it IS an instance method.
            gold_mask = self._get_hsv_mask(rgb)

        # 1. Grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.use_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
        gray_f = gray.astype(np.float32) / 255.0
        
        # 3. Mask creation
        if alpha is not None:
            mask_bool = alpha > 0
        else:
            mask_bool = np.ones_like(gray, dtype=bool)
        mask01 = mask_bool.astype(np.float32)
        gray_u8 = np.clip(gray_f * 255.0, 0, 255).astype(np.uint8)
        
        # 4. Smoothing / Filtering
        f_method = self.filter_method.get()
        if f_method == "Gaussian":
            smooth_f = masked_gaussian_smooth(gray_f, mask01, float(sigma))
        elif f_method == "Bilateral":
            smooth_f = masked_bilateral_smooth(gray_f, mask01, float(sigma))
        elif f_method == "Median":
            k = int(float(sigma) * 3)
            if k % 2 == 0: k += 1
            smooth_f = masked_median_smooth(gray_f, mask01, kernel_size=k)
        else: # None
            smooth_f = gray_f

        smooth_u8 = np.clip(smooth_f * 255.0, 0, 255).astype(np.uint8)
        
        auto_params = None
        
        # 5. Thresholding
        if use_adaptive:
            # Adaptive thresholding
            block = int(block_size)
            if block % 2 == 0: block += 1
            block = max(3, block)
            
            bw = cv2.adaptiveThreshold(
                smooth_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, int(c_value)
            )
            bw[~mask_bool] = 0
            
            # Apply Gold Mask logic:
            # We want to FIND DEFECTS (white pixels in 'bw')
            # But ONLY if they are inside the GOLD area (white pixels in 'gold_mask')
            # Actually, defects are usually "dark" spots on gold.
            # Local adaptive threshold: finds local variations.
            # If we mask out the green background (make it black), adaptive threshold might find edges there.
            # SO: We should mask the result.
            
            auto_params = (sigma, f"adapt({block},{c_value})")
            
        elif use_otsu:
            # For Otsu, we can use the helper or just standard CV2
            # Re-using local Otsu logic for consistency with previous implementation
            masked_pixels = smooth_u8[mask_bool] # Only look at non-transparent pixels
            
            if use_hsv and gold_mask is not None:
                 # If focusing on gold, only calculate Otsu on the gold pixels!
                 # This makes thresholding much more accurate for the pads.
                 gold_pixels = smooth_u8[gold_mask > 0]
                 if len(gold_pixels) > 0:
                     masked_pixels = gold_pixels

            if len(masked_pixels) == 0:
                thresh_val = 128
            else:
                # Use TRIANGLE instead of OTSU
                thresh_val, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            
            # INVERTED Threshold usually finds Dark defects on Light background
            # If standard THRESH_BINARY: Source > Thresh = 255 (White).
            # Gold is bright. Defect is Dark.
            # So Gold > Thresh -> White. Defect < Thresh -> Black.
            # We want Defect to be White in the final mask.
            # So we use THRESH_BINARY_INV.
            # Let's check current logic: `cv2.threshold(..., cv2.THRESH_BINARY)`
            # This makes bright things white.
            # The user wants to detect "black marks".
            # So we want the INVERSE of the bright stuff?
            # Existing code seems to produce BW image where *something* is white.
            # Let's stick to the current convention: High=White.
            # If we want to detect defects, we usually invert at the end or change this.
            # Assuming existing logic is: White = Defect?
            # Wait, valid mask logic isn't clear in original snippet.
            # Let's use standard logic:
            # If Thresholding finds the "Bright Gold", then "Dark Defect" is the hole in it.
            # Then we invert.
            
            # For now, let's keep the existing flow but enforce the Mask at the end.
            _, bw = cv2.threshold(smooth_u8, thresh_val, 255, cv2.THRESH_BINARY)
            bw[~mask_bool] = 0
            auto_params = (sigma, thresh_val / 255.0)
            
        else:
            # Manual
            T = int(np.clip(thresh, 0.0, 1.0) * 255)
            _, bw = cv2.threshold(smooth_u8, T, 255, cv2.THRESH_BINARY)
            bw[~mask_bool] = 0

        # [HSV INTEGRATION STEP]
        # If enabled, we want to IGNORE everything outside the Gold Mask.
        if use_hsv and gold_mask is not None:
            # Logic:
            # The background is Green. The defects are Black spots on Gold.
            # The Gold is bright.
            # 'bw' currently has White for Bright things (Gold) and Black for Dark things (Defects + Background).
            # If we purely AND with gold_mask...
            # GoldMask = White (255) on Gold, Black (0) on Green.
            # BW = White on Gold, Black on Defect, Black on Green.
            # Result = White on Clean Gold, Black on Defect, Black on Green.
            # This just isolates the gold pad. It doesn't "highlight" the defect.
            #
            # User wants: "Defect if black% > X".
            # So they are measuring the amount of BLACK pixels.
            # If we allow the Green Background to remain Black, the "Black%" will be huge (failure).
            # We need the Green Background to be ignored (count as "White" or "Transparent" or "Not Defect").
            # 
            # Solution:
            # We want to measure "Black Spots ON THE GOLD".
            # We should forcibly set the Background (Non-Gold) pixels to WHITE (Safe) in the binary map.
            # So that only the ACTUAL defects (dark spots on gold) remain Black.
            
            # 1. Invert Gold Mask: White=Background/Green, Black=Gold
            bg_mask = cv2.bitwise_not(gold_mask)
            
            # 2. Set Background pixels in 'bw' to WHITE (255)
            # This ensures they are not counted as "Defects" (Black pixels).
            bw = cv2.bitwise_or(bw, bg_mask)

        # 6. Morphological Stabilization
        if noise_level > 0:
            k_size = 3 
            if noise_level >= 5: k_size = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        
        return gray_u8, smooth_u8, bw, mask_bool, auto_params
    
    def _compute_stats(self, bw_u8, mask_bool):
        """Compute white/black pixel statistics."""
        area_px = int(np.count_nonzero(mask_bool))
        if area_px <= 0:
            return 0, 0, 0, 0.0, 0.0
        
        white_px = int(np.count_nonzero((bw_u8 > 0) & mask_bool))
        black_px = area_px - white_px
        white_pct = 100.0 * white_px / area_px
        black_pct = 100.0 * black_px / area_px
        return white_px, black_px, area_px, white_pct, black_pct
    
    # === EVENT HANDLERS ===
    
    def _on_threshold_mode_change(self):
        """Handle threshold mode change (Otsu/Adaptive/Manual)."""
        use_otsu = self.use_otsu.get()
        use_adaptive = self.use_adaptive.get()
        
        # Mutual exclusivity - if one is checked, uncheck the other
        if use_otsu and use_adaptive:
            # Last checked wins
            if hasattr(self, '_last_mode') and self._last_mode == 'otsu':
                self.use_otsu.set(False)
                use_otsu = False
            else:
                self.use_adaptive.set(False)
                use_adaptive = False
        
        self._last_mode = 'adaptive' if use_adaptive else ('otsu' if use_otsu else 'manual')
        
        # Update slider states
        if use_otsu:
            self.sigma_scale.configure(state="disabled")
            self.thresh_scale.configure(state="disabled")
            self.block_scale.configure(state="disabled")
            self.c_scale.configure(state="disabled")
        elif use_adaptive:
            self.sigma_scale.configure(state="normal")
            self.thresh_scale.configure(state="disabled")
            self.block_scale.configure(state="normal")
            self.c_scale.configure(state="normal")
        else:
            self.sigma_scale.configure(state="normal")
            self.thresh_scale.configure(state="normal")
            self.block_scale.configure(state="disabled")
            self.c_scale.configure(state="disabled")
        
        self._refresh_preview()
    
    def _on_otsu_toggle(self):
        """Legacy toggle - redirects to new handler."""
        self._on_threshold_mode_change()
    
    def _select_input(self):
        """Select input folder."""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if not folder:
            return
        
        self.input_folder.set(folder)
        self.files = self._list_images(folder)
        self.idx = 0
        self.results = []
        
        if not self.files:
            self.input_label.config(text="No images found", foreground="#FF4444")
            messagebox.showwarning("No Images", "No supported image files found in folder.")
            return
        
        self.input_label.config(text=os.path.basename(folder), foreground=self.FG_COLOR)
        self.status_var.set(f"Loaded {len(self.files)} images from {os.path.basename(folder)}")
        self._load_current()
    
    def _select_output(self):
        """Select output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
            self.output_label.config(text=os.path.basename(folder), foreground=self.FG_COLOR)
    
    def _get_output_dir(self):
        """Get or create output directory."""
        out = self.output_folder.get().strip()
        if not out:
            if self.input_folder.get().strip():
                out = os.path.join(self.input_folder.get().strip(), "BW_out")
        if out:
            os.makedirs(out, exist_ok=True)
        return out
    
    def _load_current(self):
        """Load and display current image."""
        if not self.files:
            return
        
        path = self.files[self.idx]
        try:
            rgb, alpha = self._read_image_with_alpha(path)
        except Exception as e:
            messagebox.showerror("Read Error", str(e))
            return
        
        self.current_image = rgb
        self.current_alpha = alpha
        self._refresh_preview()
    
    def _on_preset_change(self, event=None):
        """Handle preset selection change."""
        preset = self.preset_var.get()
        if preset in self.DEFECT_PRESETS:
            settings = self.DEFECT_PRESETS[preset]
            
            # Update variables without triggering callbacks recursively
            self.sigma.set(settings["sigma"])
            self.thresh.set(settings["thresh"])
            self.adaptive_block_size.set(settings["block"])
            self.adaptive_c.set(settings["c"])
            
            # Update labels
            self.sigma_label.config(text=f"{settings['sigma']:.2f}")
            self.thresh_label.config(text=f"{settings['thresh']:.2f}")
            self.block_label.config(text=f"{int(settings['block'])}")
            self.c_label.config(text=f"{int(settings['c'])}")
            
            # Force Adaptive Mode if not already
            if not self.use_adaptive.get():
                self.use_adaptive.set(True)
                self.use_otsu.set(False)
            
            self._refresh_preview()

    def _on_manual_change(self, *args):
        """Handle manual slider adjustment -> switch to Custom preset."""
        if self.preset_var.get() != "Custom":
            self.preset_var.set("Custom")
        self._refresh_preview()

    def _refresh_preview(self):
        """Refresh preview with current settings."""
        if self.current_image is None:
            return
        
        use_otsu = self.use_otsu.get()
        use_adaptive = self.use_adaptive.get()
        use_hsv = self.use_hsv.get()
        
        result = self._make_binary(
            self.current_image,
            self.current_alpha,
            sigma=float(self.sigma.get()),
            thresh=float(self.thresh.get()),
            use_otsu=use_otsu,
            use_adaptive=use_adaptive,
            block_size=int(self.adaptive_block_size.get()),
            c_value=int(self.adaptive_c.get()),
            noise_level=int(self.noise_level.get()),
            use_hsv=use_hsv
        )
        _, _, bw_u8, mask_bool, auto_params = result
        
        # Update labels based on mode
        if use_otsu and auto_params:
            auto_sigma, auto_thresh = auto_params
            self.sigma.set(auto_sigma)
            if isinstance(auto_thresh, (float, int)):
                self.thresh.set(auto_thresh)
                self.thresh_label.configure(text=f"{auto_thresh:.2f}")
            self.sigma_label.configure(text=f"{auto_sigma:.2f}")
        else:
            self.sigma_label.configure(text=f"{self.sigma.get():.2f}")
            self.thresh_label.configure(text=f"{self.thresh.get():.2f}")
        
        block_val = int(self.adaptive_block_size.get())
        if block_val % 2 == 0: block_val += 1
        self.block_label.configure(text=str(block_val))
        self.c_label.configure(text=str(int(self.adaptive_c.get())))
        
        self.current_bw = bw_u8
        
        # Defect Analysis
        self.auto_defects, _ = analyze_defects(bw_u8, mask_bool)
        
        # Compute stats
        white_px, black_px, area_px, white_pct, black_pct = self._compute_stats(bw_u8, mask_bool)
        
        if black_pct > float(self.black_defect_pct.get()):
            status = "DEFECT"
            self.result_label.config(text=f"⚠ DEFECT ({len(self.auto_defects)})", fg="#FF4444")
        else:
            status = "OK"
            self.result_label.config(text=f"✓ OK ({len(self.auto_defects)})", fg="#00FF41")
            
        # Update Info
        if self.files:
            base = os.path.basename(self.files[self.idx])
            self.nav_info.config(text=f"{self.idx+1}/{len(self.files)}: {base}")
            
        # Stats Text
        mode_str = "Adaptive" if use_adaptive else ("Otsu" if use_otsu else "Manual")
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Status: {status}\n")
        self.stats_text.insert(tk.END, f"Mode: {mode_str}\n")
        self.stats_text.insert(tk.END, f"Black: {black_pct:.2f}% ({black_px} px)\n")
        self.stats_text.insert(tk.END, f"White: {white_pct:.2f}% ({white_px} px)\n")
        self.stats_text.insert(tk.END, f"Defects: {len(self.auto_defects)}\n")
        
        self._display_on_canvas(self.current_image, self.orig_canvas, is_rgb=True)
        self._refresh_visualization()

    def _refresh_visualization(self):
        """Update overlay and treeview."""
        if self.current_bw is None: return
        
        # 1. Update List
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if self.show_auto_in_list.get():
             for d in self.auto_defects:
                 self.tree.insert("", "end", values=(d['id'], d['type'], d['area']), tags="auto")

        for m in self.manual_labels:
            self.tree.insert("", "end", values=(f"M{m['id']}", m['type'], m['area']), tags="manual")
            
        # 2. Update Image Overlay
        vis_img = cv2.cvtColor(self.current_bw, cv2.COLOR_GRAY2RGB)
        
        if self.show_overlay.get():
             for d in self.auto_defects:
                 x, y, w, h = d['bbox']
                 cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                 
             for m in self.manual_labels:
                 x, y, w, h = m['x'], m['y'], m['w'], m['h']
                 cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                 cv2.putText(vis_img, m['type'], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                 
        self._display_on_canvas(vis_img, self.bw_canvas)

    def _remove_selected_label(self):
        """Remove the selected label from the list."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select Label", "Please select a label to remove.")
            return

        item = self.tree.item(selected[0])
        tags = item.get("tags")
        values = item.get("values")
        
        # Only allow deleting manual labels
        if "manual" in tags:
            label_id = str(values[0]) # e.g. "M1"
            
            # Find in manual_labels list
            # The ID in tree is "M{id}", so we strip "M"
            try:
                raw_id = int(label_id.replace("M", ""))
                
                # Filter out the one to delete
                old_len = len(self.manual_labels)
                self.manual_labels = [m for m in self.manual_labels if m['id'] != raw_id]
                
                if len(self.manual_labels) < old_len:
                    self._refresh_visualization()
                    self._save_labels_json()
                    # Keep removed logic consistent with Undo/Clear
            except ValueError:
                 pass
        else:
            messagebox.showinfo("Cannot Delete", "Only manual labels can be removed here.\nAuto defects are generated by processing settings.")

    def _open_labeler(self):
        if self.current_image is None: return
        DefectLabelerWindow(self, self.current_image.copy(), self.manual_labels, self._update_labels)
        
    def _update_labels(self, new_labels):
        self.manual_labels = new_labels
        self._refresh_visualization()
        self._save_labels_json()
        
    def _save_labels_json(self):
        if not self.files: return
        try:
            folder = os.path.dirname(self.files[self.idx])
            base = os.path.splitext(os.path.basename(self.files[self.idx]))[0]
            path = os.path.join(folder, f"{base}_labels.json")
            with open(path, 'w') as f:
                json.dump(self.manual_labels, f, indent=2)
        except: pass
        

    
    def _prev_image(self):
        """Navigate to previous image."""
        if not self.files:
            return
        self.idx = (self.idx - 1) % len(self.files)
        self._load_current()
    
    def _next_image(self):
        """Navigate to next image."""
        if not self.files:
            return
        self.idx = (self.idx + 1) % len(self.files)
        self._load_current()
    
    def _process_all(self):
        """Process all images and save to OK/DEFECT folders."""
        if not self.files:
            messagebox.showwarning("No Images", "Please select an input folder first.")
            return
        
        out = self._get_output_dir()
        if not out:
            messagebox.showwarning("Output", "Please select an output folder.")
            return
        
        # Create OK/DEFECT subfolders
        out_ok = os.path.join(out, "OK")
        out_ng = os.path.join(out, "DEFECT")
        os.makedirs(out_ok, exist_ok=True)
        os.makedirs(out_ng, exist_ok=True)
        
        use_otsu = self.use_otsu.get()
        use_adaptive = self.use_adaptive.get()
        sigma = float(self.sigma.get())
        thresh = float(self.thresh.get())
        block_size = int(self.adaptive_block_size.get())
        c_value = int(self.adaptive_c.get())
        black_th = float(self.black_defect_pct.get())
        
        self.results = []
        defect_count = 0
        
        for i, path in enumerate(self.files, start=1):
            self.status_var.set(f"Processing {i}/{len(self.files)}: {os.path.basename(path)}")
            self.update_idletasks()
            
            try:
                rgb, alpha = self._read_image_with_alpha(path)
                result = self._make_binary(rgb, alpha, sigma=sigma, thresh=thresh, 
                                          use_otsu=use_otsu, use_adaptive=use_adaptive,
                                          block_size=block_size, c_value=c_value,
                                          use_hsv=self.use_hsv.get())
                _, _, bw, mask_bool, _ = result
                
                white_px, black_px, area_px, white_pct, black_pct = self._compute_stats(bw, mask_bool)
                status = "DEFECT" if black_pct > black_th else "OK"
                
                if status == "DEFECT":
                    defect_count += 1
                
                # Save to appropriate folder
                base = os.path.splitext(os.path.basename(path))[0]
                save_dir = out_ng if status == "DEFECT" else out_ok
                out_path = os.path.join(save_dir, f"{base}_BW.png")
                cv2.imwrite(out_path, bw)
                
                self.results.append({
                    "path": path,
                    "rgb": rgb,
                    "bw": bw,
                    "white_pct": white_pct,
                    "black_pct": black_pct,
                    "status": status,
                })
            except Exception as e:
                self.results.append({
                    "path": path,
                    "error": str(e),
                    "status": "ERROR"
                })
        
        self.status_var.set(f"Done! {defect_count}/{len(self.files)} defects found")
        
        messagebox.showinfo("Processing Complete",
            f"Processed {len(self.files)} images.\n\n"
            f"DEFECT: {defect_count}\n"
            f"OK: {len(self.files) - defect_count}\n\n"
            f"Saved to:\n• {out_ok}\n• {out_ng}")
        
        self._load_current()
    
    def _open_overview(self):
        """Open overview window showing all results."""
        if not self.results:
            if not self.files:
                messagebox.showwarning("No Data", "Please select input folder first.")
                return
            # Quick process without saving
            self._quick_process()
        
        OverviewWindow(self, self.results, self.black_defect_pct.get())
    
    def _quick_process(self):
        """Quick process all images without saving (for overview)."""
        use_otsu = self.use_otsu.get()
        use_adaptive = self.use_adaptive.get()
        sigma = float(self.sigma.get())
        thresh = float(self.thresh.get())
        block_size = int(self.adaptive_block_size.get())
        c_value = int(self.adaptive_c.get())
        black_th = float(self.black_defect_pct.get())
        
        self.results = []
        for path in self.files:
            try:
                rgb, alpha = self._read_image_with_alpha(path)
                result = self._make_binary(rgb, alpha, sigma=sigma, thresh=thresh, 
                                          use_otsu=use_otsu, use_adaptive=use_adaptive,
                                          block_size=block_size, c_value=c_value,
                                          use_hsv=self.use_hsv.get())
                _, _, bw, mask_bool, _ = result
                _, _, _, white_pct, black_pct = self._compute_stats(bw, mask_bool)
                status = "DEFECT" if black_pct > black_th else "OK"
                self.results.append({
                    "path": path, "rgb": rgb, "bw": bw,
                    "white_pct": white_pct, "black_pct": black_pct, "status": status
                })
            except Exception as e:
                self.results.append({"path": path, "error": str(e), "status": "ERROR"})
    
    def _display_on_canvas(self, img, canvas, size=(450, 400), is_gray=False, is_rgb=False):
        """Display image on canvas with centering and metadata."""
        if img is None or img.size == 0:
            return
        
        h, w = img.shape[:2]
        if h == 0 or w == 0: return
        
        # Get canvas dimensions
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw > 10 and ch > 10:
            size = (cw, ch)
            
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w <= 0 or new_h <= 0:
            return
            
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if is_gray or len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif not is_rgb:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
        
        # Center in canvas
        off_x = (size[0] - new_w) // 2
        off_y = (size[1] - new_h) // 2
        
        canvas.delete("all")
        canvas.create_image(off_x, off_y, anchor=tk.NW, image=photo)
        canvas.image = photo # Keep reference
        
        # Store metadata for coordinates
        self.canvas_meta[canvas] = (ratio, off_x, off_y, w, h)

    def _on_mouse_move(self, event):
        """Track mouse coordinates."""
        canvas = event.widget
        if canvas not in self.canvas_meta:
            return
            
        ratio, off_x, off_y, orig_w, orig_h = self.canvas_meta[canvas]
        
        # Canvas coords
        cx, cy = event.x, event.y
        
        # Image coords
        if ratio > 0:
            ix = int((cx - off_x) / ratio)
            iy = int((cy - off_y) / ratio)
        else:
            ix, iy = 0, 0
            
        # Check bounds
        if 0 <= ix < orig_w and 0 <= iy < orig_h:
            self.lbl_coords.config(text=f"XY: {ix}, {iy}")
            self._draw_crosshair(self.orig_canvas, ix, iy)
            self._draw_crosshair(self.bw_canvas, ix, iy)
        else:
            self.lbl_coords.config(text="XY: -")
            self.orig_canvas.delete("crosshair")
            self.bw_canvas.delete("crosshair")

    def _on_mouse_leave(self, event):
        """Clear coordinates on leave."""
        self.lbl_coords.config(text="XY: -")
        self.orig_canvas.delete("crosshair")
        self.bw_canvas.delete("crosshair")

    def _draw_crosshair(self, canvas, img_x, img_y):
        """Draw crosshair on target canvas."""
        canvas.delete("crosshair")
        if canvas not in self.canvas_meta:
            return
            
        ratio, off_x, off_y, w, h = self.canvas_meta[canvas]
        
        # Target canvas coords
        tx = int(img_x * ratio + off_x)
        ty = int(img_y * ratio + off_y)
        
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        canvas.create_line(0, ty, cw, ty, fill="cyan", tags="crosshair", dash=(4, 4))
        canvas.create_line(tx, 0, tx, ch, fill="cyan", tags="crosshair", dash=(4, 4))


class OverviewWindow(tk.Toplevel):
    """Overview window for Simple Defect Detection results."""
    
    BG_COLOR = "#000000"
    FG_COLOR = "#00FFFF"
    
    def __init__(self, parent, results, defect_threshold):
        super().__init__(parent)
        
        self.title("Detection Overview - OK / DEFECT")
        self.geometry("1200x800")
        self.configure(bg=self.BG_COLOR)
        
        self.results = results
        self.defect_threshold = defect_threshold
        
        self._build_ui()
    
    def _build_ui(self):
        """Build overview UI."""
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        
        # Separate results
        ng_items = [r for r in self.results if r.get("status") == "DEFECT"]
        ok_items = [r for r in self.results if r.get("status") == "OK"]
        
        # DEFECT tab
        tab_ng = ttk.Frame(nb)
        nb.add(tab_ng, text=f"DEFECT ({len(ng_items)})")
        self._build_grid(tab_ng, ng_items)
        
        # OK tab
        tab_ok = ttk.Frame(nb)
        nb.add(tab_ok, text=f"OK ({len(ok_items)})")
        self._build_grid(tab_ok, ok_items)
    
    def _build_grid(self, parent, items):
        """Build scrollable grid of thumbnails."""
        canvas = tk.Canvas(parent, bg="#111111")
        vbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        frame = ttk.Frame(canvas, padding=10)
        canvas.create_window((0, 0), window=frame, anchor=tk.NW)
        
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self._thumb_refs = []
        cols = 5
        thumb_max = (160, 160)
        
        for i, res in enumerate(items):
            if "error" in res:
                continue
            
            tile = ttk.Frame(frame, padding=5)
            tile.grid(row=i//cols, column=i%cols, padx=5, pady=5, sticky=tk.N)
            
            # Create thumbnail
            thumb = self._create_thumbnail(res.get("rgb"), thumb_max)
            if thumb:
                self._thumb_refs.append(thumb)
                
                name = os.path.basename(res.get("path", "?"))
                black_pct = res.get("black_pct", 0)
                
                lbl = ttk.Label(tile, image=thumb,
                              text=f"{name}\nblack={black_pct:.1f}%",
                              compound=tk.TOP, justify=tk.CENTER)
                lbl.pack()
                
                # Detail button
                ttk.Button(tile, text="Details",
                          command=lambda r=res: self._show_detail(r)).pack(pady=3)
    
    def _create_thumbnail(self, rgb_img, max_size):
        """Create thumbnail from RGB image."""
        if rgb_img is None:
            return None
        
        h, w = rgb_img.shape[:2]
        ratio = min(max_size[0]/w, max_size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w == 0 or new_h == 0:
            return None
        
        thumb = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(image=Image.fromarray(thumb))
    
    def _show_detail(self, res):
        """Show detail window for a result."""
        d = tk.Toplevel(self)
        d.title(os.path.basename(res.get("path", "Detail")))
        d.geometry("900x700")
        d.configure(bg=self.BG_COLOR)
        
        row = ttk.Frame(d, padding=10)
        row.pack(fill=tk.BOTH, expand=True)
        
        # Original
        if res.get("rgb") is not None:
            img1 = self._create_large_image(res["rgb"], (400, 400))
            if img1:
                l1 = ttk.Label(row, image=img1, text="Original", compound=tk.TOP)
                l1.image = img1
                l1.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Binary
        if res.get("bw") is not None:
            bw_rgb = cv2.cvtColor(res["bw"], cv2.COLOR_GRAY2RGB)
            img2 = self._create_large_image(bw_rgb, (400, 400))
            if img2:
                l2 = ttk.Label(row, image=img2, text="Binary", compound=tk.TOP)
                l2.image = img2
                l2.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Info
        info = (
            f"{os.path.basename(res.get('path', '?'))}\n"
            f"Status: {res.get('status', '?')}\n"
            f"White: {res.get('white_pct', 0):.2f}%\n"
            f"Black: {res.get('black_pct', 0):.2f}%\n"
            f"DEFECT if black% > {self.defect_threshold:.1f}%"
        )
        ttk.Label(d, text=info, padding=10).pack()
    
    def _create_large_image(self, rgb_img, max_size):
        """Create large image for detail view."""
        return self._create_thumbnail(rgb_img, max_size)




# ==============================================================================
# TAB-BASED TOOL CLASSES (for dockable notebook tabs)
# ==============================================================================

class QRCropperTab(tk.Frame):
    """QR Code Detection and Cropping Tab (for notebook embedding)."""
    
    BG_COLOR = "#000000"
    FG_COLOR = "#00FF41"
    ACCENT_COLOR = "#00FF41"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent, app):
        super().__init__(parent, bg=self.BG_COLOR)
        self.app = app  # Reference to main app
        
        # Import QR extractor
        from .qr_cropper import QRCodeExtractor
        self.extractor = QRCodeExtractor()
        
        self.current_image = None
        self.last_results = []
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the QR Cropper UI."""
        # Main container
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Colorful title bar
        title_bar = tk.Frame(main_frame, bg="#003300")
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="🔲 QR CODE EXTRACTOR", 
                font=(self.FONT_FACE, 12, 'bold'),
                bg="#003300", fg=self.ACCENT_COLOR).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Top panel - Controls
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR,
                                 highlightbackground=self.FG_COLOR, highlightthickness=1)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons with colored styling
        tk.Button(controls_frame, text="📷 Load Image",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#004400", fg=self.FG_COLOR,
                 command=self._load_image).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Button(controls_frame, text="▶ Detect QR",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#004400", fg=self.FG_COLOR,
                 command=self._detect_qr).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Button(controls_frame, text="💾 Extract & Save",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#004400", fg=self.FG_COLOR,
                 command=self._extract_and_save).pack(side=tk.LEFT, padx=2, pady=5)
        
        self.image_status = tk.Label(controls_frame, text="No image loaded", 
                                    font=(self.FONT_FACE, 9),
                                    bg=self.BG_COLOR, fg="#888888")
        self.image_status.pack(side=tk.LEFT, padx=10)
        
        # Main content - Image and results side by side
        content_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display
        self.image_label = tk.Label(content_frame, bg="#111111")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        # Results panel
        results_frame = tk.Frame(content_frame, bg=self.BG_COLOR, width=200)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=2)
        results_frame.pack_propagate(False)
        
        ttk.Label(results_frame, text="Results:").pack(anchor=tk.W, pady=2)
        
        self.results_text = tk.Text(results_frame, bg="#111111", fg=self.FG_COLOR,
                                   font=(self.FONT_FACE, 8), height=15, width=25)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Output folder
        folder_frame = tk.Frame(results_frame, bg=self.BG_COLOR)
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Label(folder_frame, text="Output:").pack(anchor=tk.W)
        self.output_folder_var = tk.StringVar(value="qrcode_extraction")
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=20).pack(fill=tk.X)
    
    def _load_image(self):
        """Load an image for QR detection."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                self.image_status.config(text=os.path.basename(path)[:20], 
                                        foreground=self.FG_COLOR)
                self._display_image(self.current_image)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded. Click 'Detect QR'.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    def _detect_qr(self):
        """Detect QR codes in loaded image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            results = self.extractor.detect_and_decode(self.current_image)
            self.last_results = results
            
            annotated = self.extractor.annotate_image(self.current_image)
            self._display_image(annotated)
            
            self.results_text.delete(1.0, tk.END)
            if not results:
                self.results_text.insert(tk.END, "No QR codes detected.")
            else:
                self.results_text.insert(tk.END, f"Found {len(results)} QR code(s):\n\n")
                for qr in results:
                    self.results_text.insert(tk.END, f"#{qr['id']}: {qr['data']}\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}")
    
    def _extract_and_save(self):
        """Extract detected QR codes and save to folder."""
        if self.current_image is None or not self.last_results:
            self._detect_qr()
            if not self.last_results:
                return
        
        output_dir = self.output_folder_var.get()
        self.extractor.output_dir = output_dir
        
        try:
            saved_paths = self.extractor.save_cropped_qr(self.current_image)
            if saved_paths:
                self.results_text.insert(tk.END, f"\n\nSaved {len(saved_paths)} files to {output_dir}")
                messagebox.showinfo("Success", f"Saved {len(saved_paths)} QR code(s)")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _display_image(self, cv2_image, size=(400, 350)):
        """Display image in the label."""
        if cv2_image is None:
            return
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if new_w == 0 or new_h == 0:
            return
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.image_label.config(image=photo)
        self.image_label.image = photo


class GoldPadExtractorTab(tk.Frame):
    """Gold Pad Extraction Tab (for notebook embedding)."""
    
    BG_COLOR = "#000000"
    FG_COLOR = "#FFD700"
    ACCENT_COLOR = "#FFD700"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent, app):
        super().__init__(parent, bg=self.BG_COLOR)
        self.app = app
        
        self.current_image = None
        self.detected_pads = []
        self.extracted_pads = []
        
        # HSV range for gold
        self.hue_low = tk.IntVar(value=15)
        self.hue_high = tk.IntVar(value=35)
        self.sat_low = tk.IntVar(value=50)
        self.sat_high = tk.IntVar(value=255)
        self.val_low = tk.IntVar(value=100)
        self.val_high = tk.IntVar(value=255)
        
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=100)
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the Gold Pad Extractor UI."""
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Gold title bar
        title_bar = tk.Frame(main_frame, bg="#332800")
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="🔶 GOLD PAD EXTRACTOR", 
                font=(self.FONT_FACE, 12, 'bold'),
                bg="#332800", fg=self.ACCENT_COLOR).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Top controls with gold styling
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR,
                                 highlightbackground=self.FG_COLOR, highlightthickness=1)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(controls_frame, text="📷 Load Image",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._load_image).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="👁 Preview",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._preview_detection).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="✂ Extract",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._extract_pads).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="💾 Save",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._save_pads).pack(side=tk.LEFT, padx=2, pady=5)
        
        self.status_label = tk.Label(controls_frame, text="No image loaded",
                                    font=(self.FONT_FACE, 9),
                                    bg=self.BG_COLOR, fg="#888888")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # HSV sliders in a compact row
        hsv_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        hsv_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(hsv_frame, text="Hue:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0, to=50, variable=self.hue_low, orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0, to=50, variable=self.hue_high, orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT)
        
        tk.Label(hsv_frame, text="  Sat:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0, to=255, variable=self.sat_low, orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT)
        
        tk.Label(hsv_frame, text="  Val:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0, to=255, variable=self.val_low, orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT)
        
        # Image display
        self.image_label = tk.Label(main_frame, bg="#111111")
        self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def _load_image(self):
        """Load an image."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                self.status_label.config(text=os.path.basename(path)[:25])
                self._display_image(self.current_image)
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _get_gold_mask(self, image):
        """Create HSV mask for gold regions."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_low.get(), self.sat_low.get(), self.val_low.get()])
        upper = np.array([self.hue_high.get(), self.sat_high.get(), self.val_high.get()])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def _preview_detection(self):
        """Preview the gold mask and detected circles."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        mask = self._get_gold_mask(self.current_image)
        preview = self.current_image.copy()
        preview[mask > 0] = [0, 215, 255]  # Highlight gold areas
        self._display_image(preview)
        self.status_label.config(text=f"Gold area: {np.count_nonzero(mask)} px")
    
    def _extract_pads(self):
        """Extract individual gold pad images."""
        if self.current_image is None:
            return
        
        mask = self._get_gold_mask(self.current_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.extracted_pads = []
        preview = self.current_image.copy()
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            pad_crop = self.current_image[y:y+h, x:x+w].copy()
            self.extracted_pads.append(pad_crop)
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(preview, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self._display_image(preview)
        self.status_label.config(text=f"Extracted {len(self.extracted_pads)} pads")
    
    def _save_pads(self):
        """Save extracted gold pads."""
        if not self.extracted_pads:
            self._extract_pads()
        if not self.extracted_pads:
            messagebox.showinfo("No Pads", "No gold pads detected to save.")
            return
        
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            for i, pad in enumerate(self.extracted_pads):
                path = os.path.join(folder, f"gold_pad_{i+1:03d}.png")
                cv2.imwrite(path, pad)
            messagebox.showinfo("Saved", f"Saved {len(self.extracted_pads)} gold pads to {folder}")
    
    def _display_image(self, cv2_image, size=(500, 400)):
        """Display image in the label."""
        if cv2_image is None:
            return
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if new_w == 0 or new_h == 0:
            return
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.image_label.config(image=photo)
        self.image_label.image = photo


class RedPadExtractorTab(tk.Frame):
    """Red Pad Extraction Tab (for notebook embedding)."""
    
    BG_COLOR = "#000000"
    FG_COLOR = "#FF4444"
    ACCENT_COLOR = "#FF4444"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent, app):
        super().__init__(parent, bg=self.BG_COLOR)
        self.app = app
        
        self.current_image = None
        self.extracted_pads = []
        
        # Red HSV range (wraps around 0)
        self.hue_low1 = tk.IntVar(value=0)
        self.hue_high1 = tk.IntVar(value=10)
        self.hue_low2 = tk.IntVar(value=160)
        self.hue_high2 = tk.IntVar(value=179)
        self.sat_low = tk.IntVar(value=100)
        self.val_low = tk.IntVar(value=100)
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the Red Pad Extractor UI."""
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Red title bar
        title_bar = tk.Frame(main_frame, bg="#330000")
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="🔴 RED PAD EXTRACTOR", 
                font=(self.FONT_FACE, 12, 'bold'),
                bg="#330000", fg=self.ACCENT_COLOR).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Controls with red styling
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR,
                                 highlightbackground=self.FG_COLOR, highlightthickness=1)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(controls_frame, text="📷 Load",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#440000", fg=self.FG_COLOR,
                 command=self._load_image).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="👁 Preview",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#440000", fg=self.FG_COLOR,
                 command=self._preview_detection).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="✂ Extract",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#440000", fg=self.FG_COLOR,
                 command=self._extract_pads).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="💾 Save",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#440000", fg=self.FG_COLOR,
                 command=self._save_pads).pack(side=tk.LEFT, padx=2, pady=5)
        
        self.status_label = tk.Label(controls_frame, text="No image loaded",
                                    font=(self.FONT_FACE, 9),
                                    bg=self.BG_COLOR, fg="#888888")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Image display
        self.image_label = tk.Label(main_frame, bg="#111111")
        self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def _load_image(self):
        """Load an image."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                self.status_label.config(text=os.path.basename(path)[:25])
                self._display_image(self.current_image)
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _get_red_mask(self, image):
        """Create HSV mask for red regions (dual hue range)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Red wraps around, so we need two ranges
        lower1 = np.array([self.hue_low1.get(), self.sat_low.get(), self.val_low.get()])
        upper1 = np.array([self.hue_high1.get(), 255, 255])
        lower2 = np.array([self.hue_low2.get(), self.sat_low.get(), self.val_low.get()])
        upper2 = np.array([self.hue_high2.get(), 255, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def _preview_detection(self):
        """Preview the red mask."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        mask = self._get_red_mask(self.current_image)
        preview = self.current_image.copy()
        preview[mask > 0] = [0, 0, 255]  # Highlight red areas
        self._display_image(preview)
        self.status_label.config(text=f"Red area: {np.count_nonzero(mask)} px")
    
    def _extract_pads(self):
        """Extract individual red pad images."""
        if self.current_image is None:
            return
        
        mask = self._get_red_mask(self.current_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.extracted_pads = []
        preview = self.current_image.copy()
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            pad_crop = self.current_image[y:y+h, x:x+w].copy()
            self.extracted_pads.append(pad_crop)
            cv2.rectangle(preview, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        self._display_image(preview)
        self.status_label.config(text=f"Extracted {len(self.extracted_pads)} pads")
    
    def _save_pads(self):
        """Save extracted red pads."""
        if not self.extracted_pads:
            self._extract_pads()
        if not self.extracted_pads:
            messagebox.showinfo("No Pads", "No red pads detected to save.")
            return
        
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            for i, pad in enumerate(self.extracted_pads):
                path = os.path.join(folder, f"red_pad_{i+1:03d}.png")
                cv2.imwrite(path, pad)
            messagebox.showinfo("Saved", f"Saved {len(self.extracted_pads)} red pads to {folder}")
    
    def _display_image(self, cv2_image, size=(500, 400)):
        """Display image in the label."""
        if cv2_image is None:
            return
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if new_w == 0 or new_h == 0:
            return
        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.image_label.config(image=photo)
        self.image_label.image = photo



# ==============================================================================
# HELPER FUNCTIONS FOR DEFECT DETECTION
# ==============================================================================

def masked_gaussian_smooth(gray01, mask01, sigma):
    """Normalized masked Gaussian (avoid boundary bleeding)."""
    if sigma <= 0:
        out = gray01.copy()
        out[mask01 <= 0] = 0.0
        return out

    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    k = max(k, 3)

    num = cv2.GaussianBlur(gray01 * mask01, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(mask01, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    out = num / (den + 1e-8)
    out[mask01 <= 0] = 0.0
    return out

def masked_median_smooth(gray01, mask01, kernel_size=3):
    """Normalized masked median (avoid boundary bleeding)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)
    
    src = (gray01 * 255.0).astype(np.uint8)
    out = cv2.medianBlur(src, kernel_size)
    out_f = out.astype(np.float32) / 255.0
    out_f[mask01 <= 0] = 0.0
    return out_f

def masked_bilateral_smooth(gray01, mask01, sigma):
    """
    Bilateral filter: smooths flat areas (texture) but keeps edges.
    """
    d = 9
    # sigmaColor: 75 is standard. sigmaSpace: controlled by slider via 'sigma' param
    s_space = max(sigma, 0.1)
    s_color = 75.0
    
    src = (gray01 * 255.0).astype(np.uint8)
    out = cv2.bilateralFilter(src, d, s_color, s_space)
    out_f = out.astype(np.float32) / 255.0
    out_f[mask01 <= 0] = 0.0
    return out_f

def analyze_defects(bw_u8, mask_bool, min_area=5):
    """
    Find connected components of 'defects' (black pixels inside mask).
    Returns list of dicts: {id, type, x, y, area, circularity, solidity, bbox}
    
    Classification Criteria:
    - Pinhole: Very small defects (area < 30 pixels)
    - Scratch: Elongated defects (aspect > 3.0) OR low circularity (< 0.3)
    - Stain: Round/blob-like defects (circularity > 0.5)
    - Irregular: Everything else (jagged, non-round, non-elongated)
    """
    if mask_bool is None:
        return [], None

    defect_map = np.zeros_like(bw_u8)
    defect_indices = (mask_bool) & (bw_u8 == 0)
    defect_map[defect_indices] = 255
    
    # Find contours for more detailed analysis
    contours, _ = cv2.findContours(defect_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defects = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Aspect Ratio (normalized to always be >= 1)
        aspect = float(w) / h if h > 0 else 1.0
        if aspect < 1.0:
            aspect = 1.0 / aspect
        
        # Circularity: 4π × Area / Perimeter² (1.0 = perfect circle)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        circularity = min(circularity, 1.0)  # Cap at 1.0
        
        # Solidity: Area / Convex Hull Area (1.0 = solid, low = jagged/hollow)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)
        
        # Classification Logic
        if area < 30:
            dtype = "Pinhole"
        elif aspect > 3.0 or circularity < 0.3:
            dtype = "Scratch"
        elif circularity > 0.5 and solidity > 0.8:
            dtype = "Stain"
        else:
            dtype = "Irregular"
        
        defects.append({
            "id": i + 1,
            "type": dtype,
            "x": cx,
            "y": cy,
            "area": int(area),
            "circularity": round(circularity, 3),
            "solidity": round(solidity, 3),
            "aspect": round(aspect, 2),
            "bbox": (x, y, w, h)
        })
        
    return defects, defect_map


class DefectLabelerWindow(tk.Toplevel):
    def __init__(self, parent, rgb_image, current_labels, on_save_callback):
        super().__init__(parent)
        self.title("Labeling Window - Fit to Screen")
        self.state("zoomed") # Maximize
        self.geometry("1400x900")
        
        self.rgb_image = rgb_image
        self.on_save_callback = on_save_callback
        
        # Deep copy labels so we don't mutate original until save
        self.labels = [dict(l) for l in current_labels] 
        self.edited = False
        
        self.orig_h, self.orig_w = rgb_image.shape[:2]
        
        # UI
        self._build_ui()
        
        # Bind resize to refit image
        self.canvas.bind("<Configure>", self._on_resize)
        
        # Mouse
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click) # Right click delete

        self.current_rect = None
        self.start_x = 0
        self.start_y = 0
        
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        
    def _build_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self, padding=5)
        toolbar.pack(side="bottom", fill="x")
        
        ttk.Button(toolbar, text="Save & Close", command=self._save_and_close).pack(side="right", padx=10)
        ttk.Button(toolbar, text="Cancel", command=self.destroy).pack(side="right", padx=10)
        
        # Edit Buttons
        ttk.Button(toolbar, text="Undo Last", command=self._undo_last).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Clear All", command=self._clear_all).pack(side="left", padx=5)
        
        ttk.Label(toolbar, text="| Left Drag: New Label | Right Click: Delete Label").pack(side="left", padx=10)
        
        # Canvas
        self.canvas = tk.Canvas(self, bg="#333", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
    def _on_resize(self, event):
        self._refresh_image()
        
    def _refresh_image(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50: return
        
        # Calculate scale to FIT
        scale_w = cw / self.orig_w
        scale_h = ch / self.orig_h
        self.scale = min(scale_w, scale_h) * 0.95 # 95% to have some margin
        
        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)
        
        # Resize
        pil = Image.fromarray(self.rgb_image)
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil)
        
        # Center
        self.off_x = (cw - new_w) // 2
        self.off_y = (ch - new_h) // 2
        
        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, image=self.tk_img, anchor="nw")
        
        # Redraw labels
        for l in self.labels:
            self._draw_label_rect(l)
            
    def _draw_label_rect(self, l):
        # l has x, y, w, h in ORIGINAL coords
        # Convert to CANVAS coords
        x1 = l["x"] * self.scale + self.off_x
        y1 = l["y"] * self.scale + self.off_y
        x2 = (l["x"] + l["w"]) * self.scale + self.off_x
        y2 = (l["y"] + l["h"]) * self.scale + self.off_y
        
        tag = f"label_{l['id']}"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00FF00", width=2, tags=tag)
        self.canvas.create_text(x1, y1-10, text=l["type"], fill="#00FF00", anchor="sw", tags=tag)
        
    def _to_orig_coords(self, cx, cy):
        ox = (cx - self.off_x) / self.scale
        oy = (cy - self.off_y) / self.scale
        return int(ox), int(oy)
        
    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="yellow", width=2, dash=(4,4)
        )

    def on_mouse_drag(self, event):
        if self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)
            
    def on_mouse_up(self, event):
        if not self.current_rect: return
        self.canvas.delete(self.current_rect)
        self.current_rect = None
        
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        
        if abs(x2-x1) < 5 or abs(y2-y1) < 5: return
        
        # Normalize
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        
        ox1, oy1 = self._to_orig_coords(x1, y1)
        ox2, oy2 = self._to_orig_coords(x2, y2)
        
        # Check bounds
        ox1 = max(0, ox1); oy1 = max(0, oy1)
        ox2 = min(self.orig_w, ox2); oy2 = min(self.orig_h, oy2)
        
        w = ox2 - ox1
        h = oy2 - oy1
        
        if w < 5 or h < 5: return
        
        # Simple askstring for now (importing simpledialog if needed, or using custom)
        # Using tkinter.simpledialog
        import tkinter.simpledialog as sd
        cls = sd.askstring("Class", "Defect Type (Scratch, Stain, Pinhole):", parent=self, initialvalue="Stain")
        if not cls: return
        
        # Add label
        new_id = 1
        if self.labels:
            try:
                # Ensure existing IDs are ints
                ids = [int(l["id"]) for l in self.labels if str(l["id"]).isdigit()]
                if ids:
                    new_id = max(ids) + 1
                else: 
                     new_id = len(self.labels) + 1
            except:
                new_id = len(self.labels) + 1
                 
        lbl = {
            "id": new_id,
            "type": cls,
            "x": ox1, "y": oy1, "w": w, "h": h, "area": w*h
        }
        self.labels.append(lbl)
        self.edited = True
        self._refresh_image() # Redraw all
        
    def on_right_click(self, event):
        ox, oy = self._to_orig_coords(event.x, event.y)
        # Find label
        to_del = None
        for l in self.labels:
            if l["x"] <= ox <= l["x"]+l["w"] and l["y"] <= oy <= l["y"]+l["h"]:
                to_del = l
                break
        
        if to_del:
            if messagebox.askyesno("Delete", f"Delete {to_del['type']}?", parent=self):
                self.labels.remove(to_del)
                self.edited = True
                self._refresh_image()

    def _undo_last(self):
        """Remove the last added label."""
        if self.labels:
            self.labels.pop()
            self.edited = True
            self._refresh_image()

    def _clear_all(self):
        """Remove all manual labels."""
        if not self.labels: return
        if messagebox.askyesno("Clear All", "Delete all manual labels?", parent=self):
            self.labels.clear()
            self.edited = True
            self._refresh_image()

    def _save_and_close(self):
        if self.on_save_callback:
            self.on_save_callback(self.labels)
        self.destroy()






# ==============================================================================
# TEXTURE ANALYSIS (FFT) WINDOW
# ==============================================================================

class TextureAnalysisWindow(tk.Toplevel):
    """FFT-based Texture Analysis Tool for filtering out periodic patterns."""
    
    BG_COLOR = "#0A0A1A"
    FG_COLOR = "#00FFFF"
    ACCENT_COLOR = "#FF00FF"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("Texture Analysis (FFT)")
        self.geometry("1400x800")
        self.configure(bg=self.BG_COLOR)
        
        # State
        self.current_image = None
        self.gray_image = None
        self.fft_spectrum = None
        self.fft_shifted = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the UI layout."""
        # Title
        title = tk.Label(self, text="[ TEXTURE ANALYSIS (FFT) ]",
                        font=(self.FONT_FACE, 14, 'bold'),
                        bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title.pack(pady=10)
        
        # Controls Frame
        ctrl_frame = tk.Frame(self, bg=self.BG_COLOR)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(ctrl_frame, text="📁 Load Image", bg="#333", fg=self.FG_COLOR,
                 font=(self.FONT_FACE, 10, 'bold'),
                 command=self._load_image).pack(side=tk.LEFT, padx=5)
        
        # Filter Type
        tk.Label(ctrl_frame, text="Filter:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=(20, 5))
        self.filter_type = tk.StringVar(value="Low Pass")
        ttk.Combobox(ctrl_frame, textvariable=self.filter_type,
                    values=["Low Pass", "High Pass", "Band Pass"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=5)
        
        # Radius Slider
        tk.Label(ctrl_frame, text="Radius:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=(20, 5))
        self.radius = tk.IntVar(value=30)
        self.radius_scale = ttk.Scale(ctrl_frame, from_=1, to=200, variable=self.radius,
                                      orient=tk.HORIZONTAL, length=150,
                                      command=self._on_radius_change)
        self.radius_scale.pack(side=tk.LEFT, padx=5)
        self.radius_label = tk.Label(ctrl_frame, text="30", bg=self.BG_COLOR, fg=self.FG_COLOR, width=4)
        self.radius_label.pack(side=tk.LEFT)
        
        # Outer Radius (for Band Pass)
        tk.Label(ctrl_frame, text="Outer:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=(20, 5))
        self.outer_radius = tk.IntVar(value=100)
        self.outer_scale = ttk.Scale(ctrl_frame, from_=1, to=300, variable=self.outer_radius,
                                     orient=tk.HORIZONTAL, length=100,
                                     command=self._on_radius_change)
        self.outer_scale.pack(side=tk.LEFT, padx=5)
        self.outer_label = tk.Label(ctrl_frame, text="100", bg=self.BG_COLOR, fg=self.FG_COLOR, width=4)
        self.outer_label.pack(side=tk.LEFT)
        
        # Apply Button
        tk.Button(ctrl_frame, text="▶ Apply Filter", bg="#004400", fg="#00FF00",
                 font=(self.FONT_FACE, 10, 'bold'),
                 command=self._apply_filter).pack(side=tk.LEFT, padx=20)
        
        # Display Frame (3 panels)
        display_frame = tk.Frame(self, bg=self.BG_COLOR)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original Image
        orig_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(orig_frame, text="ORIGINAL", bg=self.BG_COLOR, fg=self.FG_COLOR).pack()
        self.orig_canvas = tk.Canvas(orig_frame, bg="#111", highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        
        # FFT Spectrum
        fft_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        fft_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(fft_frame, text="FREQUENCY SPECTRUM", bg=self.BG_COLOR, fg=self.ACCENT_COLOR).pack()
        self.fft_canvas = tk.Canvas(fft_frame, bg="#111", highlightthickness=0)
        self.fft_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Filtered Result
        result_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(result_frame, text="FILTERED RESULT", bg=self.BG_COLOR, fg="#00FF00").pack()
        self.result_canvas = tk.Canvas(result_frame, bg="#111", highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="Load an image to begin")
        tk.Label(self, textvariable=self.status_var, bg="#222", fg=self.FG_COLOR,
                font=(self.FONT_FACE, 9)).pack(fill=tk.X, side=tk.BOTTOM)
    
    def _load_image(self):
        """Load an image file."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                from .io import read_image
                self.current_image = read_image(path)
                
                # Convert to grayscale for FFT
                if len(self.current_image.shape) == 3:
                    self.gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    self.gray_image = self.current_image.copy()
                
                self._display_on_canvas(self.current_image, self.orig_canvas)
                self._compute_fft()
                self.status_var.set(f"Loaded: {os.path.basename(path)} | Shape: {self.current_image.shape}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _compute_fft(self):
        """Compute FFT and display spectrum."""
        if self.gray_image is None:
            return
        
        # FFT
        f = np.fft.fft2(self.gray_image.astype(np.float32))
        self.fft_shifted = np.fft.fftshift(f)
        
        # Magnitude spectrum (log scale for visualization)
        magnitude = np.abs(self.fft_shifted)
        magnitude = np.log1p(magnitude)  # log(1 + x) for better visualization
        
        # Normalize to 0-255
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255
        self.fft_spectrum = magnitude.astype(np.uint8)
        
        # Display
        spectrum_color = cv2.applyColorMap(self.fft_spectrum, cv2.COLORMAP_JET)
        self._display_on_canvas(spectrum_color, self.fft_canvas)
    
    def _on_radius_change(self, val=None):
        """Update radius label."""
        self.radius_label.config(text=str(self.radius.get()))
        self.outer_label.config(text=str(self.outer_radius.get()))
    
    def _apply_filter(self):
        """Apply the selected frequency filter."""
        if self.fft_shifted is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        h, w = self.gray_image.shape
        cy, cx = h // 2, w // 2
        
        # Create mask
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        r = self.radius.get()
        r_outer = self.outer_radius.get()
        filter_type = self.filter_type.get()
        
        if filter_type == "Low Pass":
            # Keep center (low frequencies), block outer (high frequencies)
            mask = dist <= r
        elif filter_type == "High Pass":
            # Block center, keep outer
            mask = dist >= r
        elif filter_type == "Band Pass":
            # Keep ring between r and r_outer
            mask = (dist >= r) & (dist <= r_outer)
        else:
            mask = np.ones((h, w), dtype=bool)
        
        # Apply mask
        filtered_fft = self.fft_shifted * mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_fft)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize to 0-255
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        
        # Display result
        self._display_on_canvas(img_back, self.result_canvas)
        
        # Also show filtered spectrum
        filtered_magnitude = np.abs(filtered_fft)
        filtered_magnitude = np.log1p(filtered_magnitude)
        filtered_magnitude = (filtered_magnitude - filtered_magnitude.min()) / (filtered_magnitude.max() - filtered_magnitude.min() + 1e-8) * 255
        spectrum_filtered = cv2.applyColorMap(filtered_magnitude.astype(np.uint8), cv2.COLORMAP_JET)
        self._display_on_canvas(spectrum_filtered, self.fft_canvas)
        
        self.status_var.set(f"Applied {filter_type} filter with radius={r}")
    
    def _display_on_canvas(self, img, canvas):
        """Display image on a canvas."""
        if img is None:
            return
        
        h, w = img.shape[:2]
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            cw, ch = 400, 400
        
        ratio = min(cw / w, ch / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if new_w == 0 or new_h == 0:
            return
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    """Launch the GUI application.
    
    Note: Consider using hybrid_integrated.py as the main entry point instead.
    """
    app = InspectorApp()
    app.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("NOTE: For the main entry point, run hybrid_integrated.py")
    print("      python hybrid_integrated.py")
    print("=" * 60)
    print("Starting GUI anyway...")
    print("")
    main()
