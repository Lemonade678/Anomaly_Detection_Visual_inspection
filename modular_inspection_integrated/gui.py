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

class InspectorApp(tk.Tk):
    """Integrated PCB Inspector GUI Application."""
    
    # Theme colors
    BG_COLOR = "#000000"
    FG_COLOR = "#00FF41"
    ACCENT_COLOR = "#00FF41"
    ERROR_COLOR = "#FF0000"
    FONT_FACE = "Consolas"
    
    SSIM_PASS_THRESHOLD = 0.975
    
    def __init__(self):
        super().__init__()
        
        self.title("Integrated PCB Inspector - Advanced Edition")
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
        
        # Template matching settings
        self.template_grid_cols = 4
        self.template_grid_rows = 4
        self.template_search_margin = 50
        
        # Log file
        self.log_file = "inspection_log.csv"
        self._init_log_file()
        
        # Build UI
        self._setup_styles()
        self._build_ui()
        self._build_menu_bar()
    
    def _build_menu_bar(self):
        """Build the menu bar with Tools menu."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="QR Code Extractor", command=self._open_qr_cropper)
        tools_menu.add_command(label="Gold Pad Extractor", command=self._open_gold_pad_extractor)
        tools_menu.add_separator()
        tools_menu.add_command(label="Open Log File", command=self._open_log_file)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _open_qr_cropper(self):
        """Open the QR Code Extractor window."""
        qr_window = QRCropperWindow(self)
        qr_window.focus_set()
    
    def _open_gold_pad_extractor(self):
        """Open the Gold Pad Extractor window."""
        gold_window = GoldPadExtractorWindow(self)
        gold_window.focus_set()
    
    def _open_log_file(self):
        """Open the log file location."""
        import subprocess
        if os.path.exists(self.log_file):
            subprocess.run(['explorer', '/select,', os.path.abspath(self.log_file)])
        else:
            messagebox.showinfo("Log File", f"Log file location:\n{os.path.abspath(self.log_file)}")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """
INTEGRATED PCB INSPECTOR
Version 1.0.0

Features:
• Multi-method alignment (ORB/SIFT/ECC/Phase)
• Light sensitivity modes
• SSIM + Pixel matching pipeline
• Anomaly location mapping
• QR Code extraction

Combined from Modular_inspection_1 + modular_inspection2
        """
        messagebox.showinfo("About", about_text.strip())
    
    def _init_log_file(self):
        """Initialize log file with headers."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Image', 'Verdict', 'AreaScore', 
                               'AnomalyCount', 'PixelVerdict', 'SSIM', 'Time'])
    
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
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls (fixed, not scrollable)
        controls_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=320)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        controls_frame.pack_propagate(False)
        
        self._build_controls(controls_frame)
        
        # Right panel - Scrollable Image displays
        display_container = tk.Frame(main_frame, bg=self.BG_COLOR)
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
        def _on_display_mousewheel(event):
            self.display_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.display_canvas.bind_all("<MouseWheel>", _on_display_mousewheel)
        
        display_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.display_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._build_display_area(self.display_scrollable_frame)
        
        # Status bar
        self._build_status_bar()
    
    def _build_controls(self, parent):
        """Build control panel."""
        # Title
        title_label = tk.Label(parent, text="[ INTEGRATED INSPECTOR ]",
                              font=(self.FONT_FACE, 14, 'bold'),
                              bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
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
        """Build image display area as 2x3 grid.
        
        Layout:
        Row 1: [Golden Image]        [Sample/Test Image]
        Row 2: [Aligned/Contour]     [ANOMALY HEATMAP] <-- center right
        Row 3: [SSIM Heatmap]        [Contour Map]
        """
        # Row 1: Golden and Original Test
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Golden image (1,1)
        golden_frame = ttk.Frame(row1)
        golden_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(golden_frame, text="GOLDEN", font=('Consolas', 8)).pack()
        self.golden_label = tk.Label(golden_frame, bg="#111111")
        self.golden_label.pack(fill=tk.BOTH, expand=True)
        
        # Sample/Test image (1,2)
        test_frame = ttk.Frame(row1)
        test_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(test_frame, text="SAMPLE", font=('Consolas', 8)).pack()
        self.test_label = tk.Label(test_frame, bg="#111111")
        self.test_label.pack(fill=tk.BOTH, expand=True)
        
        # Row 2: Aligned and Anomaly Heatmap (center)
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.BOTH, expand=True, pady=1)
        
        # Aligned image (2,1)
        aligned_frame = ttk.Frame(row2)
        aligned_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(aligned_frame, text="ALIGNED", font=('Consolas', 8)).pack()
        self.aligned_label = tk.Label(aligned_frame, bg="#111111")
        self.aligned_label.pack(fill=tk.BOTH, expand=True)
        
        # ANOMALY HEATMAP (2,2) - Center Right Position
        anomaly_frame = ttk.Frame(row2)
        anomaly_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        anomaly_title = tk.Label(anomaly_frame, text="⚠ ANOMALY ⚠",
                                font=(self.FONT_FACE, 8, 'bold'),
                                bg=self.BG_COLOR, fg="#FF4444")
        anomaly_title.pack()
        self.pixel_label = tk.Label(anomaly_frame, bg="#111111",
                                   highlightbackground="#FF4444", highlightthickness=1)
        self.pixel_label.pack(fill=tk.BOTH, expand=True)
        
        # Row 3: SSIM and Contour Map
        row3 = ttk.Frame(parent)
        row3.pack(fill=tk.BOTH, expand=True, pady=1)
        
        # SSIM heatmap (3,1)
        ssim_frame = ttk.Frame(row3)
        ssim_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(ssim_frame, text="SSIM", font=('Consolas', 8)).pack()
        self.ssim_label = tk.Label(ssim_frame, bg="#111111")
        self.ssim_label.pack(fill=tk.BOTH, expand=True)
        
        # Contour Map (3,2)
        contour_frame = ttk.Frame(row3)
        contour_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        ttk.Label(contour_frame, text="CONTOUR", font=('Consolas', 8)).pack()
        self.contour_label = tk.Label(contour_frame, bg="#111111")
        self.contour_label.pack(fill=tk.BOTH, expand=True)
    
    def _build_status_bar(self):
        """Build status bar at bottom."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self, textvariable=self.status_var,
                             bg="#222222", fg=self.FG_COLOR,
                             font=(self.FONT_FACE, 9), anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
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
                self._display_image(self.golden_image, self.golden_label)
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
                self._display_image(self.test_image, self.test_label)  # Show test image immediately
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
            
            self._display_image(aligned, self.aligned_label)
            
            # SSIM check
            self.status_var.set("Running SSIM check...")
            self.update_idletasks()
            
            self.ssim_score, ssim_heatmap = calc_ssim(golden_proc, aligned)
            self._display_image(ssim_heatmap, self.ssim_label)
            
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
                self._display_image(blank, self.pixel_label)
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
            
            self._display_image(pixel_result['heatmap'], self.pixel_label)
            self._display_image(contour_map, self.contour_label)  # Display contour map in dedicated panel
            
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
    
    def _display_image(self, cv2_image: np.ndarray, label: tk.Label, size=(450, 350)):
        """Display OpenCV image in tkinter label."""
        if cv2_image is None or cv2_image.size == 0:
            return
        
        h, w = cv2_image.shape[:2]
        if h == 0 or w == 0:
            return
        
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
        
        self.title("QR Code Extractor")
        self.geometry("1000x700")
        self.configure(bg=self.BG_COLOR)
        
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
        
        self._build_ui()
    
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
        """Extract individual gold pad images."""
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
            
            # Create circular mask to remove background
            mask = np.zeros(pad_image.shape[:2], dtype=np.uint8)
            local_cx = cx - x1
            local_cy = cy - y1
            cv2.circle(mask, (local_cx, local_cy), r, 255, -1)
            
            # Apply mask (make background transparent or white)
            pad_masked = cv2.bitwise_and(pad_image, pad_image, mask=mask)
            
            # Create white background version
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
        """Save extracted gold pads to a folder."""
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
                # Save with white background
                filename = f"pad_{pad['id']:03d}.png"
                filepath = os.path.join(pads_folder, filename)
                cv2.imwrite(filepath, pad['image'])
                saved_files.append(filename)
                
                # Also save circular crop (transparent background as alpha)
                filename_alpha = f"pad_{pad['id']:03d}_masked.png"
                filepath_alpha = os.path.join(pads_folder, filename_alpha)
                
                # Create RGBA image with transparency
                b, g, r = cv2.split(pad['image_no_bg'])
                alpha = pad['mask']
                rgba = cv2.merge([b, g, r, alpha])
                cv2.imwrite(filepath_alpha, rgba)
            
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
                f"• pad_XXX.png (white background)\n"
                f"• pad_XXX_masked.png (transparent background)")
                
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
