"""
Inspection GUI Module - Main Application
========================================
Complete GUI application for visual inspection anomaly detection.

Features:
- Single image inspection with golden template comparison
- Batch processing mode for multiple images
- ORB-based image alignment (handles rotation/scale)
- Two-stage detection: SSIM pre-check + Pixel matching
- Auto-crop substrate detection
- Live folder monitoring
- CSV logging and 4K image export
"""
import os
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from inspection.io import read_image
from inspection.hybrid import InspectionProcessor

# Configuration constants
SSIM_PASS_THRESHOLD = 0.975  # Images above this SSIM score are considered normal


class InspectorProApp(tk.Tk):
    """Tkinter GUI for visual inspection anomaly detection.

    Pipeline: Load Images -> Align -> SSIM Check -> Pixel Match -> Report
    """

    def __init__(self):
        super().__init__()
        self.title("I.P.S. // Anomaly Detection Core v4.2")
        self.geometry("1400x950")

        # Image storage
        self.golden_image = None
        self.original_golden_image = None
        self.golden_cropped = False
        
        self.test_image_path = None
        self._cached_test_image = None
        self.original_test_image = None
        self.test_cropped = False
        
        self.aligned_test_image = None
        self.Sample_image = None
        self.ssim_score = 0.0
        self.ssim_heatmap = None

        # Core Processor
        self.processor = InspectionProcessor()

        # For saving high-res images
        self.last_pixel_heatmap = None
        self.last_aligned_image = None

        # State flags
        self._golden_loaded = False
        self._test_loaded = False
        self.log_file = "inspection_log.csv"

        # Monitoring
        self.monitor_thread = None
        self.monitoring_active = False
        self.monitored_folder = "captured_images"
        self.processed_files = set()

        # UI state vars
        self.status_bar_text = tk.StringVar(value="STATUS: System initialized. Load template to begin.")
        self.pixel_diff_var = tk.StringVar(value="40")
        self.count_tresh_var = tk.StringVar(value="1000")

        # Theme
        self.ACCENT_COLOR = "#90FF90"
        self.FONT_FACE = "Consolas"

        # Build UI
        self._apply_theme()
        self._create_widgets()
        self._create_menu_bar()
        self._setup_logging()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ==========================================================================
    # UI Setup
    # ==========================================================================
    
    def _apply_theme(self):
        """Apply dark hacker theme."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        BG_COLOR = "#000000"
        FG_COLOR = "#00FF41"

        self.configure(background=BG_COLOR)
        self.style.configure('.',
                             background=BG_COLOR,
                             foreground=FG_COLOR,
                             font=(self.FONT_FACE, 10),
                             fieldbackground="#1A1A1A")
        self.style.configure('TLabel', foreground=self.ACCENT_COLOR)

    def _create_menu_bar(self):
        """Create application menu bar with batch mode access."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Mode menu
        mode_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Mode", menu=mode_menu)
        mode_menu.add_command(label="Single Image Mode (Current)", 
                            command=lambda: messagebox.showinfo("Mode", "You are in Single Image Mode"))
        mode_menu.add_separator()
        mode_menu.add_command(label="Open Batch Processing Mode", 
                            command=self._open_batch_window)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="How to Use", command=self._show_help)
        help_menu.add_command(label="About Batch Mode", command=self._show_batch_help)

    def _open_batch_window(self):
        """Open batch processing in a new window."""
        try:
            from inspection.batch_gui import BatchProcessingTab
            
            batch_window = tk.Toplevel(self)
            batch_window.title("Batch Processing Mode - PCB Inspection")
            batch_window.geometry("900x800")
            
            batch_tab = BatchProcessingTab(batch_window)
            batch_tab.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            self.batch_window = batch_window
        except ImportError as e:
            messagebox.showerror("Import Error", f"Could not load batch processing module:\n{e}")

    def _show_batch_help(self):
        """Show help for batch mode."""
        help_text = """
BATCH PROCESSING MODE

1. Load Master Image: Select a defect-free golden template
2. Add Test Images: Add individual images or entire folders
3. Configure Parameters (strips, thresholds)
4. Run Batch Inspection
5. Review Results and Export CSV

The system automatically extracts strips from panels and compares each against the master.
        """
        messagebox.showinfo("Batch Mode Help", help_text.strip())

    def _create_widgets(self):
        """Create all GUI widgets."""
        BG_COLOR = "#000000"
        FG_COLOR = "#00FF41"
        ACCENT_COLOR = self.ACCENT_COLOR
        FONT_FACE = self.FONT_FACE

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(3):
            main_frame.columnconfigure(i, weight=1)
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=1)

        top_bar_frame = ttk.Frame(main_frame)
        top_bar_frame.grid(row=0, column=0, columnspan=3, sticky='ne', pady=(0, 5))
        help_button = ttk.Button(top_bar_frame, text=" [ ? ] ", command=self._show_help, style='Help.TButton')
        help_button.pack(anchor='e')

        # Controls column
        controls = tk.Frame(main_frame, background=BG_COLOR, highlightbackground=FG_COLOR, highlightthickness=1)
        controls.grid(row=1, column=0, sticky="nswe", padx=5, pady=5)
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="< CONTROLS // CONFIGURATION >", font=(FONT_FACE, 12, 'bold'), foreground=ACCENT_COLOR).pack(pady=(5,10), anchor='w', padx=10)

        ttk.Button(controls, text="Load Golden Image", command=self._load_golden_image).pack(fill=tk.X, padx=10, pady=5)
        
        # Golden crop controls frame
        golden_crop_frame = tk.Frame(controls, background=BG_COLOR)
        golden_crop_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.golden_display = ttk.Label(golden_crop_frame, text="Golden: N/A", foreground=FG_COLOR)
        self.golden_display.pack(side=tk.LEFT)
        
        self.golden_crop_status = ttk.Label(golden_crop_frame, text="", foreground="#FFD700")
        self.golden_crop_status.pack(side=tk.LEFT, padx=5)
        
        # Golden crop buttons
        golden_btn_frame = tk.Frame(controls, background=BG_COLOR)
        golden_btn_frame.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(golden_btn_frame, text="✂ Crop ROI", command=self._crop_golden_image, width=12).pack(side=tk.LEFT, padx=2)
        self.reset_golden_btn = ttk.Button(golden_btn_frame, text="↺ Reset", command=self._reset_golden_crop, width=10, state=tk.DISABLED)
        self.reset_golden_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(controls, text="Load Single Test Image", command=self.load_single_test_image).pack(fill=tk.X, padx=10, pady=10)
        
        # Test crop controls frame
        test_crop_frame = tk.Frame(controls, background=BG_COLOR)
        test_crop_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.test_display = ttk.Label(test_crop_frame, text="Test: N/A", foreground=FG_COLOR)
        self.test_display.pack(side=tk.LEFT)
        
        self.test_crop_status = ttk.Label(test_crop_frame, text="", foreground="#FFD700")
        self.test_crop_status.pack(side=tk.LEFT, padx=5)
        
        # Test crop buttons
        test_btn_frame = tk.Frame(controls, background=BG_COLOR)
        test_btn_frame.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(test_btn_frame, text="✂ Crop ROI", command=self._crop_test_image, width=12).pack(side=tk.LEFT, padx=2)
        self.reset_test_btn = ttk.Button(test_btn_frame, text="↺ Reset", command=self._reset_test_crop, width=10, state=tk.DISABLED)
        self.reset_test_btn.pack(side=tk.LEFT, padx=2)

        # Auto-Crop Option
        self.auto_crop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Auto-Crop Substrate", variable=self.auto_crop_var).pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(controls, text="► ENGAGE INSPECTION", command=self._run_single_inspection, style='Accent.TButton').pack(fill=tk.X, padx=10, pady=15, ipady=10)

        # Live Monitor Section
        monitor_lf = tk.Frame(controls, background=BG_COLOR, highlightbackground=FG_COLOR, highlightthickness=1)
        monitor_lf.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(monitor_lf, text="< LIVE MONITOR >", font=(FONT_FACE, 11, 'bold'), foreground=ACCENT_COLOR).pack(pady=5, anchor='w', padx=5)
        self.monitor_button = ttk.Button(monitor_lf, text="📡 INITIATE LIVE SCAN", command=self._toggle_monitoring)
        self.monitor_button.pack(fill=tk.X, padx=5, pady=5)
        self.monitor_status_label = ttk.Label(monitor_lf, text="Status: Inactive", foreground="#FFFF00")
        self.monitor_status_label.pack(fill=tk.X, padx=5, pady=5)

        # Parameters Section
        config_lf = tk.Frame(controls, background=BG_COLOR, highlightbackground=FG_COLOR, highlightthickness=1)
        config_lf.pack(fill=tk.X, padx=10, pady=10)
        config_lf.columnconfigure(1, weight=1)
        config_lf.columnconfigure(2, weight=0)
        ttk.Label(config_lf, text="< PARAMETERS >", font=(FONT_FACE, 11, 'bold'), foreground=ACCENT_COLOR).grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        ttk.Label(config_lf, text="Pixel Diff (0-255):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.pixel_slider = ttk.Scale(config_lf, from_=0, to=255, orient=tk.HORIZONTAL, command=self._update_pixel_diff_entry)
        self.pixel_slider.set(40)
        self.pixel_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.pixel_entry = ttk.Entry(config_lf, textvariable=self.pixel_diff_var, width=5, font=(FONT_FACE, 10))
        self.pixel_entry.grid(row=1, column=2, padx=(0, 5))
        self.pixel_entry.bind('<Return>', self._on_pixel_entry_change)
        self.pixel_entry.bind('<FocusOut>', self._on_pixel_entry_change)

        ttk.Label(config_lf, text="Count thresh:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.count_slider = ttk.Scale(config_lf, from_=0, to=2000, orient=tk.HORIZONTAL, command=self._update_count_entry)
        self.count_slider.set(100)
        self.count_slider.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        self.count_entry = ttk.Entry(config_lf, textvariable=self.count_tresh_var, width=5, font=(FONT_FACE, 10))
        self.count_entry.grid(row=2, column=2, padx=(0, 5))
        self.count_entry.bind('<Return>', self._on_count_entry_change)
        self.count_entry.bind('<FocusOut>', self._on_count_entry_change)

        # Pixel Analysis Frame
        pixel_frame = tk.Frame(main_frame, background=BG_COLOR, highlightbackground=FG_COLOR, highlightthickness=1)
        pixel_frame.grid(row=1, column=1, sticky="nswe", padx=5, pady=5)
        
        pf_header = tk.Frame(pixel_frame, background=BG_COLOR)
        pf_header.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(pf_header, text="< ANALYSIS // PIXEL MATCH >", font=(FONT_FACE, 12, 'bold'), foreground=ACCENT_COLOR).pack(side=tk.LEFT)
        ttk.Button(pf_header, text="SAVE IMG", command=lambda: self._save_image_4k(self.pixel_heatmap_label.image, "pixel_match"), width=10).pack(side=tk.RIGHT)

        self.pixel_heatmap_label = ttk.Label(pixel_frame, anchor="center")
        self.pixel_heatmap_label.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=5)
        self.pixel_stats_label = ttk.Label(pixel_frame, text="Stats: N/A", font=(FONT_FACE, 10))
        self.pixel_stats_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=10)

        # Aligned Image Frame
        aligned_frame = tk.Frame(main_frame, background=BG_COLOR, highlightbackground=FG_COLOR, highlightthickness=1)
        aligned_frame.grid(row=1, column=2, sticky="nswe", padx=5, pady=5)
        
        af_header = tk.Frame(aligned_frame, background=BG_COLOR)
        af_header.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(af_header, text="< TARGET // ALIGNED & BOXED >", font=(FONT_FACE, 12, 'bold'), foreground=ACCENT_COLOR).pack(side=tk.LEFT)
        ttk.Button(af_header, text="SAVE IMG", command=lambda: self._save_image_4k(self.aligned_label.image, "aligned_target"), width=10).pack(side=tk.RIGHT)

        self.aligned_label = ttk.Label(aligned_frame, anchor="center")
        self.aligned_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_label = ttk.Label(self, textvariable=self.status_bar_text, anchor="w", font=(FONT_FACE, 11, 'bold'), foreground=ACCENT_COLOR)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def _show_help(self):
        """Show usage instructions."""
        instructions = f"""
ANOMALY DETECTION WORKFLOW

--- ALIGNMENT ---
Uses ORB feature matching to align test image to golden template.
Handles rotation, scale, and perspective differences.

--- STAGE 1: SSIM Pre-Check ---
If SSIM > {SSIM_PASS_THRESHOLD:.2f}, image passes as NORMAL.
This is a fast structural similarity check.

--- STAGE 2: Pixel Matching ---
If SSIM fails, detailed pixel analysis runs:
- Computes absolute difference
- Applies threshold and dilation
- Counts anomalous pixels and contours

STEPS:
1) Load Golden Image (defect-free template)
2) Load Test Image
3) Optionally crop ROI or enable auto-crop
4) Configure thresholds
5) Click ENGAGE INSPECTION
        """
        messagebox.showinfo("How to Use", instructions.strip())

    # ==========================================================================
    # Slider/Entry Handlers
    # ==========================================================================
    
    def _update_pixel_diff_entry(self, value):
        self.pixel_diff_var.set(f"{int(float(value))}")

    def _on_pixel_entry_change(self, event=None):
        try:
            val = int(self.pixel_diff_var.get())
            min_val = int(self.pixel_slider.cget('from'))
            max_val = int(self.pixel_slider.cget('to'))
            clamped_val = max(min_val, min(max_val, val))
            if val != clamped_val:
                self.pixel_diff_var.set(str(clamped_val))
            self.pixel_slider.set(clamped_val)
        except ValueError:
            self.pixel_diff_var.set(str(int(self.pixel_slider.get())))

    def _update_count_entry(self, value):
        self.count_tresh_var.set(f"{int(float(value))}")

    def _on_count_entry_change(self, event=None):
        try:
            val = int(self.count_tresh_var.get())
            min_val = int(self.count_slider.cget('from'))
            max_val = int(self.count_slider.cget('to'))
            clamped_val = max(min_val, min(max_val, val))
            if val != clamped_val:
                self.count_tresh_var.set(str(clamped_val))
            self.count_slider.set(clamped_val)
        except ValueError:
            self.count_tresh_var.set(str(int(self.count_slider.get())))

    # ==========================================================================
    # Image Loading
    # ==========================================================================

    def _load_golden_image(self):
        """Load golden/master template image."""
        path = filedialog.askopenfilename(
            title="Select Golden Template Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.cr2;*.cr3;*.nef;*.arw;*.dng"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as e:
            messagebox.showerror("Load Golden Error", f"Could not load image:\n{path}\n\n{e}")
            self.golden_image = None
            self.golden_display.config(text="Golden: N/A", foreground="#EAEAEA")
            self._golden_loaded = False
            return
        self.golden_image = img
        self.original_golden_image = None  # Reset original
        self.golden_cropped = False
        self._golden_loaded = True
        self.processor.set_golden_image(self.golden_image, path)
        self.golden_display.config(text=f"Golden: ...{os.path.basename(path)[-20:]}", foreground="#00FF41")
        self.golden_crop_status.config(text="")
        self.reset_golden_btn.config(state=tk.DISABLED)

    def load_single_test_image(self):
        """Load test image for inspection."""
        path = filedialog.askopenfilename(
            title="Select Single Test Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.cr2;*.cr3;*.nef;*.arw;*.dng"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            test = read_image(path)
        except Exception as e:
            messagebox.showerror("Load Test Error", f"Could not load image:\n{path}\n\n{e}")
            self.test_image_path = None
            self.test_display.config(text="Test: N/A", foreground="#EAEAEA")
            self._test_loaded = False
            return
        self._cached_test_image = test
        self.original_test_image = None  # Reset original
        self.test_cropped = False
        self.test_image_path = path
        self._test_loaded = True
        self.test_display.config(text=f"Test: ...{os.path.basename(path)[-20:]}", foreground="#00FF41")
        self.test_crop_status.config(text="")
        self.reset_test_btn.config(state=tk.DISABLED)

    # ==========================================================================
    # Cropping
    # ==========================================================================

    def _crop_golden_image(self):
        """Interactive crop for golden image."""
        from inspection.crop_tool import select_roi, crop_image, show_tkinter_crop_preview
        
        if not self._golden_loaded or self.golden_image is None:
            messagebox.showwarning("No Image", "Please load a Golden Image first.")
            return
        
        if self.original_golden_image is None:
            self.original_golden_image = self.golden_image.copy()
        
        current_image = self.golden_image.copy()
        self.status_bar_text.set("STATUS: Select ROI on the image...")
        self.update_idletasks()
        
        roi = select_roi(current_image, "Select Golden Image ROI")
        
        if roi is None:
            self.status_bar_text.set("STATUS: Crop cancelled.")
            return
        
        cropped = crop_image(current_image, roi)
        self.status_bar_text.set("STATUS: Review crop preview...")
        self.update_idletasks()
        
        accepted = show_tkinter_crop_preview(current_image, cropped, parent=self, 
                                            window_title="Golden Image Crop Preview")
        
        if accepted:
            self.golden_image = cropped
            self.golden_cropped = True
            self.processor.set_golden_image(self.golden_image, "Cropped_Golden")
            self.golden_crop_status.config(text="✓ Cropped", foreground="#00FF41")
            self.reset_golden_btn.config(state=tk.NORMAL)
            self.status_bar_text.set(f"STATUS: Golden image cropped ({cropped.shape[1]}x{cropped.shape[0]})")
        else:
            self.status_bar_text.set("STATUS: Crop rejected.")
    
    def _reset_golden_crop(self):
        """Reset golden image to original."""
        if self.original_golden_image is not None:
            self.golden_image = self.original_golden_image.copy()
            self.golden_cropped = False
            self.processor.set_golden_image(self.golden_image, "Reset_Golden")
            self.golden_crop_status.config(text="")
            self.reset_golden_btn.config(state=tk.DISABLED)
            self.status_bar_text.set("STATUS: Golden image reset to original.")
    
    def _crop_test_image(self):
        """Interactive crop for test image."""
        from inspection.crop_tool import select_roi, crop_image, show_tkinter_crop_preview
        
        if not self._test_loaded or self._cached_test_image is None:
            messagebox.showwarning("No Image", "Please load a Test Image first.")
            return
        
        if self.original_test_image is None:
            self.original_test_image = self._cached_test_image.copy()
        
        current_image = self._cached_test_image.copy()
        self.status_bar_text.set("STATUS: Select ROI on the test image...")
        self.update_idletasks()
        
        roi = select_roi(current_image, "Select Test Image ROI")
        
        if roi is None:
            self.status_bar_text.set("STATUS: Crop cancelled.")
            return
        
        cropped = crop_image(current_image, roi)
        self.status_bar_text.set("STATUS: Review crop preview...")
        self.update_idletasks()
        
        accepted = show_tkinter_crop_preview(current_image, cropped, parent=self,
                                            window_title="Test Image Crop Preview")
        
        if accepted:
            self._cached_test_image = cropped
            self.test_cropped = True
            self.test_crop_status.config(text="✓ Cropped", foreground="#00FF41")
            self.reset_test_btn.config(state=tk.NORMAL)
            self.status_bar_text.set(f"STATUS: Test image cropped ({cropped.shape[1]}x{cropped.shape[0]})")
        else:
            self.status_bar_text.set("STATUS: Crop rejected.")
    
    def _reset_test_crop(self):
        """Reset test image to original."""
        if self.original_test_image is not None:
            self._cached_test_image = self.original_test_image.copy()
            self.test_cropped = False
            self.test_crop_status.config(text="")
            self.reset_test_btn.config(state=tk.DISABLED)
            self.status_bar_text.set("STATUS: Test image reset to original.")

    # ==========================================================================
    # Live Monitoring
    # ==========================================================================

    def _toggle_monitoring(self):
        """Toggle live folder monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self.monitor_button.config(text="📡 INITIATE LIVE SCAN")
            self.monitor_status_label.config(text="Status: Inactive", foreground="#FFFF00")
            self.status_bar_text.set("STATUS: Live scan deactivated.")
            return

        if not self._golden_loaded or self.golden_image is None:
            messagebox.showwarning("Input Missing", "Please load a Golden Template before starting the monitor.")
            return

        self.monitoring_active = True
        self.monitor_button.config(text="🛑 CEASE LIVE SCAN")
        self.monitor_status_label.config(text="Status: ACTIVE", foreground="#00FF41")
        self.status_bar_text.set(f"STATUS: Live scan active. Monitoring folder: {self.monitored_folder}")
        self.processed_files.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_folder_backend, daemon=True)
        self.monitor_thread.start()

    def _monitor_folder_backend(self):
        """Background thread for folder monitoring."""
        if not os.path.exists(self.monitored_folder):
            os.makedirs(self.monitored_folder)
        while self.monitoring_active:
            try:
                files = [f for f in os.listdir(self.monitored_folder)
                         if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
                newest_file, latest_time = None, 0
                for f in files:
                    if f in self.processed_files:
                        continue
                    p = os.path.join(self.monitored_folder, f)
                    t = os.path.getmtime(p)
                    if t > latest_time:
                        newest_file, latest_time = f, t
                if newest_file:
                    path = os.path.join(self.monitored_folder, newest_file)
                    self.processed_files.add(newest_file)
                    self.after(0, self.status_bar_text.set, f"STATUS: New file detected. Processing: {newest_file}")
                    self.after(0, self._process_single_image, path, None)
                time.sleep(1)
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                self.after(0, self.status_bar_text.set, f"ERROR: Monitor thread failed: {e}")

    # ==========================================================================
    # Core Inspection Pipeline
    # ==========================================================================

    def _run_single_inspection(self):
        """Run inspection on loaded test image."""
        if not self._golden_loaded or self.golden_image is None:
            messagebox.showwarning("Input Missing", "Please load a Golden Template image.")
            return
        if not self._test_loaded or self.test_image_path is None:
            messagebox.showwarning("Input Missing", "Please load a Single Test image.")
            return
        self._process_single_image(self.test_image_path, self._cached_test_image)

    def _process_single_image(self, test_image_path, preloaded=None):
        """
        Main inspection pipeline delegated to InspectionProcessor.
        """
        start_time = time.time()
        self.status_bar_text.set(f"STATUS: Processing: {os.path.basename(test_image_path)}...")
        self.update_idletasks()

        # Step 1: Load image
        try:
            test_image = preloaded if preloaded is not None else read_image(test_image_path)
            if test_image is None:
                raise ValueError("Image read failed")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read test image:\n{test_image_path}\n\n{e}")
            self.status_bar_text.set(f"ERROR: Failed to load test image")
            return

        # Delegate to processor
        results = self.processor.process_image(
            test_image=test_image,
            test_path=test_image_path,
            auto_crop=self.auto_crop_var.get(),
            pixel_diff_thresh=int(self.pixel_slider.get()),
            count_thresh=int(self.count_slider.get())
        )

        # Handle errors
        if 'error' in results:
            if results.get('verdict') == "ALIGN_FAIL":
                messagebox.showwarning("Alignment Error", results['error'])
            else:
                messagebox.showerror("Processing Error", results['error'])
            self.status_bar_text.set(f"ERROR: {results['error']}")
            return

        # Update GUI state
        self.aligned_test_image = results.get('aligned_image')
        self.Sample_image = self.aligned_test_image # Alias
        self.ssim_score = results.get('ssim_score', 0.0)
        self.ssim_heatmap = results.get('ssim_heatmap')
        
        verdict = results.get('verdict', "Unknown")
        pixel_res = results.get('pixel_result')
        
        if "SSIM" in verdict:
            # SSIM Pass
            self._update_gui_for_ssim_pass()
            final_color = "#00FF41"
        else:
            # Pixel Match
            if pixel_res:
                final_color = "#FF0000" if pixel_res['verdict'] == "Anomaly" else "#00FF41"
                self._update_gui(pixel_res, verdict, final_color)
            else:
                final_color = "#FFFF00" # Should not happen if not SSIM pass
        
        processing_time = time.time() - start_time
        self.status_bar_text.set(f"STATUS: Completed in {processing_time:.2f}s. Verdict: {verdict}")
        self.status_label.config(foreground=final_color)

    # ==========================================================================
    # GUI Updates
    # ==========================================================================

    def _update_gui(self, pixel_res, final_verdict, final_color):
        """Update GUI with pixel matching results."""
        aligned_photo = self._prepare_image_for_display(pixel_res['contour_map'], (400, 400))
        self.aligned_label.image = aligned_photo
        self.aligned_label.config(image=aligned_photo)

        pixel_photo = self._prepare_image_for_display(pixel_res['heatmap'], (400, 400))
        self.pixel_heatmap_label.image = pixel_photo
        self.pixel_heatmap_label.config(image=pixel_photo)

        # Store for saving
        self.last_aligned_image = self.aligned_test_image
        self.last_pixel_heatmap = pixel_res['heatmap']

        pixel_verdict_color = "#FF0000" if pixel_res['verdict'] == "Anomaly" else self.ACCENT_COLOR
        count_thresh_val = pixel_res.get('count_thresh', int(self.count_slider.get()))
        anomalous_pixels = pixel_res.get('anomalous_pixel_count', 0)

        self.pixel_stats_label.config(
            text=(f"Verdict: {pixel_res['verdict']}\n"
                  f"Area Score:       {pixel_res['area_score']:.2f}% (Thresh: 20%)\n"
                  f"Anomalous Pixels: {anomalous_pixels:,} (Thresh: {count_thresh_val:,})\n"
                  f"Contour Count:    {pixel_res['anomaly_count']}\n"
                  f"SSIM Score:       {self.ssim_score:.4f} (Pre-Check)"),
            font=(self.FONT_FACE, 10, 'bold'),
            foreground=pixel_verdict_color,
            justify=tk.LEFT
        )

    def _update_gui_for_ssim_pass(self):
        """Update GUI when SSIM passes."""
        aligned_photo = self._prepare_image_for_display(self.Sample_image, (400, 400))
        self.aligned_label.image = aligned_photo
        self.aligned_label.config(image=aligned_photo)

        pixel_photo = self._prepare_image_for_display(self.ssim_heatmap, (400, 400))
        self.pixel_heatmap_label.image = pixel_photo
        self.pixel_heatmap_label.config(image=pixel_photo)

        # Store for saving
        self.last_aligned_image = self.Sample_image
        self.last_pixel_heatmap = self.ssim_heatmap

        self.pixel_stats_label.config(
            text=(f"Verdict: Normal (SSIM)\n"
                  f"SSIM Score: {self.ssim_score:.4f} (Thresh: {SSIM_PASS_THRESHOLD})\n"
                  f"Pixel analysis skipped."),
            font=(self.FONT_FACE, 11, 'bold'),
            foreground=self.ACCENT_COLOR,
            justify=tk.LEFT
        )

    # ==========================================================================
    # Utilities
    # ==========================================================================

    def _save_image_4k(self, photo_image, prefix):
        """Save image at 4K resolution."""
        if prefix == "pixel_match":
            img_to_save = self.last_pixel_heatmap
        elif prefix == "aligned_target":
            img_to_save = self.last_aligned_image
        else:
            img_to_save = None

        if img_to_save is None:
            messagebox.showwarning("Save Error", "No image available to save.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"{prefix}_{timestamp}.png"
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            title="Save 4K Image",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        
        if not path:
            return

        target_width = 3840
        h, w = img_to_save.shape[:2]
        scale = target_width / w
        target_height = int(h * scale)
        
        try:
            img_4k = cv2.resize(img_to_save, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(path, img_4k)
            messagebox.showinfo("Saved", f"Image saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save image:\n{e}")

    def _setup_logging(self):
        """Initialize CSV log file."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Test_Image", "Final_Verdict",
                                 "Pixel_Area_Score_%", "Pixel_Anomaly_Count",
                                 "Pixel_Verdict", "SSIM_Score", "Processing_Time_s"])

    def _log_result(self, test_path, verdict, p_score, p_count, p_verdict, p_time, ssim_score):
        """Log inspection result to CSV."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, os.path.basename(test_path), verdict,
                             f"{p_score:.4f}", p_count, p_verdict,
                             f"{ssim_score:.4f}", f"{p_time:.2f}"])

    def _prepare_image_for_display(self, cv2_image, size):
        """Prepare OpenCV image for Tkinter display with aspect ratio."""
        h, w = cv2_image.shape[:2]
        if h == 0 or w == 0:
            return ImageTk.PhotoImage(image=Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)))

        ratio = min(size[0] / w, size[1] / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if new_w == 0 or new_h == 0:
            return ImageTk.PhotoImage(image=Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)))

        img_resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        x_offset = (size[0] - new_w) // 2
        y_offset = (size[1] - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

    def on_closing(self):
        """Handle window close."""
        self.monitoring_active = False
        self.destroy()


# ==========================================================================
# Main Entry Point
# ==========================================================================

if __name__ == "__main__":
    app = InspectorProApp()
    app.mainloop()


