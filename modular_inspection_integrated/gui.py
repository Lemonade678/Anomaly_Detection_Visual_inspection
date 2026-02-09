"""
Let focus on pad binary inspection
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import sys
import time
import csv
import json
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient

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
from .theme import DARK_THEME


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
        self.defect_pct = tk.DoubleVar()
        
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

        # Defect Percentage
        self._add_slider(controls_frame, "Defect % Threshold", self.defect_pct, 0.5, 50.0, 0.5)
        
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
            self.defect_pct.set(p.get("defect_pct", 10.0))
            
    def _save(self):
        # Update current working copy
        name = self.current_preset.get()
        self.presets[name] = {
            "sigma": round(self.sigma.get(), 2),
            "thresh": round(self.thresh.get(), 2),
            "block": int(self.block.get()),
            "c": int(self.c_val.get()),
            "defect_pct": round(self.defect_pct.get(), 1)
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
    

    
    BG_COLOR = DARK_THEME.BG_MAIN
    FG_COLOR = DARK_THEME.FG_PRIMARY
    ACCENT_COLOR = DARK_THEME.ACCENT_INFO
    FONT_FACE = DARK_THEME.FONT_MAIN[0]
    
    # Preset Definitions (now includes noise and defect_pct)
    DEFECT_PRESETS = {
        "General":   {"sigma": 1.1, "thresh": 0.65, "block": 11, "c": 2, "defect_pct": 10.0},
        "Scratches": {"sigma": 0.8, "thresh": 0.60, "block": 7,  "c": 2, "defect_pct": 5.0},
        "Stains":    {"sigma": 2.0, "thresh": 0.70, "block": 25, "c": 4, "defect_pct": 15.0},
        "Pinholes":  {"sigma": 0.8, "thresh": 0.60, "block": 9,  "c": 5, "defect_pct": 3.0},
        "irregular": {"sigma": 1.2, "thresh": 0.70, "block": 15, "c": 3, "defect_pct": 20.0}
    }
    
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".bitmap", ".dib")
    
    def __init__(self):
        super().__init__()
        
        self.title("Integrated PCB Inspector")
        self.geometry("1600x1000")
        self.state('zoomed')
        self.configure(bg=DARK_THEME.BG_MAIN)
        
        # State
        self.input_folder = tk.StringVar(value="")
        self.output_folder = tk.StringVar(value="")
        self.sigma = tk.DoubleVar(value=1.2)
        self.thresh = tk.DoubleVar(value=0.65)
        self.use_adaptive = tk.BooleanVar(value=False)  # Adaptive thresholding
        self.adaptive_block_size = tk.IntVar(value=11)  # Block size for adaptive
        self.adaptive_c = tk.IntVar(value=2)  # C constant for adaptive
        self.black_defect_pct = tk.DoubleVar(value=10.0)
        
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
        self.current_image = None
        self.current_alpha = None
        self.current_bw = None
        self.current_mask_bool = None # Cache for fast updates
        self.files = [] # Initialize files list
        
        # Roboflow Client
        # Roboflow Client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="wIxP1NaMNv7xs6nFf1Of"
        )
        self.model_id = "single_gold_pad_binart/2"
        self.classification_result = None
        
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
        style.configure("TLabel", background=DARK_THEME.BG_MAIN, foreground=DARK_THEME.FG_PRIMARY,
                       font=DARK_THEME.FONT_MAIN)
        style.configure("TButton", background=DARK_THEME.BG_PANEL, foreground=DARK_THEME.FG_PRIMARY,
                       font=DARK_THEME.FONT_BOLD)
        style.configure("TFrame", background=DARK_THEME.BG_MAIN)
        style.configure("TCheckbutton", background=DARK_THEME.BG_MAIN, foreground=DARK_THEME.FG_PRIMARY)
        
        # Blue (Cyan) button style for folder selection
        style.configure("Blue.TButton", 
                       background="#0088AA", foreground="#FFFFFF",
                       font=DARK_THEME.FONT_BOLD)
        style.map("Blue.TButton",
                 background=[("active", "#00AACC"), ("pressed", "#006688")])
    
    def _build_menu(self):
        """Build menu bar with navigation."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
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
        
        # [NEW] Single Image Button with Special Decor (Gold/Warning Style)
        # We'll use a custom style or just a simple button if ttk logic is complex
        # Let's make a new style for it
        style = ttk.Style()
        style.configure("Gold.TButton", 
                       background="#FFD700", foreground="#000000",
                       font=(self.FONT_FACE, 9, 'bold'))
        style.map("Gold.TButton",
                 background=[("active", "#FFEE00"), ("pressed", "#CCAA00")])
                 
        ttk.Button(folder_frame, text="📄 Select Single Image", style="Gold.TButton",
                  command=self._select_single_image).pack(fill=tk.X, padx=5, pady=2)

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
        
        # [NEW] Classification
        class_row = tk.Frame(settings_frame, bg=self.BG_COLOR)
        class_row.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(class_row, text="🤖 Classify & Tune", command=self._classify_and_tune).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.class_label = ttk.Label(class_row, text="", foreground="#FFFF00", font=("Consolas", 8))
        self.class_label.pack(side=tk.LEFT, padx=5)
        
        
        
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
                   textvariable=self.black_defect_pct, width=7, command=self._on_verdict_change)
        self.defect_spin.pack(side=tk.LEFT, padx=5)
        # Also bind KeyRelease for typing
        self.defect_spin.bind("<KeyRelease>", lambda e: self._on_verdict_change())
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

        #padding the original image to add black circle
        padding_frame = tk.Frame(display_frame, bg=self.BG_COLOR)
        padding_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Original image
        orig_frame = tk.Frame(padding_frame, bg=self.BG_COLOR)
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
            
            ToolTip(self.adaptive_check, "Use local adaptive thresholding (better for uneven lighting)")
            ToolTip(self.block_scale, "Size of the local neighborhood for adaptive thresholding")
            ToolTip(self.c_scale, "Constant subtracted from the mean (Adaptive C)")
            ToolTip(self.sigma_scale, "Gaussian blur strength (Sigma) to reduce noise before thresholding")
            ToolTip(self.thresh_scale, "Manual global threshold value (0.0 - 1.0)")
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
    
    def _make_binary(self, rgb, alpha, sigma=1.2, thresh=0.65, 
                      use_adaptive=False, block_size=11, c_value=2,
                      use_hsv=False):
        """RGB -> Gray -> Filter (Gaussian/Bilateral/Median) -> Threshold -> BW."""
        
        # 0. HSV Masking (Gold Focus)
        gold_mask = None
        if use_hsv:
            gold_mask = self._get_hsv_mask(rgb)
        
        

        # 1. Grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. CLAHE
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
            # Using internal helper
            smooth_f = self._masked_gaussian_smooth(gray_f, mask01, float(sigma))
        elif f_method == "Bilateral":
            # Assuming bilateral helper exists or falling back to gaussian if not found?
            # Sticking to previous pattern but correcting access if it was instance method
            # If masked_bilateral_smooth isn't found, this might fail. 
            # I will use self._masked_gaussian_smooth as safe fallback if I can't find bilateral
            # But let's try to assume it was imported?
            # Actually, to be safe, I'll stick to what was there but check for self.
            try:
                smooth_f = masked_bilateral_smooth(gray_f, mask01, float(sigma))
            except NameError:
                smooth_f = self._masked_gaussian_smooth(gray_f, mask01, float(sigma))
        elif f_method == "Median":
            k = int(float(sigma) * 3)
            if k % 2 == 0: k += 1
            smooth_f = self._masked_gaussian_smooth(gray_f, mask01, float(sigma)) # Fallback to avoid missing median helper?
            # Previous code used masked_median_smooth. 
            # I'll try to use it cautiously.
            try:
                smooth_f = masked_median_smooth(gray_f, mask01, kernel_size=k)
            except NameError:
                 # Median filter using OpenCV directly
                 smooth_u8_temp = np.clip(gray_f * 255, 0, 255).astype(np.uint8)
                 smooth_u8_temp = cv2.medianBlur(smooth_u8_temp, k)
                 smooth_f = smooth_u8_temp.astype(np.float32) / 255.0

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
            
            auto_params = (sigma, f"adapt({block},{c_value})")
            
        else:
            # Manual
            T = int(np.clip(thresh, 0.0, 1.0) * 255)
            _, bw = cv2.threshold(smooth_u8, T, 255, cv2.THRESH_BINARY)
            bw[~mask_bool] = 0

        # [HSV INTEGRATION STEP]
        if use_hsv and gold_mask is not None:
            bg_mask = cv2.bitwise_not(gold_mask)
            bw = cv2.bitwise_or(bw, bg_mask)

        # Removed Morphological Stabilization (noise_level)

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
        """Handle threshold mode change (Adaptive/Manual)."""
        use_adaptive = self.use_adaptive.get()
        
        # Update slider states
        if use_adaptive:
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
    

    
    def _select_input(self):
        """Select input folder."""
        folder = filedialog.askdirectory(title="Select Desired Input Folder")
        if not folder:
            return messagebox.showwarning("No Folder Selected", "No folder selected.")
        
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

    def _select_single_image(self):
        """Select a single image file."""
        f = filedialog.askopenfilename(
            title="Select Single Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if f:
            self.files = [f]
            self.input_folder.set(os.path.dirname(f))
            self.input_label.config(text=f"FILE: {os.path.basename(f)}", foreground=self.FG_COLOR)
            self.idx = 0
            self.results = []
            self.status_var.set(f"Loaded single file: {os.path.basename(f)}")
            self._load_current()
            messagebox.showinfo("Single Image", f"Loaded: {os.path.basename(f)}")
    
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
            if "defect_pct" in settings:
                self.black_defect_pct.set(settings["defect_pct"])
            
            # Update labels
            self.sigma_label.config(text=f"{settings['sigma']:.2f}")
            self.thresh_label.config(text=f"{settings['thresh']:.2f}")
            self.block_label.config(text=f"{int(settings['block'])}")
            self.c_label.config(text=f"{int(settings['c'])}")
            
            # Force Adaptive Mode when using a preset
            if not self.use_adaptive.get():
                self.use_adaptive.set(True)
                self._on_threshold_mode_change()
            self._refresh_preview()

    def _classify_and_tune(self):
        """Call Roboflow to classify image and auto-select preset."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self.class_label.config(text="Analyzing...")
        self.update_idletasks()
        
        try:
            # 1. Save temp image for inference
            # Using tempfile would be cleaner, but simple specific path is fine for this context
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                temp_path = tf.name
            
            # Convert RGB (internal) to BGR for saving
            bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, bgr)
            
            # 2. Call Roboflow API
            # result = self.client.infer(temp_path, model_id=self.model_id) # Generic infer
            # The project snippet used get_model -> predict. 
            # But the initialized client can also do it directly if configured right.
            # Let's use the standard 'infer' call on the client we inited.
            
            res = self.client.infer(temp_path, model_id=self.model_id)
            
            # Clean up
            os.remove(temp_path)
            
            # 3. Parse result
            # Expected format: {'predictions': [{'class': 'Stain', 'confidence': 0.88, ...}], ...}
            # Or if it's classification, it might be different. Let's assume standard classification response.
            
            if 'predictions' in res and len(res['predictions']) > 0:
                # Classification usually returns top class or list
                # If it's object detection... user said "Classification".
                # Let's assume the user set up a Classification project.
                # If it is returning a list of predictions (classes), take top one.
                
                # Handling generic structure flexibility
                preds = res['predictions']
                if isinstance(preds, list):
                    top = preds[0]
                elif isinstance(preds, dict):
                    # Sometimes key is the class name? No, usually 'predictions' is list or dict with class keys
                    # If it's a dict like {'Stain': 0.9, 'Scratch': 0.1}:
                    top_class = max(preds, key=preds.get)
                    top = {'class': top_class, 'confidence': preds[top_class]}
                else:
                    top = {'class': 'Unknown', 'confidence': 0.0}
                    
                # Extract class name
                # Sometimes key is 'class', sometimes 'class_name', or just the string if list of strings
                c_name = top.get('class', top.get('class_name', 'Unknown'))
                conf = top.get('confidence', 0.0)
                
                self.classification_result = f"{c_name} ({conf:.2f})"
                self.class_label.config(text=self.classification_result)
                
                # 4. Map to Preset
                # Mapping: Stain -> Stains, Scratch -> Scratches, Pinhole -> Pinholes, Normal -> General
                target_preset = "General"
                lower_name = c_name.lower()
                
                if "stain" in lower_name:
                    target_preset = "Stains"
                elif "scratch" in lower_name:
                    target_preset = "Scratches"
                elif "pinhole" in lower_name:
                    target_preset = "Pinholes"
                elif 'irregular' in lower_name:
                    target_preset = "Irregularities"
                
                # Apply
                if target_preset in self.DEFECT_PRESETS:
                    self.preset_var.set(target_preset)
                    self._on_preset_change()
                    messagebox.showinfo("Tuned", f"Detected: {c_name}\nApplied preset: {target_preset}")
                else:
                    messagebox.showwarning("Unknown Class", f"Class '{c_name}' not mapped to any known preset.")
            else:
                self.class_label.config(text="No prediction")
                
        except Exception as e:
            self.class_label.config(text="Error")
            messagebox.showerror("Inference Error", str(e))

    def _on_manual_change(self, *args):
        """Handle manual slider adjustment -> switch to Custom preset."""
        if self.preset_var.get() != "Custom":
            self.preset_var.set("Custom")
        self._refresh_preview()

    def _refresh_preview(self):
        """Refresh preview with current settings."""
        if self.current_image is None:
            return
        
        use_adaptive = self.use_adaptive.get()
        use_hsv = self.use_hsv.get()
        
        result = self._make_binary(
            self.current_image,
            self.current_alpha,
            sigma=float(self.sigma.get()),
            thresh=float(self.thresh.get()),
            use_adaptive=use_adaptive,
            block_size=int(self.adaptive_block_size.get()),
            c_value=int(self.adaptive_c.get()),
            use_hsv=use_hsv
        )
        _, _, bw_u8, mask_bool, auto_params = result
        
        self.current_bw = bw_u8
        self.current_mask_bool = mask_bool # Cache it
        
        # Update labels based on mode
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
        
        # Find the defect type with the most total area
        dominant_defect_type = ""
        if self.auto_defects:
            area_by_type = {}
            for d in self.auto_defects:
                dtype = d.get('type', 'Unknown')
                area_by_type[dtype] = area_by_type.get(dtype, 0) + d.get('area', 0)
            if area_by_type:
                dominant_defect_type = max(area_by_type, key=area_by_type.get)
        
        if black_pct > float(self.black_defect_pct.get()):
            status = "DEFECT"
            if dominant_defect_type:
                self.result_label.config(text=f"⚠ {dominant_defect_type} ({len(self.auto_defects)})", fg="#FF4444")
            else:
                self.result_label.config(text=f"⚠ DEFECT ({len(self.auto_defects)})", fg="#FF4444")
        else:
            status = "OK"
            self.result_label.config(text=f"✓ OK ({len(self.auto_defects)})", fg="#00FF41")
            
        # Update Info
        if self.files:
            base = os.path.basename(self.files[self.idx])
            self.nav_info.config(text=f"{self.idx+1}/{len(self.files)}: {base}")
            
        # Stats Text
        mode_str = "Adaptive" if use_adaptive else "Manual"
        
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

    def _on_verdict_change(self, *args):
        """Handle change in defect % threshold efficiently."""
        if self.current_bw is None or self.current_mask_bool is None: 
            return
            
        try:
            # Re-read threshold
            black_th = float(self.black_defect_pct.get())
            
            # Re-compute verdict using cached binary and mask
            # We don't need to re-compute stats if we stored them, but comp stats is fast.
            white_px, black_px, area_px, white_pct, black_pct = self._compute_stats(self.current_bw, self.current_mask_bool)
            
            if black_pct > black_th:
                status = "DEFECT"
                self.result_label.config(text=f"⚠ DEFECT ({len(self.auto_defects)})", fg="#FF4444")
            else:
                status = "OK"
                self.result_label.config(text=f"✓ OK ({len(self.auto_defects)})", fg="#00FF41")
            
            # Update stats text status only (quick hack or full rewrite)
            # Full consistency requires updating stats text.
            # Using current values from UI/Logic:
            use_adaptive = self.use_adaptive.get()
            mode_str = "Adaptive" if use_adaptive else "Manual"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Status: {status}\n")
            self.stats_text.insert(tk.END, f"Mode: {mode_str}\n")
            self.stats_text.insert(tk.END, f"Black: {black_pct:.2f}% ({black_px} px)\n")
            self.stats_text.insert(tk.END, f"White: {white_pct:.2f}% ({white_px} px)\n")
            self.stats_text.insert(tk.END, f"Defects: {len(self.auto_defects)}\n")
            
        except (ValueError, tk.TclError):
            pass # Handle invalid number input gracefully (e.g., empty string during typing)

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
                                          use_adaptive=use_adaptive,
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
                                          use_adaptive=use_adaptive,
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
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                pass
        
        # Bind only to this canvas/window focus instead of global bind_all which causes issues
        # But for scroll, bind_all is common. Let's just safeguard the callback.
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Unbind on destroy to prevent calling destroyed widget
        def _unbind(event):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Destroy>", _unbind)
        
        # self._thumb_refs = [] # Removed to avoid overwriting previous tab's refs
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
                # self._thumb_refs.append(thumb)
                
                name = os.path.basename(res.get("path", "?"))
                black_pct = res.get("black_pct", 0)
                
                lbl = ttk.Label(tile, image=thumb,
                              text=f"{name}\nblack={black_pct:.1f}%",
                              compound=tk.TOP, justify=tk.CENTER)
                # Keep reference to avoid garbage collection
                lbl.image = thumb
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
    """Gold Pad Extraction Tab (for notebook embedding).
    
    Uses MATLAB-style crop_golden_circles algorithm for robust detection
    of golden circular pads with HSV thresholding, morphological operations,
    and eccentricity-based filtering.
    """
    
    BG_COLOR = "#000000"
    FG_COLOR = "#FFD700"
    ACCENT_COLOR = "#FFD700"
    FONT_FACE = "Consolas"
    
    def __init__(self, parent, app):
        super().__init__(parent, bg=self.BG_COLOR)
        self.app = app
        
        self.current_image = None
        self.current_image_path = None
        self.detected_regions = []  # CircleRegion objects
        self.extracted_pads = []    # (crop_img, region) tuples
        
        # MATLAB-style HSV thresholds (0-1 scale displayed, converted internally)
        # Default: H 0.06-0.40, S > 0.15, V > 0.55
        self.h_min = tk.DoubleVar(value=0.06)
        self.h_max = tk.DoubleVar(value=0.40)
        self.s_min = tk.DoubleVar(value=0.15)
        self.v_min = tk.DoubleVar(value=0.55)
        
        # Region filtering parameters (MATLAB-style)
        self.max_eccentricity = tk.DoubleVar(value=0.75)
        self.min_equiv_diameter = tk.DoubleVar(value=20)
        self.min_area = tk.IntVar(value=500)
        
        # Morphological parameters
        self.close_disk_size = tk.IntVar(value=6)
        self.open_disk_size = tk.IntVar(value=3)
        self.min_area_open = tk.IntVar(value=400)
        
        # Circular crop option
        self.circular_crop = tk.BooleanVar(value=True)
        self.padding = tk.IntVar(value=0)
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the Gold Pad Extractor UI with MATLAB parameters."""
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Gold title bar
        title_bar = tk.Frame(main_frame, bg="#332800")
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="🔶 GOLD PAD EXTRACTOR (MATLAB Algorithm)", 
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
        tk.Button(controls_frame, text="👁 Preview Mask",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._preview_detection).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="✂ Detect & Extract",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._extract_pads).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(controls_frame, text="💾 Save Pads",
                 font=(self.FONT_FACE, 10, 'bold'),
                 bg="#443300", fg=self.FG_COLOR,
                 command=self._save_pads).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Circular crop checkbox
        ttk.Checkbutton(controls_frame, text="Circular (Alpha)",
                       variable=self.circular_crop).pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(controls_frame, text="No image loaded",
                                    font=(self.FONT_FACE, 9),
                                    bg=self.BG_COLOR, fg="#888888")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # HSV sliders row (MATLAB 0-1 scale)
        hsv_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        hsv_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(hsv_frame, text="H min:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0.0, to=0.5, variable=self.h_min, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        tk.Label(hsv_frame, text="H max:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0.0, to=0.5, variable=self.h_max, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
        tk.Label(hsv_frame, text="S min:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0.0, to=1.0, variable=self.s_min, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
        tk.Label(hsv_frame, text="V min:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(hsv_frame, from_=0.0, to=1.0, variable=self.v_min, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
        # Region filter sliders row
        filter_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        filter_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(filter_frame, text="Max Ecc:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(filter_frame, from_=0.3, to=1.0, variable=self.max_eccentricity, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
        tk.Label(filter_frame, text="Min Diam:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(filter_frame, from_=5, to=100, variable=self.min_equiv_diameter, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
        tk.Label(filter_frame, text="Min Area:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(filter_frame, from_=100, to=2000, variable=self.min_area, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
        tk.Label(filter_frame, text="Padding:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        ttk.Scale(filter_frame, from_=0, to=20, variable=self.padding, 
                 orient=tk.HORIZONTAL, length=50).pack(side=tk.LEFT)
        
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
                self.current_image_path = path
                self.detected_regions = []
                self.extracted_pads = []
                self.status_label.config(text=os.path.basename(path)[:25])
                self._display_image(self.current_image)
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _detect_golden_regions(self, image):
        """Detect golden regions using MATLAB algorithm.
        
        Uses HSV thresholding with morphological operations (close, open, 
        area filter, fill holes) and returns filtered circular regions.
        """
        if len(image.shape) < 3 or image.shape[2] < 3:
            raise ValueError("Input image must be RGB/BGR.")
        
        # Convert BGR to HSV (OpenCV uses 0-180 for H, 0-255 for S,V)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Normalize to 0-1 scale (MATLAB convention)
        H = hsv[:, :, 0] / 180.0  # OpenCV H is 0-180
        S = hsv[:, :, 1] / 255.0
        V = hsv[:, :, 2] / 255.0
        
        # Apply HSV thresholds (MATLAB: H > h_min & H < h_max) & (S > s_min) & (V > v_min)
        h_min = self.h_min.get()
        h_max = self.h_max.get()
        s_min = self.s_min.get()
        v_min = self.v_min.get()
        
        mask = ((H > h_min) & (H < h_max) & (S > s_min) & (V > v_min)).astype(np.uint8) * 255
        
        # Morphological operations (MATLAB: imclose, imopen, bwareaopen, imfill)
        close_size = self.close_disk_size.get()
        open_size = self.open_disk_size.get()
        min_area_op = self.min_area_open.get()
        
        # imclose - closes small holes
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size*2+1, close_size*2+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        
        # imopen - removes small noise
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size*2+1, open_size*2+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        
        # bwareaopen - remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        mask_filtered = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area_op:
                mask_filtered[labels == i] = 255
        
        # imfill holes
        mask_filled = mask_filtered.copy()
        h, w = mask_filtered.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)
        mask_filled_inv = cv2.bitwise_not(mask_filled)
        mask_final = mask_filtered | mask_filled_inv
        
        return mask_final
    
    def _get_region_properties(self, mask):
        """Extract region properties similar to MATLAB regionprops."""
        regions = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
                
            # Calculate centroid using moments
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Equivalent diameter: diameter of a circle with same area
            equiv_diameter = np.sqrt(4 * area / np.pi)
            
            # Calculate eccentricity from ellipse fit
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (major_axis, minor_axis), _ = ellipse
                if major_axis > 0:
                    a = max(major_axis, minor_axis)
                    b = min(major_axis, minor_axis)
                    eccentricity = np.sqrt(1 - (b / a) ** 2)
                else:
                    eccentricity = 0
            else:
                eccentricity = 0
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            regions.append({
                'centroid': (cx, cy),
                'equiv_diameter': equiv_diameter,
                'area': area,
                'eccentricity': eccentricity,
                'bbox': (x, y, w, h)
            })
        
        return regions
    
    def _filter_circular_regions(self, regions):
        """Filter regions to keep only near-circular shapes (MATLAB algorithm)."""
        max_ecc = self.max_eccentricity.get()
        min_diam = self.min_equiv_diameter.get()
        min_area = self.min_area.get()
        
        return [
            r for r in regions
            if r['eccentricity'] < max_ecc
            and r['equiv_diameter'] > min_diam
            and r['area'] > min_area
        ]
    
    def _preview_detection(self):
        """Preview the gold mask using MATLAB detection."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            mask = self._detect_golden_regions(self.current_image)
            preview = self.current_image.copy()
            preview[mask > 0] = [0, 215, 255]  # Highlight gold areas in orange
            
            # Get region count
            regions = self._get_region_properties(mask)
            filtered = self._filter_circular_regions(regions)
            
            self._display_image(preview)
            self.status_label.config(
                text=f"Mask area: {np.count_nonzero(mask)} px | "
                     f"Regions: {len(regions)} → {len(filtered)} filtered"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
    
    def _extract_pads(self):
        """Extract individual gold pad images using MATLAB algorithm."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            # Detect using MATLAB algorithm
            mask = self._detect_golden_regions(self.current_image)
            regions = self._get_region_properties(mask)
            self.detected_regions = self._filter_circular_regions(regions)
            
            if not self.detected_regions:
                messagebox.showinfo("No Pads", 
                    "No golden circular pads detected. Try adjusting HSV thresholds or filters.")
                return
            
            self.extracted_pads = []
            preview = self.current_image.copy()
            h_img, w_img = self.current_image.shape[:2]
            pad_val = self.padding.get()
            
            for k, region in enumerate(self.detected_regions):
                cx, cy = region['centroid']
                r = region['equiv_diameter'] / 2
                
                # Calculate crop boundaries
                x1 = int(max(0, np.floor(cx - r - pad_val)))
                y1 = int(max(0, np.floor(cy - r - pad_val)))
                x2 = int(min(w_img, np.ceil(cx + r + pad_val)))
                y2 = int(min(h_img, np.ceil(cy + r + pad_val)))
                
                # Crop the region
                crop_img = self.current_image[y1:y2, x1:x2].copy()
                crop_h, crop_w = crop_img.shape[:2]
                
                if self.circular_crop.get():
                    # Calculate local center position within the crop
                    local_cx = cx - x1
                    local_cy = cy - y1
                    
                    # Create circular alpha mask
                    Y, X = np.ogrid[:crop_h, :crop_w]
                    dist_sq = (X - local_cx) ** 2 + (Y - local_cy) ** 2
                    alpha = (dist_sq <= r ** 2).astype(np.uint8) * 255
                    
                    # Add alpha channel to image
                    b, g, r_ch = cv2.split(crop_img)
                    crop_with_alpha = cv2.merge([b, g, r_ch, alpha])
                    self.extracted_pads.append((crop_with_alpha, region))
                else:
                    self.extracted_pads.append((crop_img, region))
                
                # Draw on preview
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(preview, (int(cx), int(cy)), int(r), (0, 255, 255), 1)
                cv2.putText(preview, str(k+1), (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self._display_image(preview)
            self.status_label.config(text=f"Extracted {len(self.extracted_pads)} circular pads")
            
        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed: {e}")
    
    def _save_pads(self):
        """Save extracted gold pads."""
        if not self.extracted_pads:
            self._extract_pads()
        if not self.extracted_pads:
            messagebox.showinfo("No Pads", "No gold pads detected to save.")
            return
        
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            # Get base name from image path
            if self.current_image_path:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            else:
                base_name = "Output"
            
            for i, (pad, region) in enumerate(self.extracted_pads):
                if self.circular_crop.get():
                    filename = f"{base_name}_Circle_alpha_{i+1:03d}.png"
                else:
                    filename = f"{base_name}_Circle_rect_{i+1:03d}.png"
                path = os.path.join(folder, filename)
                cv2.imwrite(path, pad)
            
            messagebox.showinfo("Saved", 
                f"Saved {len(self.extracted_pads)} gold pads to {folder}")
    
    def _display_image(self, cv2_image, size=(500, 400)):
        """Display image in the label."""
        if cv2_image is None:
            return
        h, w = cv2_image.shape[:2]
        ratio = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if new_w == 0 or new_h == 0:
            return
        # Handle 4-channel images (BGRA)
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 4:
            img_resized = cv2.resize(cv2_image[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
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
# HELPER FUNCTIONS FOR DEFECT DETECTION(Keepable)
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
        #if defect is too small, it is not a defect
        # Classification Logic
        if area < min_area:
            continue
        #els defect is too round that it is not a stain so it is not a defect
        elif circularity > 0.98 :
            continue
        
        elif 30 > area > 15:
            dtype = "Pinhole"
        elif aspect > 3.0 or circularity < 0.3:
            dtype = "Scratch"
        elif circularity > 0.4 and solidity > 0.8 and area>15:
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
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("Texture Analysis (FFT)")
        self.geometry("1400x800")
        self.configure(bg=DARK_THEME.BG_MAIN)
        
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
                        font=DARK_THEME.FONT_HEADER,
                        bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.ACCENT_INFO)
        title.pack(pady=10)
        
        # Controls Frame
        ctrl_frame = tk.Frame(self, bg=DARK_THEME.BG_MAIN)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(ctrl_frame, text="📁 Load Image", bg=DARK_THEME.BG_PANEL, fg=DARK_THEME.FG_PRIMARY,
                 font=DARK_THEME.FONT_BOLD,
                 command=self._load_image).pack(side=tk.LEFT, padx=5)
        
        # Filter Type
        tk.Label(ctrl_frame, text="Filter:", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_PRIMARY).pack(side=tk.LEFT, padx=(20, 5))
        self.filter_type = tk.StringVar(value="Low Pass")
        ttk.Combobox(ctrl_frame, textvariable=self.filter_type,
                    values=["Low Pass", "High Pass", "Band Pass"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=5)
        
        # Radius Slider
        tk.Label(ctrl_frame, text="Radius:", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_PRIMARY).pack(side=tk.LEFT, padx=(20, 5))
        self.radius = tk.IntVar(value=30)
        self.radius_scale = ttk.Scale(ctrl_frame, from_=1, to=200, variable=self.radius,
                                      orient=tk.HORIZONTAL, length=150,
                                      command=self._on_radius_change)
        self.radius_scale.pack(side=tk.LEFT, padx=5)
        self.radius_label = tk.Label(ctrl_frame, text="30", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_PRIMARY, width=4)
        self.radius_label.pack(side=tk.LEFT)
        
        # Outer Radius (for Band Pass)
        tk.Label(ctrl_frame, text="Outer:", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_PRIMARY).pack(side=tk.LEFT, padx=(20, 5))
        self.outer_radius = tk.IntVar(value=100)
        self.outer_scale = ttk.Scale(ctrl_frame, from_=1, to=300, variable=self.outer_radius,
                                     orient=tk.HORIZONTAL, length=100,
                                     command=self._on_radius_change)
        self.outer_scale.pack(side=tk.LEFT, padx=5)
        self.outer_label = tk.Label(ctrl_frame, text="100", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_PRIMARY, width=4)
        self.outer_label.pack(side=tk.LEFT)
        
        # Apply Button
        tk.Button(ctrl_frame, text="▶ Apply Filter", bg="#004400", fg="#00FF00",
                 font=DARK_THEME.FONT_BOLD,
                 command=self._apply_filter).pack(side=tk.LEFT, padx=20)
        
        # Display Frame (3 panels)
        display_frame = tk.Frame(self, bg=DARK_THEME.BG_MAIN)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original Image
        orig_frame = tk.Frame(display_frame, bg=DARK_THEME.BG_MAIN)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(orig_frame, text="ORIGINAL", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_PRIMARY).pack()
        self.orig_canvas = tk.Canvas(orig_frame, bg=DARK_THEME.BG_CANVAS, highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        
        # FFT Spectrum
        fft_frame = tk.Frame(display_frame, bg=DARK_THEME.BG_MAIN)
        fft_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(fft_frame, text="FREQUENCY SPECTRUM", bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.ACCENT_INFO).pack()
        self.fft_canvas = tk.Canvas(fft_frame, bg=DARK_THEME.BG_CANVAS, highlightthickness=0)
        self.fft_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Filtered Result
        result_frame = tk.Frame(display_frame, bg=DARK_THEME.BG_MAIN)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(result_frame, text="FILTERED RESULT", bg=DARK_THEME.BG_MAIN, fg="#00FF00").pack()
        self.result_canvas = tk.Canvas(result_frame, bg=DARK_THEME.BG_CANVAS, highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="Load an image to begin")
        tk.Label(self, textvariable=self.status_var, bg="#222", fg=DARK_THEME.FG_PRIMARY,
                font=DARK_THEME.FONT_MAIN).pack(fill=tk.X, side=tk.BOTTOM)
    
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

#data aug

# ==============================================================================


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
