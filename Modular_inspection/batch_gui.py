"""Batch processing GUI for multi-strip PCB inspection."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from typing import List, Optional
import cv2

from inspection.batch_processor import BatchInspector
from inspection.layout_visualizer import create_defect_layout, create_comparison_layout
from inspection.strip_extractor import extract_strips


class BatchProcessingTab(ttk.Frame):
    """GUI tab for batch inspection workflow.
    
    Provides interface for:
    - Master image selection
    - Test image folder/file selection
    - Batch processing execution
    - Progress tracking
    - Results visualization
    - Export functionality
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # State variables
        self.master_image_path: Optional[str] = None
        self.test_image_paths: List[str] = []
        self.batch_inspector: Optional[BatchInspector] = None
        self.processing = False
        
        # Results storage
        self.current_results = []
        self.defect_layout_image = None
        
        # Create UI
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the batch processing UI."""
        
        # Main container with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # === SECTION 1: Master Image ===
        master_section = ttk.LabelFrame(main_frame, text="1. Master Image (Golden Template)", 
                                       padding="10")
        master_section.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.master_label = ttk.Label(master_section, text="No master image loaded",
                                     foreground="gray")
        self.master_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        ttk.Button(master_section, text="Load Master Image", 
                  command=self._load_master_image).grid(row=0, column=1, padx=5)
        
        # === SECTION 2: Test Images ===
        test_section = ttk.LabelFrame(main_frame, text="2. Test Images", padding="10")
        test_section.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(test_section, text="Add Single Image",
                  command=self._add_single_image).grid(row=0, column=0, padx=5)
        
        ttk.Button(test_section, text="Add Folder",
                  command=self._add_folder).grid(row=0, column=1, padx=5)
        
        ttk.Button(test_section, text="Clear All",
                  command=self._clear_test_images).grid(row=0, column=2, padx=5)
        
        # Test images list
        list_frame = ttk.Frame(test_section)
        list_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.test_listbox = tk.Listbox(list_frame, height=6, 
                                       yscrollcommand=scrollbar.set)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.test_listbox.yview)
        
        # === SECTION 3: Configuration ===
        config_section = ttk.LabelFrame(main_frame, text="3. Configuration", padding="10")
        config_section.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Strip count
        ttk.Label(config_section, text="Strips per panel:").grid(row=0, column=0, sticky=tk.W)
        self.strip_count_var = tk.IntVar(value=6)
        strip_spinbox = ttk.Spinbox(config_section, from_=1, to=12, 
                                   textvariable=self.strip_count_var, width=10)
        strip_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Pixel threshold
        ttk.Label(config_section, text="Pixel Threshold:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.pixel_thresh_var = tk.IntVar(value=30)
        ttk.Spinbox(config_section, from_=10, to=100, 
                   textvariable=self.pixel_thresh_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Count threshold
        ttk.Label(config_section, text="Count Threshold:").grid(row=1, column=0, sticky=tk.W)
        self.count_thresh_var = tk.IntVar(value=5)
        ttk.Spinbox(config_section, from_=1, to=50, 
                   textvariable=self.count_thresh_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # SSIM threshold
        ttk.Label(config_section, text="SSIM Threshold:").grid(row=1, column=2, sticky=tk.W, padx=(20,0))
        self.ssim_thresh_var = tk.DoubleVar(value=0.97)
        ttk.Spinbox(config_section, from_=0.80, to=0.99, increment=0.01,
                   textvariable=self.ssim_thresh_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # === SECTION 4: Processing ===
        process_section = ttk.LabelFrame(main_frame, text="4. Processing", padding="10")
        process_section.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.run_button = ttk.Button(process_section, text="▶ Run Batch Inspection",
                                     command=self._run_batch_inspection)
        self.run_button.grid(row=0, column=0, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_section, variable=self.progress_var,
                                           maximum=100, length=300)
        self.progress_bar.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(process_section, textvariable=self.status_text,
                 foreground="blue").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        process_section.columnconfigure(1, weight=1)
        
        # === SECTION 5: Results ===
        results_section = ttk.LabelFrame(main_frame, text="5. Results", padding="10")
        results_section.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Results text area
        result_frame = ttk.Frame(results_section)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        result_scroll = ttk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(result_frame, height=8, width=80,
                                   yscrollcommand=result_scroll.set,
                                   font=("Consolas", 9))
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.config(command=self.result_text.yview)
        
        # Export buttons
        export_frame = ttk.Frame(results_section)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export CSV",
                  command=self._export_csv).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(export_frame, text="Save Defect Layout",
                  command=self._save_defect_layout).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(export_frame, text="View Layout",
                  command=self._view_defect_layout).pack(side=tk.LEFT, padx=5)
        
        main_frame.rowconfigure(4, weight=1)
    
    def _load_master_image(self):
        """Load master/golden image."""
        filepath = filedialog.askopenfilename(
            title="Select Master Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.master_image_path = filepath
            self.master_label.config(
                text=f"✓ {os.path.basename(filepath)}",
                foreground="green"
            )
    
    def _add_single_image(self):
        """Add a single test image."""
        filepath = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if filepath and filepath not in self.test_image_paths:
            self.test_image_paths.append(filepath)
            self.test_listbox.insert(tk.END, os.path.basename(filepath))
    
    def _add_folder(self):
        """Add all images from a folder."""
        folder = filedialog.askdirectory(title="Select Folder with Test Images")
        
        if folder:
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
            files = [os.path.join(folder, f) for f in os.listdir(folder)
                    if os.path.splitext(f)[1] in image_exts]
            
            added = 0
            for filepath in files:
                if filepath not in self.test_image_paths:
                    self.test_image_paths.append(filepath)
                    self.test_listbox.insert(tk.END, os.path.basename(filepath))
                    added += 1
            
            messagebox.showinfo("Folder Added", f"Added {added} images from folder.")
    
    def _clear_test_images(self):
        """Clear all test images."""
        self.test_image_paths.clear()
        self.test_listbox.delete(0, tk.END)
    
    def _run_batch_inspection(self):
        """Run batch inspection in background thread."""
        
        # Validation
        if not self.master_image_path:
            messagebox.showwarning("Missing Master", "Please load a master image first.")
            return
        
        if not self.test_image_paths:
            messagebox.showwarning("Missing Test Images", "Please add test images.")
            return
        
        if self.processing:
            messagebox.showinfo("Processing", "Batch inspection already running.")
            return
        
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        self.current_results.clear()
        
        # Start processing in background
        self.processing = True
        self.run_button.config(state='disabled')
        
        thread = threading.Thread(target=self._batch_processing_worker, daemon=True)
        thread.start()
    
    def _batch_processing_worker(self):
        """Background worker for batch processing."""
        try:
            # Initialize batch inspector
            self.batch_inspector = BatchInspector(
                self.master_image_path,
                pixel_thresh=self.pixel_thresh_var.get(),
                count_thresh=self.count_thresh_var.get(),
                ssim_threshold=self.ssim_thresh_var.get(),
                num_strips=self.strip_count_var.get()
            )
            
            total_images = len(self.test_image_paths)
            
            # Process each image
            for idx, test_path in enumerate(self.test_image_paths):
                # Update progress
                progress = ((idx + 1) / total_images) * 100
                self.progress_var.set(progress)
                self.status_text.set(f"Processing {idx+1}/{total_images}: {os.path.basename(test_path)}")
                
                # Inspect image
                result = self.batch_inspector.inspect_image(test_path)
                self.current_results.append(result)
                
                # Update results display
                self._append_result(result)
            
            # Generate defect layout
            self.status_text.set("Generating defect layout...")
            self._generate_defect_layout()
            
            # Show summary
            summary = self.batch_inspector.get_summary()
            self._show_summary(summary)
            
            self.status_text.set(f"✓ Completed: {summary['total_images']} images, "
                               f"{summary['defective_strips']} defects found")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error during batch processing:\n{e}")
            self.status_text.set("✗ Processing failed")
        
        finally:
            self.processing = False
            self.run_button.config(state='normal')
            self.progress_var.set(0)
    
    def _append_result(self, result: dict):
        """Append result to text display."""
        self.result_text.insert(tk.END, f"\n{'='*60}\n")
        self.result_text.insert(tk.END, f"Image: {os.path.basename(result['test_image_path'])}\n")
        self.result_text.insert(tk.END, f"Verdict: {result['verdict']}\n")
        self.result_text.insert(tk.END, f"Defects: {result['defect_count']}/{result['total_strips']} strips\n")
        
        for strip_res in result['strip_results']:
            status_icon = "✓" if strip_res['verdict'] == 'NORMAL' else "✗"
            self.result_text.insert(tk.END,
                f"  {status_icon} Strip {strip_res['strip_number']}: {strip_res['verdict']} "
                f"(SSIM: {strip_res.get('ssim_score', 0):.3f})\n")
        
        self.result_text.see(tk.END)
    
    def _show_summary(self, summary: dict):
        """Show summary statistics."""
        self.result_text.insert(tk.END, f"\n{'='*60}\n")
        self.result_text.insert(tk.END, "BATCH SUMMARY\n")
        self.result_text.insert(tk.END, f"{'='*60}\n")
        self.result_text.insert(tk.END, f"Total Images:      {summary['total_images']}\n")
        self.result_text.insert(tk.END, f"Total Strips:      {summary['total_strips']}\n")
        self.result_text.insert(tk.END, f"Pass Count:        {summary['pass_count']}\n")
        self.result_text.insert(tk.END, f"Fail Count:        {summary['fail_count']}\n")
        self.result_text.insert(tk.END, f"Defective Strips:  {summary['defective_strips']}\n")
        self.result_text.insert(tk.END, f"Defect Rate:       {summary['defect_rate']}\n")
        self.result_text.see(tk.END)
    
    def _generate_defect_layout(self):
        """Generate defect layout visualization."""
        if not self.current_results or not self.batch_inspector:
            return
        
        # Combine all strip results
        all_strip_results = []
        for result in self.current_results:
            all_strip_results.extend(result['strip_results'])
        
        # Create layout
        self.defect_layout_image = create_defect_layout(
            all_strip_results,
            strip_images=None,  # Could add actual strip images if needed
            layout_type="grid",
            title="Batch Inspection - Defect Layout"
        )
    
    def _export_csv(self):
        """Export results to CSV."""
        if not self.batch_inspector:
            messagebox.showinfo("No Results", "No inspection results to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="batch_inspection_results.csv"
        )
        
        if filepath:
            self.batch_inspector.export_results_csv(filepath)
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filepath}")
    
    def _save_defect_layout(self):
        """Save defect layout image."""
        if self.defect_layout_image is None:
            messagebox.showinfo("No Layout", "No defect layout generated yet.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile="defect_layout.png"
        )
        
        if filepath:
            cv2.imwrite(filepath, self.defect_layout_image)
            messagebox.showinfo("Save Complete", f"Defect layout saved to:\n{filepath}")
    
    def _view_defect_layout(self):
        """View defect layout in a new window."""
        if self.defect_layout_image is None:
            messagebox.showinfo("No Layout", "No defect layout generated yet.")
            return
        
        # Create new window
        viewer = tk.Toplevel(self)
        viewer.title("Defect Layout Visualization")
        
        # Convert to PhotoImage
        import cv2
        from PIL import Image, ImageTk
        
        rgb_image = cv2.cvtColor(self.defect_layout_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Resize if too large
        max_width, max_height = 1200, 900
        if pil_image.width > max_width or pil_image.height > max_height:
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_image)
        
        label = tk.Label(viewer, image=photo)
        label.image = photo  # Keep a reference
        label.pack(padx=10, pady=10)
