"""Integrated Hybrid Inspector - MAIN ENTRY POINT for PCB Inspection.

This is the PRIMARY processing module for the Anomaly Detection Visual Inspection system.
Use this file as the main entry point instead of modular_inspection_integrated/gui.py.

Usage (Command Line):
    python program_demo.py              # Launch GUI (default)
    python program_demo.py --demo       # Run pipeline demo
    python program_demo.py --grid-demo  # Run grid analysis demo
    python program_demo.py --batch ...  # Run batch mode
    python program_demo.py --help       # Show help
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modular_inspection_integrated.config import AlignmentMethod, LightSensitivityMode
from modular_inspection_integrated.io import read_image
from modular_inspection_integrated.pipeline import run_inspection, run_inspection_with_config
from modular_inspection_integrated.grid_analyzer import GridAnalyzer
from modular_inspection_integrated.theme import DARK_THEME

# ==============================================================================
# DEMO FUNCTIONS
# ==============================================================================

def run_inspection_demo():
    """Demonstrates the integrated inspection pipeline."""
    
    print("=" * 60)
    print("INTEGRATED ANOMALY DETECTION PIPELINE DEMO")
    print("=" * 60)
    
    # Create synthetic test images
    print("\nCreating synthetic test images...")
    golden_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    # Add recognizable features
    cv2.circle(golden_image, (100, 100), 30, (255, 0, 0), -1)
    cv2.rectangle(golden_image, (200, 200), (300, 300), (0, 255, 0), -1)
    cv2.circle(golden_image, (400, 400), 50, (0, 0, 255), -1)
    
    # Test image with defect
    test_image = golden_image.copy()
    cv2.circle(test_image, (350, 150), 15, (255, 255, 255), -1)  # Simulated defect
    
    print(f"Golden image shape: {golden_image.shape}")
    print(f"Test image shape:   {test_image.shape}")
    
    # Run inspection
    result = run_inspection(
        golden_image, 
        test_image,
        alignment_method=AlignmentMethod.AUTO,
        light_mode=LightSensitivityMode.AUTO,
        use_multi_scale=False,
        verbose=True
    )
    
    print(f"\nReturned result: {result['verdict']}")
    return result


def run_grid_analysis_demo():
    """Demo of 9-part grid analysis."""
    
    print("\n" + "=" * 60)
    print("GRID ANALYSIS DEMO (9-Part)")
    print("=" * 60)
    
    # Create test images
    golden_image = np.random.randint(100, 150, (600, 600, 3), dtype=np.uint8)
    test_image = golden_image.copy()
    
    # Add defects to specific grid sectors
    cv2.rectangle(test_image, (10, 10), (150, 150), (255, 255, 255), -1)  # Sector 0
    cv2.circle(test_image, (500, 500), 50, (0, 0, 0), -1)  # Sector 8
    
    analyzer = GridAnalyzer()
    results = analyzer.analyze_images(golden_image, test_image)
    
    print(f"Overall verdict: {results['verdict']}")
    print(f"Anomaly count: {results['anomaly_count']} / 9 segments")
    
    return results


# ==============================================================================
# BATCH INSPECTION CLI
# ==============================================================================

def run_batch_inspection_cli(input_dir: str, output_dir: str, golden_path: str = None):
    """Run batch inspection from command line."""
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    ok_dir = os.path.join(output_dir, "OK")
    defect_dir = os.path.join(output_dir, "DEFECT")
    error_dir = os.path.join(output_dir, "ERROR")
    
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    
    # Find Images
    import glob
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff','.raw','.bin','.tif']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    image_files = sorted(list(set(image_files)))
    total = len(image_files)
    
    if total == 0:
        print(f"No images found in {input_dir}")
        return

    # Determine Golden Image
    if not golden_path:
        potential_golden = [f for f in image_files if "golden" in os.path.basename(f).lower() or "ref" in os.path.basename(f).lower()]
        if potential_golden:
            golden_path = potential_golden[0]
            print(f"Auto-selected golden image: {os.path.basename(golden_path)}")
        elif image_files:
            golden_path = image_files[0]
            print(f"Warning: No golden/ref image found. Using first image as golden: {os.path.basename(golden_path)}")
        else:
            print("Error: Could not determine golden image.")
            return

    golden_img = read_image(golden_path)
    if golden_img is None:
        print(f"Error: Failed to load golden image: {golden_path}")
        return

    print(f"\nStarting Batch Inspection of {total} images...")
    
    import csv
    csv_path = os.path.join(output_dir, "batch_summary.csv")
    
    processed_count = 0
    anomalies_count = 0
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Verdict", "Confidence", "SSIM", "Area_Score", "Anomalies", "Processing_Time"])
        
        for i, img_path in enumerate(image_files):
            fname = os.path.basename(img_path)
            if os.path.abspath(img_path) == os.path.abspath(golden_path):
                continue
                
            print(f"Processing [{i+1}/{total}]: {fname}...", end="", flush=True)
            
            try:
                test_img = read_image(img_path)
                if test_img is None:
                    print(" Failed to load.")
                    continue
                
                result = run_inspection(
                    golden_img, test_img,
                    verbose=False
                )
                
                verdict = result['verdict']
                import shutil
                
                if verdict == 'Normal':
                    shutil.copy2(img_path, os.path.join(ok_dir, fname))
                    print(" OK")
                elif verdict == 'Anomaly':
                    anomalies_count += 1
                    if 'contour_map' in result:
                        cv2.imwrite(os.path.join(defect_dir, f"annotated_{fname}"), result['contour_map'])
                    shutil.copy2(img_path, os.path.join(defect_dir, fname))
                    print(" DEFECT")
                else:
                    shutil.copy2(img_path, os.path.join(error_dir, fname))
                    print(f" ERROR ({result.get('error', 'Unknown')})")
                
                writer.writerow([
                    fname, verdict, f"{result.get('confidence',0):.2f}",
                    f"{result.get('ssim_score',0):.4f}", f"{result.get('area_score',0):.2f}",
                    result.get('anomaly_count', 0), f"{result.get('processing_time',0):.3f}"
                ])
                processed_count += 1
                
            except Exception as e:
                print(f" Exception: {e}")
                import shutil
                shutil.copy2(img_path, os.path.join(error_dir, fname))

    print("-" * 60)
    print(f"Batch completed. Defects Found: {anomalies_count}")


# ==============================================================================
# GUI LAUNCHER
# ==============================================================================

def run_gui():
    """Launch the integrated GUI application."""
    from modular_inspection_integrated.gui import InspectorApp
    print("Launching Integrated Inspector GUI...")
    app = InspectorApp()
    app.mainloop()


class LauncherWindow:
    """Simple launcher window for choosing application mode."""
    
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.messagebox = messagebox
        
        self.root = tk.Tk()
        self.root.title("PCB Inspection Launcher")
        self.root.geometry("450x400")
        self.root.configure(bg=DARK_THEME.BG_MAIN)
        self.root.resizable(False, False)
        
        # Center window
        x = (self.root.winfo_screenwidth() // 2) - 225
        y = (self.root.winfo_screenheight() // 2) - 200
        self.root.geometry(f"+{x}+{y}")
        
        # UI Elements
        tk.Label(self.root, text="PCB Inspection System", 
                font=DARK_THEME.FONT_HEADER, bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.ACCENT_INFO).pack(pady=(20, 5))
        
        tk.Label(self.root, text="Select a mode to start", 
                font=DARK_THEME.FONT_MAIN, bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_SECONDARY).pack(pady=(0, 20))
        
        btn_style = {"font": DARK_THEME.FONT_BOLD, "width": 35, "height": 2,
                    "bg": DARK_THEME.BG_PANEL, "fg": DARK_THEME.ACCENT_PRIMARY, "activebackground": DARK_THEME.BORDER_FOCUS,
                    "activeforeground": DARK_THEME.ACCENT_HOVER, "relief": "flat", "cursor": "hand2"}
        
        tk.Button(self.root, text="▶ Standard Inspection", command=self._launch_main, **btn_style).pack(pady=5)
        
        tk.Button(self.root, text="📁 Batch Processing Info", command=self._show_batch_help, **btn_style).pack(pady=5)
        tk.Button(self.root, text="🔬 Run Demo", command=self._run_demo, **btn_style).pack(pady=5)
        
        tk.Label(self.root, text="v2.0 - Refactored MVP", font=DARK_THEME.FONT_MAIN,
                bg=DARK_THEME.BG_MAIN, fg=DARK_THEME.FG_DISABLED).pack(side=tk.BOTTOM, pady=10)
        
    def _launch_main(self):
        self.root.destroy()
        run_gui()
    

    def _show_batch_help(self):
        self.messagebox.showinfo("Batch Processing", 
            "For batch processing, use the command line:\n\n"
            "python program_demo.py --batch -i <input_dir> -o <output_dir>\n\n"
            "This ensures stability for large datasets.")
    
    def _run_demo(self):
        self.root.destroy()
        run_inspection_demo()
        run_grid_analysis_demo()
        print("\nDemo completed. Press Enter to exit...")
        input()
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated PCB Inspection System")
    parser.add_argument("--demo", action="store_true", help="Run inspection pipeline demo")
    parser.add_argument("--grid-demo", action="store_true", help="Run grid analysis demo")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("-i", "--input", help="Input directory for batch mode")
    parser.add_argument("-o", "--output", help="Output directory for batch mode")
    parser.add_argument("-g", "--golden", help="Golden image path for batch mode")
    
    args = parser.parse_args()
    
    if args.demo:
        run_inspection_demo()
    elif args.grid_demo:
        run_grid_analysis_demo()
    elif args.batch:
        if not args.input or not args.output:
            print("Error: Batch mode requires --input and --output")
        else:
            run_batch_inspection_cli(args.input, args.output, args.golden)
    else:
        # Default: Launch GUI
        LauncherWindow().run()
