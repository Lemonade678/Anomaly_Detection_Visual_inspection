"""
Anomaly Detection Demo - Hybrid Inspection Pipeline
====================================================

This file demonstrates the visual inspection pipeline in a clean, readable way.
It can be used as:
1. A standalone demo script (python hybrid.py)
2. An educational reference for understanding the detection workflow

The full GUI application is in: inspection/gui.py
"""

# ==============================================================================
# PIPELINE OVERVIEW
# ==============================================================================
#
# The anomaly detection follows this workflow:
#
#   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
#   │ Load Golden  │ --> │  Load Test   │ --> │    Align     │
#   │   (Master)   │     │    Image     │     │  (ORB/SIFT)  │
#   └──────────────┘     └──────────────┘     └──────────────┘
#             │                                       │
#             v                                       v
#   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
#   │  SSIM Check  │ <-- │   Compare    │ <-- │   Resize to  │
#   │  (Stage 1)   │     │              │     │    Match     │
#   └──────────────┘     └──────────────┘     └──────────────┘
#             │
#             v
#   ┌──────────────────────────────────────────────┐
#   │ SSIM > 0.975?  ──YES──>  ✓ NORMAL (PASS)    │
#   │      │                                       │
#   │     NO                                       │
#   │      v                                       │
#   │ Pixel Matching (Stage 2)                     │
#   │      │                                       │
#   │      v                                       │
#   │ Anomalous Pixels > Threshold?               │
#   │    YES -> ✗ ANOMALY DETECTED                │
#   │    NO  -> ✓ NORMAL                          │
#   └──────────────────────────────────────────────┘
#
# ==============================================================================


def run_inspection_demo():
    """
    Demonstrates the inspection pipeline step by step.
    
    This function shows how to use the inspection modules programmatically
    without the GUI, useful for automation or batch scripting.
    """
    import cv2
    import numpy as np
    
    # Import the core inspection modules
    from inspection.io import read_image
    from inspection.align_revamp import align_images
    from inspection.ssim import calc_ssim
    from inspection.pixel_match import run_pixel_matching
    
    # Configuration
    SSIM_THRESHOLD = 0.975
    PIXEL_DIFF_THRESHOLD = 40
    COUNT_THRESHOLD = 1000
    
    print("=" * 60)
    print("ANOMALY DETECTION PIPELINE DEMO")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Images
    # -------------------------------------------------------------------------
    print("\n[1] LOAD IMAGES")
    
    # In a real scenario, you would load actual images:
    # golden_image = read_image("path/to/golden_template.png")
    # test_image = read_image("path/to/test_image.png")
    
    # For demo, create synthetic images
    print("    Creating synthetic test images...")
    golden_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    # Add recognizable features for ORB
    cv2.circle(golden_image, (100, 100), 30, (255, 0, 0), -1)
    cv2.rectangle(golden_image, (200, 200), (300, 300), (0, 255, 0), -1)
    cv2.circle(golden_image, (400, 400), 50, (0, 0, 255), -1)
    
    # Test image: copy with slight modification (simulated defect)
    test_image = golden_image.copy()
    # Add a small "defect" 
    cv2.circle(test_image, (350, 150), 15, (255, 255, 255), -1)
    
    print(f"    Golden image shape: {golden_image.shape}")
    print(f"    Test image shape:   {test_image.shape}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Align Images
    # -------------------------------------------------------------------------
    print("\n[2] ALIGN IMAGES (ORB Feature Matching)")
    
    aligned_image, (dx, dy), confidence = align_images(golden_image, test_image)
    
    print(f"    Translation: dx={dx:.2f}, dy={dy:.2f}")
    print(f"    Match confidence: {confidence:.4f}")
    
    if confidence < 0.1:
        print("    ⚠ WARNING: Low alignment confidence!")
        return None
    
    # -------------------------------------------------------------------------
    # STEP 3: SSIM Pre-Check (Stage 1)
    # -------------------------------------------------------------------------
    print("\n[3] SSIM STRUCTURAL CHECK (Stage 1)")
    
    ssim_score, ssim_heatmap = calc_ssim(golden_image, aligned_image)
    
    print(f"    SSIM Score: {ssim_score:.4f}")
    print(f"    Threshold:  {SSIM_THRESHOLD}")
    
    if ssim_score > SSIM_THRESHOLD:
        print(f"\n    [OK] RESULT: NORMAL (SSIM Pass)")
        print("    >> Pixel analysis skipped")
        return {
            'verdict': 'Normal',
            'method': 'SSIM',
            'ssim_score': ssim_score
        }
    
    print(f"    [!] SSIM failed, proceeding to pixel analysis...")
    
    # -------------------------------------------------------------------------
    # STEP 4: Pixel Matching (Stage 2)
    # -------------------------------------------------------------------------
    print("\n[4] PIXEL MATCHING (Stage 2)")
    
    pixel_result = run_pixel_matching(
        golden_image, 
        aligned_image,
        pixel_thresh=PIXEL_DIFF_THRESHOLD,
        count_thresh=COUNT_THRESHOLD
    )
    
    print(f"    Area Score:       {pixel_result['area_score']:.2f}%")
    print(f"    Anomalous Pixels: {pixel_result.get('anomalous_pixel_count', 'N/A')}")
    print(f"    Contour Count:    {pixel_result['anomaly_count']}")
    print(f"    Pixel Verdict:    {pixel_result['verdict']}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Final Verdict
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    if pixel_result['verdict'] == "Anomaly":
        print("FINAL VERDICT: !! ANOMALY DETECTED !!")
    else:
        print("FINAL VERDICT: NORMAL")
    print("=" * 60)
    
    return {
        'verdict': pixel_result['verdict'],
        'method': 'Pixel Matching',
        'ssim_score': ssim_score,
        'area_score': pixel_result['area_score'],
        'anomaly_count': pixel_result['anomaly_count'],
        'heatmap': pixel_result['heatmap'],
        'contour_map': pixel_result['contour_map']
    }


def run_gui():
    """Launch the full GUI application."""
    from inspection.gui import InspectorProApp
    
    print("Launching Anomaly Detection GUI...")
    app = InspectorProApp()
    app.mainloop()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run the demo pipeline
        result = run_inspection_demo()
        if result:
            print(f"\nReturned result: {result['verdict']}")
    else:
        # Launch the GUI (default)
        run_gui()
