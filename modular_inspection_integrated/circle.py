import cv2
import numpy as np
import os

def extract_pcb_pads(image_path, output_folder='extracted_pads'):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Preprocessing & Color Segmentation (Robustness Step 1)
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for Gold/Yellow color
    # Note: You may need to tune these slightly depending on exact lighting
    lower_gold = np.array([15, 60, 60])   # Hue ~20 is Yellow/Orange
    upper_gold = np.array([35, 255, 255]) 

    # Create a binary mask
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # 3. Morphological Operations (Robustness Step 2)
    # Removes small noise (white specks) inside the background
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Optional: Dilate slightly to ensure the full circle is captured
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

    # 4. Find Contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} potential candidates.")

    pad_count = 0
    debug_img = img.copy()

    for cnt in contours:
        # 5. Geometric Filtering (Robustness Step 3)
        
        # Area Filter: Remove tiny noise or huge glitches
        area = cv2.contourArea(cnt)
        if area < 300 or area > 5000: 
            continue

        # Aspect Ratio Filter: Ensure it's roughly square (1:1)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if not (0.8 < aspect_ratio < 1.2): 
            continue

        # Circularity Filter: (4 * pi * Area) / (Perimeter^2)
        # A perfect circle has a circularity of 1.0
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Allow slight imperfections (0.7 - 1.2)
        if circularity < 0.7: 
            continue

        # 6. Extraction
        # Add a small padding to the crop so we don't cut the edges
        padding = 5
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(img.shape[1] - x_pad, w + 2*padding)
        h_pad = min(img.shape[0] - y_pad, h + 2*padding)

        roi = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        
        # Save the individual pad
        pad_count += 1
        cv2.imwrite(f"{output_folder}/pad_{pad_count}_456.png", roi)

        # Draw on debug image for visualization
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(pad_count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show results
    # (Resize for display if the image is huge)
    scale_percent = 50 
    width = int(debug_img.shape[1] * scale_percent / 100)
    height = int(debug_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_debug = cv2.resize(debug_img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Detected Pads", resized_debug)
    cv2.imshow("Mask", cv2.resize(mask_clean, dim))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Successfully extracted {pad_count} pads.")

# Usage
extract_pcb_pads(r'C:\Users\User\Downloads\antigravity_soft\456.png')                                                                                                                                                                                             
