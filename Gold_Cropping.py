import cv2
import numpy as np
import os
from tkinter import filedialog, Tk

def crop_golden_circles(IorFile, outDir="crops", circularCrop=True):
    # Ensure output directory exists
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # --- Read image and extract Base Name ---
    if isinstance(IorFile, str):
        baseName = os.path.splitext(os.path.basename(IorFile))[0]
        img = cv2.imread(IorFile)
        if img is None:
            print(f"Error: Could not read image {IorFile}")
            return
    else:
        baseName = "Output"
        img = IorFile

    # OpenCV uses BGR by default, convert to RGB for logic consistency if needed, 
    # but here we go straight to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- 2) Apply a threshold to find gold-ish regions ---
    # MATLAB HSV ranges: H[0,1], S[0,1], V[0,1]
    # OpenCV HSV ranges: H[0,180], S[0,255], V[0,255]
    lower_gold = np.array([int(0.06*180), int(0.15*255), int(0.55*255)])
    upper_gold = np.array([int(0.40*180), 255, 255])
    
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # Morphological operations (Close -> Open -> Fill)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)) # disk 6
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))    # disk 3
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # --- 3) Region Props (Contour Analysis) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    saved_count = 0
    img_h, img_w = img.shape[:2]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500 or area > 90000:
            continue
            
        # Get bounding circle/ellipse for eccentricity/diameter equivalent
        if len(cnt) < 5: continue # Need points for ellipse
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        
        # Eccentricity approx
        eccentricity = np.sqrt(1 - (min(MA, ma)**2 / max(MA, ma)**2))
        equiv_diameter = np.sqrt(4 * area / np.pi)

        # Filter: ecc < 0.75, eqD > 20
        if eccentricity < 0.75 and equiv_diameter > 20:
            saved_count += 1
            r = equiv_diameter / 2
            
            # --- 4) Crop ---
            x1, y1 = int(max(0, x - r)), int(max(0, y - r))
            x2, y2 = int(min(img_w, x + r)), int(min(img_h, y + r))
            
            crop_img = img[y1:y2, x1:x2]
            h_c, w_c = crop_img.shape[:2]

            if circularCrop:
                # Create alpha channel
                cx, cy = x - x1, y - y1
                yy, xx = np.ogrid[:h_c, :w_c]
                dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                alpha = np.where(dist_from_center <= r, 255, 0).astype(np.uint8)
                
                # Merge BGR and Alpha
                b, g, r_chan = cv2.split(crop_img)
                rgba = cv2.merge([b, g, r_chan, alpha])
                
                out_name = os.path.join(outDir, f"{baseName}_Circle_alpha_{saved_count:03d}.png")
                cv2.imwrite(out_name, rgba)
            else:
                out_name = os.path.join(outDir, f"{baseName}_Circle_rect_{saved_count:03d}.png")
                cv2.imwrite(out_name, crop_img)

    print(f"Saved {saved_count} crops to folder: {outDir}")

# --- UI Selection ---
if __name__ == "__main__":
    root = Tk()
    root.withdraw() # Hide the main tkinter window

    file_path = filedialog.askopenfilename(title="Select an Image File", 
                                          filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    if not file_path:
        print("No file selected")
    else:
        output_folder = filedialog.askdirectory(title="Select Output Folder")
        if not output_folder:
            output_folder = "crops"
            
        crop_golden_circles(file_path, output_folder, circularCrop=True)