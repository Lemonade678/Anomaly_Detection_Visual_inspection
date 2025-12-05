import cv2
import numpy as np
import os
import argparse
#WIPWWW
def run_edge_detection(image_or_path, debug_output_dir='cropped_substrates_automatic'):
    """
    Reads an image (path or array), detects substrates using Canny edge detection,
    and returns a list of cropped substrates (sorted Top-Left -> Right).
    """
    # --- 1. Load Your Original Image ---
    if isinstance(image_or_path, str):
        if not os.path.exists(image_or_path):
            print(f"Error: Image file not found at '{image_or_path}'")
            return []
        original_image = cv2.imread(image_or_path)
        if original_image is None:
            print(f"Error: Could not load '{image_or_path}'.")
            return []
        img_name = os.path.basename(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        original_image = image_or_path
        img_name = "memory_image.png"
    else:
        print("Error: Invalid input type. Expected str (path) or np.ndarray.")
        return []

    # --- 2. Perform Canny Edge Detection (Integrated) ---
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, 40, 125)
    
    # --- 3. Create Output Directory (for debug) ---
    if debug_output_dir:
        if not os.path.exists(debug_output_dir):
            os.makedirs(debug_output_dir)

    # --- 4. Find Contours (Shapes) from Canny Edges ---
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours were found.")
        return []

    # --- 5. Filter and Sort Contours ---
    valid_contours = []
    
    # Adjust these values based on your actual substrate size
    min_width = 80
    max_width = 240
    min_height = 200
    max_height = 800

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (min_width <= w <= max_width) and (min_height <= h <= max_height):
            valid_contours.append(contour)

    if not valid_contours:
        print("No valid substrates found after size filtering.")
        return []

    # Sort contours: Top-to-bottom, then Left-to-right
    # We use a tolerance for the y-coordinate to group items in the same row
    def sort_key(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y // 50) * 10000 + x # Group by row (approx 50px height tolerance), then sort by x

    sorted_contours = sorted(valid_contours, key=sort_key)

    # --- 6. Crop Substrates ---
    cropped_substrates = []
    
    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        crop = original_image[y:y+h, x:x+w]
        cropped_substrates.append(crop)
        
        if debug_output_dir:
            output_filename = os.path.join(debug_output_dir, f'crop_{i}_{img_name}')
            cv2.imwrite(output_filename, crop)
            # print(f"  - Cropped and saved '{output_filename}' (x={x}, y={y}, w={w}, h={h})")

    return cropped_substrates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Detection for Substrate Extraction')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    results = run_edge_detection(args.image_path)
    if results:
        print(f"Success: Got {len(results)} cropped images.")
        for i, res in enumerate(results):
            cv2.imshow(f"Cropped Result {i}", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to crop image.")