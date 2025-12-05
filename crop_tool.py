"""Interactive ROI/Crop tool for selecting regions of interest in images."""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class InteractiveCropTool:
    """Interactive ROI selection tool using OpenCV window."""
    
    def __init__(self, image: np.ndarray, window_name: str = "Select ROI - Crop Tool"):
        """Initialize crop tool with image.
        
        Args:
            image: Input image (BGR format)
            window_name: Name for the OpenCV window
        """
        self.original_image = image.copy()
        self.image = image.copy()
        self.window_name = window_name
        
        # ROI selection state
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.roi_confirmed = False
        self.roi_rect = None  # (x, y, w, h)
        
        # Display state
        self.zoom_level = 1.0
        self.display_image = self.image.copy()
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection."""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing rectangle
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update rectangle as mouse moves
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing rectangle
            self.drawing = False
            self.end_point = (x, y)
            
            # Calculate ROI rectangle (ensure positive width/height)
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            if w > 0 and h > 0:
                self.roi_rect = (x, y, w, h)
    
    def _draw_roi(self):
        """Draw current ROI on display image."""
        self.display_image = self.image.copy()
        
        # Draw current selection rectangle (while dragging)
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(self.display_image, self.start_point, self.end_point,
                         (0, 255, 0), 2)  # Green for active drawing
        
        # Draw confirmed ROI rectangle
        elif self.roi_rect:
            x, y, w, h = self.roi_rect
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h),
                         (0, 255, 255), 2)  # Yellow for confirmed
            
            # Add label
            label = f"ROI: {w}x{h}"
            cv2.putText(self.display_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw instructions
        self._draw_instructions()
    
    def _draw_instructions(self):
        """Draw instruction text overlay."""
        instructions = [
            "Click and drag to select region",
            "[C] Confirm  [R] Reset  [ESC] Cancel",
            "[+/-] Zoom  [Space] Toggle help"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(self.display_image, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA)
            # Add black outline for visibility
            cv2.putText(self.display_image, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                       cv2.LINE_AA)
            cv2.putText(self.display_image, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA)
    
    def select_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Run interactive ROI selection.
        
        Returns:
            ROI rectangle as (x, y, w, h) tuple, or None if cancelled
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Resize window to fit screen better
        h, w = self.image.shape[:2]
        max_h, max_w = 900, 1400
        
        if h > max_h or w > max_w:
            scale = min(max_w / w, max_h / h)
            cv2.resizeWindow(self.window_name, int(w * scale), int(h * scale))
        
        while True:
            # Update display
            self._draw_roi()
            cv2.imshow(self.window_name, self.display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):
                # Confirm ROI
                if self.roi_rect:
                    self.roi_confirmed = True
                    break
                    
            elif key == ord('r') or key == ord('R'):
                # Reset ROI
                self.start_point = None
                self.end_point = None
                self.roi_rect = None
                self.drawing = False
                
            elif key == 27:  # ESC
                # Cancel
                self.roi_rect = None
                break
                
            elif key == ord('+') or key == ord('='):
                # Zoom in (future enhancement)
                pass
                
            elif key == ord('-') or key == ord('_'):
                # Zoom out (future enhancement)
                pass
        
        cv2.destroyWindow(self.window_name)
        return self.roi_rect


def select_roi(image: np.ndarray, window_name: str = "Select ROI") -> Optional[Tuple[int, int, int, int]]:
    """Convenience function for quick ROI selection.
    
    Args:
        image: Input image (BGR format)
        window_name: Window title
        
    Returns:
        ROI rectangle as (x, y, w, h) tuple, or None if cancelled
        
    Example:
        >>> import cv2
        >>> img = cv2.imread('test.jpg')
        >>> roi = select_roi(img)
        >>> if roi:
        >>>     x, y, w, h = roi
        >>>     cropped = img[y:y+h, x:x+w]
    """
    tool = InteractiveCropTool(image, window_name)
    return tool.select_roi()


def crop_image(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using ROI coordinates.
    
    Args:
        image: Input image
        roi: ROI as (x, y, w, h)
        
    Returns:
        Cropped image
    """
    x, y, w, h = roi
    return image[y:y+h, x:x+w]


def select_multiple_rois(image: np.ndarray, 
                        num_rois: int = 6,
                        window_name: str = "Select Multiple ROIs") -> List[Tuple[int, int, int, int]]:
    """Select multiple ROIs sequentially.
    
    Args:
        image: Input image
        num_rois: Number of ROIs to select
        window_name: Window title base
        
    Returns:
        List of ROI rectangles [(x, y, w, h), ...]
    """
    rois = []
    
    for i in range(num_rois):
        tool = InteractiveCropTool(image, f"{window_name} - Region {i+1}/{num_rois}")
        roi = tool.select_roi()
        
        if roi is None:
            # User cancelled
            break
        
        rois.append(roi)
    
    return rois


def preview_crop(image: np.ndarray, roi: Tuple[int, int, int, int], 
                window_name: str = "Crop Preview") -> bool:
    """Show preview of cropped region.
    
    Args:
        image: Original image
        roi: ROI to preview
        window_name: Window title
        
    Returns:
        True if user accepts, False if rejected
    """
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    
    # Create preview with original ROI overlay
    preview = image.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Combine original and cropped view
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Show original with ROI
    cv2.imshow(window_name, preview)
    cv2.waitKey(1000)
    
    # Show cropped region
    cv2.imshow(window_name, cropped)
    
    print("Press 'y' to accept, 'n' to reject...")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y') or key == ord('Y'):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('n') or key == ord('N') or key == 27:
            cv2.destroyWindow(window_name)
            return False




class TkinterCropPreview:
    """Tkinter-based crop preview dialog with Accept/Reject buttons.
    
    Displays original and cropped images side-by-side for user review.
    """
    
    def __init__(self, original_image: np.ndarray, cropped_image: np.ndarray, 
                 parent=None, window_title: str = "Crop Preview"):
        """Initialize crop preview dialog.
        
        Args:
            original_image: Original image (BGR)
            cropped_image: Cropped image (BGR)
            parent: Parent Tkinter window (optional)
            window_title: Window title
        """
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        
        self.original_image = original_image
        self.cropped_image = cropped_image
        self.accepted = False
        
        # Create dialog window
        if parent:
            self.dialog = tk.Toplevel(parent)
        else:
            self.dialog = tk.Tk()
        
        self.dialog.title(window_title)
        self.dialog.geometry("1000x600")
        self.dialog.configure(background="#1A1A1A")
        
        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="Crop Preview - Review and Confirm",
                               font=("Consolas", 14, "bold"), 
                               bg="#1A1A1A", fg="#00FF41")
        title_label.pack(pady=(0, 10))
        
        # Image comparison frame
        compare_frame = ttk.Frame(main_frame)
        compare_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Original
        left_frame = self._create_image_panel(compare_frame, "Original Image", 
                                              original_image, side=tk.LEFT)
        
        # Right: Cropped
        right_frame = self._create_image_panel(compare_frame, "Cropped Result", 
                                               cropped_image, side=tk.RIGHT)
        
        # Info labels
        info_frame = tk.Frame(main_frame, bg="#1A1A1A")
        info_frame.pack(fill=tk.X, pady=5)
        
        orig_h, orig_w = original_image.shape[:2]
        crop_h, crop_w = cropped_image.shape[:2]
        
        info_text = f"Original: {orig_w} x {orig_h} px  |  Cropped: {crop_w} x {crop_h} px"
        info_label = tk.Label(info_frame, text=info_text,
                             font=("Consolas", 10), bg="#1A1A1A", fg="#90FF90")
        info_label.pack()
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#1A1A1A")
        button_frame.pack(pady=10)
        
        # Accept button
        accept_btn = tk.Button(button_frame, text="✓ Accept Crop", 
                              command=self._accept,
                              font=("Consolas", 12, "bold"),
                              bg="#00AA00", fg="white",
                              padx=30, pady=10,
                              cursor="hand2")
        accept_btn.pack(side=tk.LEFT, padx=10)
        
        # Reject button
        reject_btn = tk.Button(button_frame, text="✗ Reject", 
                              command=self._reject,
                              font=("Consolas", 12, "bold"),
                              bg="#AA0000", fg="white",
                              padx=30, pady=10,
                              cursor="hand2")
        reject_btn.pack(side=tk.LEFT, padx=10)
        
        # Center the window BEFORE making it modal
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"1000x600+{x}+{y}")
        
        # Make dialog modal and bring to front
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # CRITICAL: Raise window and force focus
        self.dialog.lift()
        self.dialog.focus_force()
        self.dialog.attributes('-topmost', True)
        self.dialog.after(100, lambda: self.dialog.attributes('-topmost', False))
    
    def _create_image_panel(self, parent, title, image, side):
        """Create image panel with title and image display."""
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        
        panel = tk.Frame(parent, bg="#1A1A1A", highlightbackground="#00FF41", 
                        highlightthickness=2)
        panel.pack(side=side, fill=tk.BOTH, expand=True, padx=5)
        
        # Title
        title_label = tk.Label(panel, text=title, 
                              font=("Consolas", 11, "bold"),
                              bg="#1A1A1A", fg="#00FF41")
        title_label.pack(pady=5)
        
        # Convert and resize image for display
        display_img = self._prepare_for_display(image, max_size=(450, 450))
        photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        
        # Image label
        img_label = tk.Label(panel, image=photo, bg="#000000")
        img_label.image = photo  # Keep reference
        img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return panel
    
    def _prepare_for_display(self, cv2_image, max_size=(450, 450)):
        """Convert CV2 image to RGB and resize for display."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio
        h, w = rgb_image.shape[:2]
        max_w, max_h = max_size
        
        ratio = min(max_w / w, max_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        if new_w > 0 and new_h > 0:
            resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = rgb_image
        
        return resized
    
    def _accept(self):
        """Accept the crop and close dialog."""
        self.accepted = True
        self.dialog.quit()
        self.dialog.destroy()
    
    def _reject(self):
        """Reject the crop and close dialog."""
        self.accepted = False 
        self.dialog.quit()
        self.dialog.destroy()
    
    def show(self) -> bool:
        """Show the preview dialog and wait for user decision.
        
        Returns:
            True if accepted, False if rejected
        """
        self.dialog.wait_window()
        return self.accepted


def show_tkinter_crop_preview(original_image: np.ndarray, 
                               cropped_image: np.ndarray,
                               parent=None,
                               window_title: str = "Crop Preview") -> bool:
    """Convenience function to show Tkinter crop preview.
    
    Args:
        original_image: Original image (BGR)
        cropped_image: Cropped image (BGR)
        parent: Parent Tkinter window (optional)
        window_title: Window title
        
    Returns:
        True if user accepted crop, False if rejected
        
    Example:
        >>> roi = select_roi(image)
        >>> if roi:
        >>>     cropped = crop_image(image, roi)
        >>>     if show_tkinter_crop_preview(image, cropped):
        >>>         # User accepted, use cropped image
        >>>         process(cropped)
    """
    preview = TkinterCropPreview(original_image, cropped_image, parent, window_title)
    return preview.show()


if __name__ == "__main__":
    # Demo usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image '{image_path}'")
            sys.exit(1)
        
        print("Select ROI on the image...")
        roi = select_roi(image, "Demo: Select Region of Interest")
        
        if roi:
            x, y, w, h = roi
            print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
            
            # Crop and show result
            cropped = crop_image(image, roi)
            cv2.imshow("Cropped Result", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("ROI selection cancelled")
    else:
        print("Usage: python crop_tool.py <image_path>")
        print("Example: python crop_tool.py DSC00052.JPG")

