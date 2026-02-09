import cv2
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import albumentations as A

# ==========================================
# 1. ส่วนตั้งค่า Pipeline (ลดความแรง 50% + No Camera Gain)
# ==========================================
def get_augmentation_pipeline():
    return A.Compose([
        # หมุน/กลับด้าน
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=0.5),
        
        # Shift Scale Rotate (เบาลง)
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=20, p=0.5),
        
        # สีและแสง (เบาลง)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1),
            A.RandomGamma(gamma_limit=(90, 110), p=1),
        ], p=0.5),

        # Noise & Blur (เบาลง, ตัด ISONoise ออก)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 25.0), p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.5),

        # Cutout & Distortion (เบาลง)
        A.OneOf([
            A.CoarseDropout(max_holes=4, max_height=16, max_width=16, fill_value=0, p=1),
            A.GridDistortion(distort_limit=0.15, p=1), 
        ], p=0.3),
        
        # Grayscale
        A.ToGray(p=0.1) 
    ])

# ==========================================
# 2. คลาสสำหรับหน้าต่างโปรแกรม (GUI Class)
# ==========================================
class AugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Augmentation Tool (Jellyfish Edition)")
        self.root.geometry("600x500")
        
        # ตัวแปรเก็บค่า
        self.input_files = []
        self.output_folder = ""
        
        # สร้าง Style
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('TLabel', font=('Helvetica', 10))

        # --- สร้างระบบ Tab ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # สร้าง Tab 1: หน้าหลัก (Home)
        self.tab_home = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_home, text='  หน้าหลัก (Home)  ')

        # สร้าง Tab 2: สถานะ (Logs)
        self.tab_logs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_logs, text='  สถานะการทำงาน (Logs)  ')

        # --- Setup UI ใน Tab Home ---
        self.setup_home_tab()
        
        # --- Setup UI ใน Tab Logs ---
        self.setup_log_tab()

    def setup_home_tab(self):
        frame = ttk.Frame(self.tab_home)
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        # ส่วนเลือกไฟล์ Input
        lbl_input = ttk.Label(frame, text="1. เลือกไฟล์รูปภาพต้นฉบับ (.bmp, .tiff, .jpg, etc.)", font=('Helvetica', 11, 'bold'))
        lbl_input.pack(anchor='w', pady=(0, 5))
        
        btn_input = ttk.Button(frame, text="เลือกไฟล์รูปภาพ (Select Images)", command=self.select_images)
        btn_input.pack(fill='x', pady=(0, 5))
        
        self.lbl_input_status = ttk.Label(frame, text="ยังไม่ได้เลือกไฟล์", foreground="gray")
        self.lbl_input_status.pack(anchor='w', pady=(0, 15))

        # ส่วนเลือกโฟลเดอร์ Output
        lbl_output = ttk.Label(frame, text="2. เลือกโฟลเดอร์เก็บผลลัพธ์", font=('Helvetica', 11, 'bold'))
        lbl_output.pack(anchor='w', pady=(0, 5))
        
        btn_output = ttk.Button(frame, text="เลือกโฟลเดอร์ปลายทาง (Select Output Folder)", command=self.select_folder)
        btn_output.pack(fill='x', pady=(0, 5))
        
        self.lbl_output_status = ttk.Label(frame, text="ยังไม่ได้เลือกโฟลเดอร์", foreground="gray")
        self.lbl_output_status.pack(anchor='w', pady=(0, 15))

        # ส่วนตั้งค่าจำนวนภาพ
        lbl_count = ttk.Label(frame, text="3. จำนวนภาพสังเคราะห์ต่อ 1 ภาพต้นฉบับ", font=('Helvetica', 11, 'bold'))
        lbl_count.pack(anchor='w', pady=(0, 5))
        
        self.entry_count = ttk.Entry(frame)
        self.entry_count.insert(0, "10") # ค่าเริ่มต้น 10
        self.entry_count.pack(fill='x', pady=(0, 15))

        # ปุ่ม Start
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)
        self.btn_start = ttk.Button(frame, text="เริ่มทำงาน (START PROCESS)", command=self.start_thread)
        self.btn_start.pack(fill='x', ipady=10)

    def setup_log_tab(self):
        # สร้างกล่องข้อความไว้แสดง Log
        self.log_area = scrolledtext.ScrolledText(self.tab_logs, state='disabled', font=('Consolas', 9))
        self.log_area.pack(expand=True, fill='both', padx=10, pady=10)

    # --- Functions การทำงาน ---
    def log(self, message):
        # ฟังก์ชันสำหรับเขียนข้อความลงใน Tab Logs
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END) # เลื่อนบรรทัดล่างสุดเสมอ
        self.log_area.config(state='disabled')

    def select_images(self):
        files = filedialog.askopenfilenames(
            title="เลือกรูปภาพ",
            filetypes=[("Image files", "*.bmp *.tiff *.tif *.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if files:
            self.input_files = files
            self.lbl_input_status.config(text=f"เลือกแล้ว {len(files)} ไฟล์", foreground="green")
            self.log(f"-> Selected {len(files)} input images.")

    def select_folder(self):
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ปลายทาง")
        if folder:
            self.output_folder = folder
            self.lbl_output_status.config(text=f".../{os.path.basename(folder)}", foreground="green")
            self.log(f"-> Selected output folder: {folder}")

    def start_thread(self):
        # ใช้ Thread แยก เพื่อไม่ให้หน้าต่างค้างเวลาประมวลผล
        if not self.input_files:
            messagebox.showwarning("Warning", "กรุณาเลือกไฟล์รูปภาพก่อน")
            return
        if not self.output_folder:
            messagebox.showwarning("Warning", "กรุณาเลือกโฟลเดอร์ปลายทางก่อน")
            return
            
        # ล็อกปุ่ม
        self.btn_start.config(state='disabled')
        self.notebook.select(self.tab_logs) # สลับไปหน้า Log อัตโนมัติ
        
        # เริ่ม Thread
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            aug_count = int(self.entry_count.get())
        except ValueError:
            aug_count = 10
            self.log("Invalid number, defaulting to 10.")

        self.log("--- Starting Augmentation Process ---")
        transform = get_augmentation_pipeline()
        
        total_files = len(self.input_files)
        success_count = 0

        for idx, file_path in enumerate(self.input_files):
            try:
                # อ่านไฟล์
                image = cv2.imread(file_path)
                filename = os.path.basename(file_path)
                
                if image is None:
                    self.log(f"Error: Could not read {filename}")
                    continue

                name, ext = os.path.splitext(filename)
                
                self.log(f"Processing ({idx+1}/{total_files}): {filename}")

                # Loop สร้างภาพ
                for i in range(aug_count):
                    augmented = transform(image=image)['image']
                    new_filename = f"{name}_aug_{i}{ext}"
                    save_path = os.path.join(self.output_folder, new_filename)
                    cv2.imwrite(save_path, augmented)
                
                success_count += 1
                
            except Exception as e:
                self.log(f"Error processing {file_path}: {e}")

        self.log("--- Completed ---")
        self.log(f"Successfully processed {success_count} files.")
        self.log(f"Total images generated: {success_count * aug_count}")
        
        # แจ้งเตือนเมื่อเสร็จ (ต้องใช้ root.after เพื่อความปลอดภัยของ Thread)
        self.root.after(0, lambda: messagebox.showinfo("Success", "การทำงานเสร็จสิ้นเรียบร้อย!"))
        self.root.after(0, lambda: self.btn_start.config(state='normal'))

# ==========================================
# 3. Main Entry Point
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    # ตั้งค่า icon (ถ้ามีไฟล์ .ico ให้ใส่บรรทัดล่างนี้)
    #root.iconbitmap('icon.ico') 
    app = AugmentationApp(root)
    root.mainloop()