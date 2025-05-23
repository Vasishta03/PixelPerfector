import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread

class DuplicateFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Perfector")
        self.root.geometry("800x600")
        
        # Constants
        self.HASH_THRESHOLD = 5
        self.HIST_CORR_THRESHOLD = 0.9
        self.valid_extensions = (".jpg", ".jpeg", ".png")
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(sticky="nsew")
        
        # Folder selection
        folder_frame = ttk.LabelFrame(self.main_frame, text="Folder Selection", padding="5")
        folder_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Source folder only
        ttk.Label(folder_frame, text="Source Folder:").grid(row=0, column=0, sticky="w", padx=5)
        self.source_path = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.source_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_source).grid(row=0, column=2, padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="5")
        progress_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Log section
        log_frame = ttk.LabelFrame(self.main_frame, text="Log", padding="5")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Control buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(button_frame, text="Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)
        
    def browse_source(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_path.set(folder)
            
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def read_and_hash_image(self, file_path):
        try:
            img = cv2.imread(file_path)
            if img is None:
                self.log_message(f"Warning: Could not read {file_path}")
                return None
                
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            pil_ver = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_hash = imagehash.phash(pil_ver)
            
            return {
                "path": file_path,
                "hash": img_hash,
                "hist": hist,
                "gray": gray_img
            }
        except Exception as e:
            self.log_message(f"Error processing {file_path}: {str(e)}")
            return None
            
    def get_image_quality(self, gray_img):
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        sharp_val = laplacian.var()
        
        low, high = np.percentile(gray_img, [5, 95])
        exposure_val = float(high - low)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)
        contrast_val = float(np.std(enhanced))
        
        return sharp_val, exposure_val, contrast_val
    
    def group_near_duplicates(self, image_data_list):  # Fixed indentation
        num_imgs = len(image_data_list)
        already_checked = [False] * num_imgs
        groups = []
        
        self.log_message(f"Starting duplicate analysis of {num_imgs} images...")
        self.log_message(f"Using thresholds: Hash={self.HASH_THRESHOLD}, Histogram={self.HIST_CORR_THRESHOLD}")
        
        for i in range(num_imgs):
            if already_checked[i]:
                continue
            
            current_group = [i]
            already_checked[i] = True
            base_img_name = os.path.basename(image_data_list[i]['path'])
            
            for j in range(i + 1, num_imgs):
                if already_checked[j]:
                    continue
                
                # Calculate both hash difference and histogram correlation
                hash_gap = image_data_list[i]["hash"] - image_data_list[j]["hash"]
                hist_similarity = cv2.compareHist(
                    image_data_list[i]["hist"],
                    image_data_list[j]["hist"],
                    cv2.HISTCMP_CORREL
                )
                
                compare_img_name = os.path.basename(image_data_list[j]['path'])
                self.log_message(f"Comparing {base_img_name} with {compare_img_name}:")
                self.log_message(f"  Hash difference: {hash_gap}")
                self.log_message(f"  Histogram similarity: {hist_similarity:.3f}")
                
                # Make thresholds more lenient for testing
                is_duplicate = (hash_gap <= 10) or (hist_similarity >= 0.85)
                
                if is_duplicate:
                    current_group.append(j)
                    already_checked[j] = True
                    self.log_message(f"  => DUPLICATE FOUND!")
                else:
                    self.log_message(f"  => Not a duplicate")
            
            if len(current_group) > 1:
                groups.append(current_group)
                self.log_message(f"\nFound group of {len(current_group)} similar images")
        
        self.log_message(f"\nTotal duplicate groups found: {len(groups)}")
        return groups
        
    def process_images(self):
        try:
            source_dir = self.source_path.get()
            # Create to-delete folder inside source directory
            output_dir = os.path.join(source_dir, "to-delete")
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log_message(f"Created 'to-delete' folder at {output_dir}")
                
            # Get all images
            all_images = [
                os.path.join(source_dir, f) for f in os.listdir(source_dir)
                if f.lower().endswith(self.valid_extensions)
            ]
            
            if not all_images:
                self.log_message("No images found in source directory")
                self.status_var.set("No images found")
                return
                
            self.log_message(f"Found {len(all_images)} images")
            
            # Process images
            with ThreadPoolExecutor() as executor:
                image_data = list(executor.map(self.read_and_hash_image, all_images))
            
            # Filter out None values
            image_data = [x for x in image_data if x is not None]
            
            # Find duplicates
            groups = self.group_near_duplicates(image_data)
            self.log_message(f"Found {len(groups)} groups of similar images")
            
            # Process each group
            moved_count = 0
            for i, group in enumerate(groups):
                if len(group) <= 1:
                    continue
                    
                # Calculate quality metrics
                quality_metrics = []
                for idx in group:
                    metrics = self.get_image_quality(image_data[idx]["gray"])
                    quality_metrics.append(metrics)
                    
                # Find best image
                max_vals = list(map(max, zip(*quality_metrics)))
                scores = []
                for sharp, exp, contrast in quality_metrics:
                    norm_score = sum(v/m if m else 0 for v, m in zip((sharp, exp, contrast), max_vals))
                    scores.append(norm_score)
                    
                best_idx = group[int(np.argmax(scores))]
                best_path = image_data[best_idx]["path"]
                self.log_message(f"Best quality image in group: {os.path.basename(best_path)}")
                
                # Move duplicates
                for idx in group:
                    if idx == best_idx:
                        continue
                        
                    src = image_data[idx]["path"]
                    name = os.path.basename(src)
                    target = os.path.join(output_dir, name)
                    
                    if os.path.exists(target):
                        name_base, ext = os.path.splitext(name)
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        target = os.path.join(output_dir, f"{name_base}_{timestamp}{ext}")
                        
                    shutil.move(src, target)
                    moved_count += 1
                    self.log_message(f"Moved duplicate: {name} -> {os.path.basename(target)}")
                    
                self.progress['value'] = ((i + 1) / len(groups)) * 100
                
            self.status_var.set(f"Complete - Moved {moved_count} duplicates")
            self.progress['value'] = 100
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", str(e))
            
    def start_processing(self):
        if not self.source_path.get():
            messagebox.showerror("Error", "Please select source folder")
            return
            
        self.progress['value'] = 0
        self.status_var.set("Processing...")
        self.log_text.delete(1.0, tk.END)
        
        Thread(target=self.process_images, daemon=True).start()

def main():
    root = tk.Tk()
    app = DuplicateFinderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()