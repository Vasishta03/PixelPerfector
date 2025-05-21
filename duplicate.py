import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Thresholds 
HASH_THRESHOLD = 5          # phash is generally 64 bits so a differerence of 5 is good
HIST_CORR_THRESHOLD = 0.9   # allows slight variations in lighting or exposure

# Folder setup 
base_folder = r"C:\Users\vasis\Downloads\Duplicate"
image_folder = os.path.join(base_folder, "Images")
trash_folder = os.path.join(base_folder, "to-delete")
log_folder = os.path.join(base_folder, "log")
log_path = os.path.join(log_folder, "cleanup_log.txt")


def make_sure_folder_exists(folder_path):
    """make the folder if it doesn't exists"""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def read_and_hash_image(file_path):
    """
    Loads image, convert grayscale, and computes hash and histogram
    """
    try:
        img = cv2.imread(file_path)  # OpenCv Reads images in BGR format
        if img is None:
            print(f"Could not read {file_path}")
            return None

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Imagehash wants a PIL image, and in RGB, not BGR
        pil_ver = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_hash = imagehash.phash(pil_ver)

        return {
            "path": file_path,
            "hash": img_hash,
            "hist": hist,
            "gray": gray_img
        }

    except Exception as err:
        print(f"Something went wrong while processing {file_path}: {err}")
        return None


def get_image_quality(gray_img):
    """ 
    Sharpness = how much detail/edges via Laplacian 
    Higher values = more edges = sharper image**
    Lower values = blurry image , out of focus
    """
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    sharp_val = laplacian.var()
    
    
    """
    Exposure = range between shadows/highlights 95th and 5th percentile
    wider range = well exposed = covers both dark and bright areas well
    narrow range = poorly exposed = either too dark or too bright
    """
    low, high = np.percentile(gray_img, [5, 95])
    exposure_val = float(high - low)

    # Contrast = local variations after enhancing with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    contrast_val = float(np.std(enhanced))

    return sharp_val, exposure_val, contrast_val


def group_near_duplicates(image_data_list):
    """
    Groups images that look similar based on hash distance or histogram similarity.
    """
    num_imgs = len(image_data_list)
    already_checked = [False] * num_imgs
    groups = []

    for i in range(num_imgs):
        if already_checked[i]:
            continue
        current_group = [i]
        already_checked[i] = True
        for j in range(i + 1, num_imgs):
            if already_checked[j]:
                continue

            hash_gap = image_data_list[i]["hash"] - image_data_list[j]["hash"]
            hist_similarity = cv2.compareHist(image_data_list[i]["hist"], image_data_list[j]["hist"], cv2.HISTCMP_CORREL)

            if hash_gap <= HASH_THRESHOLD or hist_similarity >= HIST_CORR_THRESHOLD:
                current_group.append(j)
                already_checked[j] = True

        groups.append(current_group)

    return groups


def main():
    # Set folders if they don't exist
    make_sure_folder_exists(trash_folder)
    make_sure_folder_exists(log_folder)

    # Grab all image files 
    valid_exts = (".jpg", ".jpeg", ".png")
    all_images = [
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(valid_exts)
    ]

    print(f"Scanning {len(all_images)} image(s) in {image_folder}")
    if not all_images:
        print("No images found. Check your folder path?")
        return

    # Analyze the images in parallel (faster)
    with ThreadPoolExecutor() as pool:
        raw_features = list(pool.map(read_and_hash_image, all_images))

    # Filter failed image loads
    image_data = [feat for feat in raw_features if feat]

    similar_sets = group_near_duplicates(image_data) #groups visually similar images into clusters
    print(f"Found {len(similar_sets)} potential duplicate group(s).")

    with open(log_path, "a") as log:
        for group in similar_sets:
            if len(group) <= 1:
                continue  

            quality_metrics = []
            for idx in group:
                gray_frame = image_data[idx]["gray"]
                sharp, exp, contrast = get_image_quality(gray_frame)
                quality_metrics.append((sharp, exp, contrast))

            # Find the best image (exposure + contrast + sharpness)
            max_vals = list(map(max, zip(*quality_metrics)))
            scores = []
            for sharp, exp, contrast in quality_metrics:
                norm_score = (
                    (sharp / max_vals[0] if max_vals[0] else 0) +
                    (exp / max_vals[1] if max_vals[1] else 0) +
                    (contrast / max_vals[2] if max_vals[2] else 0)
                )
                scores.append(norm_score)

            best_in_group_idx = group[int(np.argmax(scores))]
            best_image_path = image_data[best_in_group_idx]["path"]
            print(f"Best in group: {os.path.basename(best_image_path)}")

            # Move remaining to to_delete folder
            for idx in group:
                if idx == best_in_group_idx:
                    continue  # keep the best image (highets score)

                src = image_data[idx]["path"]
                name = os.path.basename(src)
                target = os.path.join(trash_folder, name)

                # name clash
                if os.path.exists(target):
                    name_base, ext = os.path.splitext(name)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    target = os.path.join(trash_folder, f"{name_base}_{timestamp}{ext}")

                shutil.move(src, target)
                log.write(f"{datetime.now()}: Moved '{src}' -> '{target}'\n")
                print(f"Moved duplicate: {name} -> {target}")


if __name__ == "__main__":
    main()
