import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import h5py
import cv2
import glob
from config import BASE_PATH, PROJECT_ROOT
RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_images')
OUTPUT_H5 = os.path.join(BASE_PATH, 'processed_data', 'natural_scenes.h5')
IMG_HEIGHT = 100
IMG_WIDTH = 75

def find_file(filename, search_paths):
    """Search for a file in a list of possible paths, return the first match."""
    print(f"🔍 Searching for '{filename}' in:")
    for path in search_paths:
        full_path = os.path.join(path, filename)
        print(f"   - {path}")
        if os.path.exists(full_path):
            print(f"   ✅ Found at: {full_path}")
            return full_path
    return None

def find_image_path(image_id, search_dir):
    """Search for an image recursively (JPG or PNG)."""
    pattern_jpg = os.path.join(search_dir, '**', f"{image_id}.jpg")
    files = glob.glob(pattern_jpg, recursive=True)
    
    if not files:
        pattern_png = os.path.join(search_dir, '**', f"{image_id}.png")
        files = glob.glob(pattern_png, recursive=True)
        
    return files[0] if files else None

def process_images(list_file_path, dataset_name, h5_file):
    print(f"\n--- Processing dataset: {dataset_name} ---")
    
    with open(list_file_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip() and not line.startswith('[')]
    
    print(f"Loaded {len(image_ids)} image IDs.")
    
    images_data = []
    found_count = 0
    
    for img_id in image_ids:
        img_path = find_image_path(img_id, RAW_IMAGES_DIR)
        
        if img_path:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"⚠️ Error reading file: {img_path}")
                continue

            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            
            images_data.append(img_resized)
            found_count += 1
        else:
            print(f"❌ Image ID {img_id} not found in {RAW_IMAGES_DIR}")

    print(f"✅ Successfully processed {found_count}/{len(image_ids)} images.")
    
    if found_count > 0:
        data_np = np.array(images_data, dtype=np.uint8)
        h5_file.create_dataset(dataset_name, data=data_np)
    
def main():
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ Error: Raw images directory not found at {RAW_IMAGES_DIR}")
        return

    possible_locs = [
        os.path.join(RAW_IMAGES_DIR, 'BSDS'),
        os.path.join(BASE_PATH, 'code_for_stim_reconstruction'),
        RAW_IMAGES_DIR
    ]
    
    train_path = find_file('iids_train.txt', possible_locs)
    test_path = find_file('iids_test.txt', possible_locs)
    
    if not train_path:
        print("\n❌ CRITICAL ERROR: Could not find 'iids_train.txt'.")
        print("Please make sure the file is inside: data/raw_images/BSDS/")
        return

    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)

    print(f"\n💾 Creating H5 dataset at: {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, 'w') as f:
        process_images(train_path, 'train_images', f)
        
        if test_path:
            process_images(test_path, 'test_images', f)
        else:
            print("⚠️ Warning: iids_test.txt not found (skipping test set).")
            
    print(f"\n✨ DONE! Natural Scenes dataset is ready.")

if __name__ == "__main__":
    main()