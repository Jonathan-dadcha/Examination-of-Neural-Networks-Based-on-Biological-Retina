import os
import h5py
import numpy as np

# Adjust this path if necessary to point to the new data folder
NEW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', '10.12751_g-node.2j3d2i')

def inspect_all_h5_files():
    print(f"🚀 Scanning for ALL H5 files in: {NEW_DATA_PATH}\n")
    
    h5_files_found = []

    # 1. Find all H5 files recursively
    for root, dirs, files in os.walk(NEW_DATA_PATH):
        for file in files:
            if file.endswith('.h5') or file.endswith('.hdf5'):
                full_path = os.path.join(root, file)
                h5_files_found.append(full_path)

    if not h5_files_found:
        print("❌ No H5 files found. Please check the path.")
        return

    # 2. Inspect each file
    for file_path in h5_files_found:
        print(f"{'='*60}")
        print(f"📂 File: {os.path.basename(file_path)}")
        print(f"📍 Path: {file_path}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                print(f"🔑 Keys: {keys}")
                
                # Print details for each key
                for key in keys:
                    try:
                        obj = f[key]
                        if isinstance(obj, h5py.Dataset):
                            print(f"   - Dataset '{key}': Shape={obj.shape}, Type={obj.dtype}")
                        elif isinstance(obj, h5py.Group):
                            print(f"   - Group '{key}': {list(obj.keys())}")
                    except Exception as e:
                        print(f"   - Error reading key '{key}': {e}")
                        
        except Exception as e:
            print(f"❌ Could not read file: {e}")
        print("\n")

if __name__ == "__main__":
    inspect_all_h5_files()