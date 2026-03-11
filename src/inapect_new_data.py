import os
import h5py
import numpy as np

# עדכן את הנתיב בהתאם למה שמופיע אצלך בצילום מסך
NEW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', '10.12751_g-node.2j3d2i')

def inspect_h5_structure():
    # נחפש קובץ h5 ראשון בתיקייה
    found_file = None
    for root, dirs, files in os.walk(NEW_DATA_PATH):
        for file in files:
            if file.endswith('.h5') or file.endswith('.hdf5'):
                found_file = os.path.join(root, file)
                break
        if found_file: break
    
    if not found_file:
        print("❌ לא נמצאו קבצי H5 בתיקייה החדשה.")
        return

    print(f"🔍 בודק את הקובץ: {found_file}")
    
    with h5py.File(found_file, 'r') as f:
        print("\n--- Keys in File ---")
        print(list(f.keys()))
        
        # ננסה להבין מה הגודל של כל רכיב
        for key in f.keys():
            try:
                data = f[key]
                print(f"Key: '{key}', Shape: {data.shape}, Type: {data.dtype}")
            except:
                print(f"Key: '{key}' (Group or complex structure)")

if __name__ == "__main__":
    inspect_h5_structure()