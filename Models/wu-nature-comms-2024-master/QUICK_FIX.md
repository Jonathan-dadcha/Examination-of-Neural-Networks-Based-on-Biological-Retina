# Quick Fix for BASEPATH Error

## ⚠️ The Problem

You're getting this error because the data files don't exist at the default path:
```
/Volumes/Backup/Scratch/Users/wueric/SUBMISSION_DATA_reconstruction/flashed/2018_08_07_5_flashed_demo_data.p
```

## ✅ Quick Solution

**In your Jupyter notebook, before the cell that loads the data, add this cell:**

```python
import os

# Option 1: If you downloaded and extracted the data, set the path here:
BASEPATH = '/path/to/your/extracted/data/flashed/'

# Option 2: Check if file exists first
test_file = os.path.join(BASEPATH, '2018_08_07_5_flashed_demo_data.p')
if not os.path.exists(test_file):
    print(f"✗ File not found: {test_file}")
    print("Please:")
    print("  1. Download the demo data ZIP file")
    print("  2. Extract it")
    print("  3. Update BASEPATH above to point to the extracted location")
else:
    print(f"✓ Data found at: {BASEPATH}")
```

## 📥 Where to Get the Data

The demo data is **NOT in this repository**. You need to:

1. **Find the download link** in:
   - The publication's supplementary materials
   - The repository's releases/downloads
   - Contact the authors

2. **Download the ZIP file** (it's large, several GB)

3. **Extract it** to a location like:
   - `~/Downloads/demo_data/`
   - `~/Desktop/demo_data/`
   - Any other convenient location

4. **Update BASEPATH** to point to the `flashed/` subdirectory

## 🔍 Expected File Structure

After extraction, you should have:
```
demo_data/
└── flashed/
    ├── 2018_08_07_5_flashed_demo_data.p
    ├── pickles/
    │   ├── reclassed.p
    │   └── bigger_crop_bbox_with_midgets.pickle
    └── models/
        └── nojitter.yaml
```

## 💡 Temporary Workaround

If you just want to test the code structure without the actual data, you can comment out the data loading cell and work with dummy data, but the full reconstruction won't work without the real data files.

