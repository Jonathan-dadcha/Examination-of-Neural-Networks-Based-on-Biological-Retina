# How to Update BASEPATH in the Notebook

## ⚠️ Current Issue

The notebook is trying to load data from:
```
/Volumes/Backup/Scratch/Users/wueric/SUBMISSION_DATA_reconstruction/flashed/2018_08_07_5_flashed_demo_data.p
```

This path doesn't exist on your machine. You need to update it.

## 🔧 Solution

### Step 1: Find or Download the Data

The demo data files are **NOT included** in this repository. You need to:

1. **Download the demo data ZIP file** from:
   - The publication's supplementary materials
   - The repository's releases/downloads section
   - Contact the authors if the link is not available

2. **Extract the ZIP file** to a location on your machine

### Step 2: Update BASEPATH in the Notebook

1. Open the notebook: `SUBMISSION_flashed_reconstruction_demo.ipynb`

2. Find the cell that contains:
   ```python
   BASEPATH='/Volumes/Backup/Scratch/Users/wueric/SUBMISSION_DATA_reconstruction/flashed/' # change this
   ```

3. **Change it to your data location**, for example:
   ```python
   BASEPATH = '/Users/yourname/Downloads/demo_data/flashed/'
   ```
   Or if you're on Windows:
   ```python
   BASEPATH = 'C:\\Users\\yourname\\Downloads\\demo_data\\flashed\\'
   ```

4. **Make sure the path ends with `/` (or `\\` on Windows)**

### Step 3: Verify the Path

Before running the notebook, test if the file exists:

```python
import os
BASEPATH = '/your/path/here/flashed/'
test_file = os.path.join(BASEPATH, '2018_08_07_5_flashed_demo_data.p')
print(f"Looking for: {test_file}")
print(f"Exists: {os.path.exists(test_file)}")
```

### Step 4: Expected File Structure

After extracting the demo data ZIP, you should have:

```
your_data_directory/
├── flashed/
│   ├── 2018_08_07_5_flashed_demo_data.p          ← Required
│   ├── pickles/
│   │   ├── reclassed.p                           ← Required
│   │   └── bigger_crop_bbox_with_midgets.pickle  ← Required
│   └── models/
│       └── nojitter.yaml                         ← Required
└── jittered/
    └── 2018_08_07_5_jittered_demo_data.p         ← For jittered demo
```

## 🔍 Quick Check Script

Run this in a Python cell to check your setup:

```python
import os

# Update this to your actual path
BASEPATH = '/path/to/your/data/flashed/'

required_files = [
    '2018_08_07_5_flashed_demo_data.p',
    'pickles/reclassed.p',
    'pickles/bigger_crop_bbox_with_midgets.pickle',
    'models/nojitter.yaml'
]

print(f"Checking BASEPATH: {BASEPATH}\n")
all_exist = True
for file in required_files:
    full_path = os.path.join(BASEPATH, file)
    exists = os.path.exists(full_path)
    status = "✓" if exists else "✗"
    print(f"{status} {file}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✓ All required files found! You can proceed.")
else:
    print("\n✗ Some files are missing. Please check your BASEPATH and data files.")
```

## 📝 Alternative: Use Relative Path

If you extract the data to a subdirectory in the repository:

```python
# Get the notebook's directory
import os
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
BASEPATH = os.path.join(notebook_dir, 'demo_data', 'flashed')
```

## ⚠️ Important Notes

- The demo data files are **large** (several GB)
- You **must download them separately** - they're not in the git repository
- The path must be **absolute** or **relative to where you run the notebook**
- Make sure the path uses forward slashes `/` on Mac/Linux or double backslashes `\\` on Windows

