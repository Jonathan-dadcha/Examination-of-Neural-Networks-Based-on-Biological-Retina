# Data Setup for Wu Nature Communications Demo Notebooks

## ⚠️ Important: Demo Data Required

The demo notebooks require demonstration data files that are **not included** in this repository.

## 📥 How to Get the Data

According to the README:

> **Fitted models and demonstration data are provided in a .zip file.**

You need to:

1. **Download the demo data ZIP file** from the repository or publication
2. **Extract it** to a location on your machine
3. **Update the `BASEPATH` variable** in the notebook to point to the extracted data

## 📝 Current BASEPATH in Notebooks

The notebooks currently have:
```python
BASEPATH='/Volumes/Backup/Scratch/Users/wueric/SUBMISSION_DATA_reconstruction/flashed/' # change this
```

## 🔧 How to Update the Path

1. **For flashed reconstruction demo:**
   - Open `SUBMISSION_flashed_reconstruction_demo.ipynb`
   - Find the cell with `BASEPATH=...`
   - Change it to your data location, e.g.:
     ```python
     BASEPATH = '/path/to/your/extracted/data/flashed/'
     ```

2. **For jittered reconstruction demo:**
   - Open `SUBMISSION_jitter_reconstruction_demo.ipynb`
   - Find the cell with `BASEPATH=...`
   - Change it to your data location, e.g.:
     ```python
     BASEPATH = '/path/to/your/extracted/data/jittered/'
     ```

## 📁 Expected Data Structure

After extracting the data, you should have:

```
your_data_directory/
├── flashed/
│   ├── 2018_08_07_5_flashed_demo_data.p
│   ├── pickles/
│   │   ├── reclassed.p
│   │   └── bigger_crop_bbox_with_midgets.pickle
│   └── models/
│       └── nojitter.yaml
└── jittered/
    ├── 2018_08_07_5_jittered_demo_data.p
    └── (other files...)
```

## 🔍 Where to Find the Data

- Check the publication/repository for download links
- Look for a "Data" or "Supplementary Materials" section
- Contact the authors if the data link is not available

## ⚡ Quick Test

Once you've set up the data, test the path:

```python
import os
BASEPATH = '/your/path/here'
test_file = os.path.join(BASEPATH, '2018_08_07_5_flashed_demo_data.p')
if os.path.exists(test_file):
    print('✓ Data file found!')
else:
    print('✗ Data file not found. Check your BASEPATH.')
```

