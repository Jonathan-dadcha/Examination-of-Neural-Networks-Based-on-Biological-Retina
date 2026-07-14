#!/usr/bin/env python3
"""
Script to help check and update BASEPATH in the notebook
"""
import os
import json
import sys

def check_basepath_in_notebook(notebook_path='SUBMISSION_flashed_reconstruction_demo.ipynb'):
    """Check current BASEPATH in notebook"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if 'source' in cell:
            source = ''.join(cell['source'])
            if 'BASEPATH' in source and 'change this' in source:
                print(f"Found BASEPATH in cell {i}:")
                print(source)
                return i, source, nb
    
    return None, None, nb

def update_basepath_in_notebook(notebook_path, new_basepath):
    """Update BASEPATH in notebook"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if 'source' in cell:
            source_lines = cell['source']
            for j, line in enumerate(source_lines):
                if 'BASEPATH' in line and 'change this' in line:
                    # Update the line
                    new_line = f"BASEPATH='{new_basepath}' # change this\n"
                    source_lines[j] = new_line
                    updated = True
                    print(f"Updated BASEPATH to: {new_basepath}")
    
    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Notebook updated: {notebook_path}")
    else:
        print("✗ Could not find BASEPATH to update")
    
    return updated

def check_data_files(basepath):
    """Check if required data files exist"""
    required_files = [
        '2018_08_07_5_flashed_demo_data.p',
        'pickles/reclassed.p',
        'pickles/bigger_crop_bbox_with_midgets.pickle',
        'models/nojitter.yaml'
    ]
    
    print(f"\nChecking data files in: {basepath}\n")
    all_exist = True
    for file in required_files:
        full_path = os.path.join(basepath, file)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

if __name__ == '__main__':
    print("="*60)
    print("BASEPATH Checker and Updater")
    print("="*60)
    
    # Check current BASEPATH
    cell_idx, current_source, nb = check_basepath_in_notebook()
    
    if cell_idx is not None:
        print(f"\nCurrent BASEPATH found in cell {cell_idx}")
    else:
        print("\nCould not find BASEPATH in notebook")
        sys.exit(1)
    
    # Check if user provided a new path
    if len(sys.argv) > 1:
        new_path = sys.argv[1]
        if not new_path.endswith('/'):
            new_path += '/'
        
        # Check if files exist
        if check_data_files(new_path):
            # Update notebook
            update_basepath_in_notebook('SUBMISSION_flashed_reconstruction_demo.ipynb', new_path)
            print("\n✓ Notebook updated successfully!")
        else:
            print("\n✗ Some required files are missing.")
            print("Please make sure you have:")
            print("  1. Downloaded the demo data ZIP file")
            print("  2. Extracted it to a location")
            print("  3. Provided the correct path to this script")
            print(f"\nExample usage:")
            print(f"  python check_and_fix_basepath.py '/path/to/your/data/flashed/'")
    else:
        print("\nUsage:")
        print(f"  python check_and_fix_basepath.py '/path/to/your/data/flashed/'")
        print("\nThis will:")
        print("  1. Check if the data files exist")
        print("  2. Update BASEPATH in the notebook if files are found")

