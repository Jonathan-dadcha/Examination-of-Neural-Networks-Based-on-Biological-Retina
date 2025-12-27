#!/usr/bin/env python3
"""
Script to test and run all three retinal models:
1. CBEM (Conductance-based Encoding Model)
2. Deep-Retina
3. Wu Nature Communications 2024
"""

import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to use conda Python if available
_conda_python = os.path.join(SCRIPT_DIR, '.conda', 'bin', 'python')
if os.path.exists(_conda_python):
    # If running with system Python but conda exists, suggest using conda
    if 'conda' not in sys.executable:
        print(f"Note: Conda environment detected at {_conda_python}")
        print(f"Current Python: {sys.executable}")
        print("Consider using the conda Python for consistency.\n")

def test_cbem():
    """Test CBEM model"""
    print("\n" + "="*60)
    print("Testing CBEM (Conductance-based Encoding Model)")
    print("="*60)
    
    try:
        import pyCBEM.RGC_CBEM as cbem
        import h5py
        
        # Check if example data exists
        data_path = os.path.join(SCRIPT_DIR, "CBEM-master/Data/example.mat")
        if os.path.exists(data_path):
            print("✓ Example data file found")
            with h5py.File(data_path, "r") as f:
                print(f"✓ Data file readable. Keys: {list(f.keys())[:5]}")
        else:
            print("✗ Example data file not found at", data_path)
            return False
        
        # Test initialization
        cbem.CBEM_basic(0.1)
        print("✓ CBEM model initialized successfully")
        print("\nTo run CBEM:")
        print("  cd CBEM-master")
        print("  jupyter notebook exampleCBEMfitting.ipynb")
        print("  OR run the MATLAB scripts: exampleScript.m")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_deep_retina():
    """Test Deep-Retina model"""
    print("\n" + "="*60)
    print("Testing Deep-Retina")
    print("="*60)
    
    try:
        import deepretina.models as models
        
        print("✓ Deep-Retina modules imported successfully")
        print(f"  Available models: {[x for x in dir(models) if not x.startswith('_')]}")
        
        print("\nTo run Deep-Retina:")
        print("  cd deep-retina-master/scripts")
        print("  python fit_models.py --expt <expt> --stim <stim> --model BN_CNN")
        print("  OR")
        print("  python fit_models.py --expt <expt> --stim <stim> --model LN_softplus --cell 0")
        print("\nNote: You need your own experimental data files")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wu_nature():
    """Test Wu Nature Communications model"""
    print("\n" + "="*60)
    print("Testing Wu Nature Communications 2024")
    print("="*60)
    
    try:
        import numpy as np
        import torch
        import scipy
        
        print("✓ Core dependencies available")
        print(f"  NumPy: {np.__version__}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  SciPy: {scipy.__version__}")
        
        # Check for demo notebooks
        notebook1 = os.path.join(SCRIPT_DIR, "wu-nature-comms-2024-master/SUBMISSION_flashed_reconstruction_demo.ipynb")
        notebook2 = os.path.join(SCRIPT_DIR, "wu-nature-comms-2024-master/SUBMISSION_jitter_reconstruction_demo.ipynb")
        
        if os.path.exists(notebook1):
            print(f"✓ Found: {notebook1}")
        if os.path.exists(notebook2):
            print(f"✓ Found: {notebook2}")
        
        print("\nTo run Wu Nature Communications:")
        print("  cd wu-nature-comms-2024-master")
        print("  jupyter notebook SUBMISSION_flashed_reconstruction_demo.ipynb")
        print("  OR")
        print("  jupyter notebook SUBMISSION_jitter_reconstruction_demo.ipynb")
        print("\nNote: Update the data path in the notebook before running")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RETINAL MODELS SETUP VERIFICATION")
    print("="*60)
    
    results = {
        "CBEM": test_cbem(),
        "Deep-Retina": test_deep_retina(),
        "Wu Nature": test_wu_nature()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, status in results.items():
        status_symbol = "✓" if status else "✗"
        print(f"{status_symbol} {model}: {'Ready' if status else 'Not ready'}")
    
    all_ready = all(results.values())
    if all_ready:
        print("\n✓ All models are set up and ready to use!")
    else:
        print("\n⚠ Some models need additional setup")
    
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())

