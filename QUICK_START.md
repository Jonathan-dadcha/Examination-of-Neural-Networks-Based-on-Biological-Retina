# Quick Start Guide - Retinal Models

All models are installed and ready to use!

## ✅ Installation Status

All three models have been successfully installed:
- ✓ CBEM (Conductance-based Encoding Model)
- ✓ Deep-Retina
- ✓ Wu Nature Communications 2024

## 🚀 How to Run Each Model

### 1. CBEM (Conductance-based Encoding Model)

**Python (Jupyter Notebook):**
```bash
cd CBEM-master

# Option 1: Use the convenience script
./run_notebook.sh

# Option 2: Use conda Python directly
../.conda/bin/python -m jupyter notebook exampleCBEMfitting.ipynb

# Option 3: If jupyter is in PATH
jupyter notebook exampleCBEMfitting.ipynb
```

**MATLAB:**
```bash
cd CBEM-master
matlab -r "run('exampleScript.m')"
# OR for spatiotemporal version:
matlab -r "run('exampleScript_spatiotemporal.m')"
```

**Example data:** `CBEM-master/Data/example.mat` and `example_spatiotemporal.mat`

---

### 2. Deep-Retina

```bash
cd deep-retina-master/scripts

# Fit BN_CNN model (use conda Python)
# Example: ../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model BN_CNN
../../.conda/bin/python fit_models.py --expt <expt> --stim <stim> --model BN_CNN

# Fit Linear-Nonlinear model (requires --cell)
# Example: ../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model LN_softplus --cell 0
../../.conda/bin/python fit_models.py --expt <expt> --stim <stim> --model LN_softplus --cell 0
```

**Available values:**
- `--expt`: `15-10-07`, `15-11-21a`, `15-11-21b`, `16-01-07`, `16-01-08`, `16-05-31`
- `--stim`: `whitenoise`, `naturalscene`
- `--model`: `BN_CNN`, `LN_softplus`, `LN_sigmoid`, `LN_relu`, `LN_rbf`
- `--cell`: Cell index (0-31, depends on experiment)

**Important:** Code expects data at `~/experiments/data/<expt>/<stim>.h5`

See `deep-retina-master/HOW_TO_RUN.md` and `deep-retina-master/EXAMPLE_COMMANDS.md` for more details.

---

### 3. Wu Nature Communications 2024

```bash
cd wu-nature-comms-2024-master

# Flashed reconstruction demo (use conda Python)
../.conda/bin/jupyter notebook SUBMISSION_flashed_reconstruction_demo.ipynb

# Jittered eye movements demo
../.conda/bin/jupyter notebook SUBMISSION_jitter_reconstruction_demo.ipynb
```

**Note:** Update the data path at the top of the notebook.

See `wu-nature-comms-2024-master/DATA_SETUP.md` for data setup instructions.

---

## 🧪 Verify Installation

Run this script to verify everything works:

```bash
# Use the conda Python (recommended)
.conda/bin/python run_all_models.py

# OR use system Python (if packages installed there)
python3 run_all_models.py
```

**Note:** Modules are installed in the local conda environment (`.conda/bin/python`)

---

## 📦 Installed Dependencies

- **CBEM:** numpy, jax, scipy, h5py, matplotlib
- **Deep-Retina:** tensorflow, keras, numpy, scipy, matplotlib, deepdish, pyret, tableprint
- **Wu Nature:** pytorch, numpy, scipy, matplotlib, statsmodels, shapely, scikit-image, h5py

---

## ⚠️ Important Notes

1. **Deep-Retina:** Updated to work with TensorFlow 2.x and Keras 3.x

2. **NumPy:** Installed NumPy 1.26.4 (compatible with deepdish)

3. **CBEM:** Includes built-in example data

4. **Wu Nature:** Requires external data (not included)

---

## 🔧 Troubleshooting

If you encounter issues, try:

```bash
# Re-run verification
.conda/bin/python run_all_models.py

# Check Python version
.conda/bin/python --version  # Should be 3.11+

# Reinstall if needed
cd <model-directory>
.conda/bin/python -m pip install -r requirements.txt  # if available
```
