# Comparative Analysis of Retinal Neural Networks

**A computational neuroscience framework exploring how Statistical, Biophysical, and Deep Learning approaches predict retinal ganglion cell responses to Natural Scenes.**

---

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

---

## Overview

This repository hosts a **Unified Comparative Framework** designed to evaluate retinal encoding models. Unlike traditional studies using white noise stimuli, this project focuses on **Natural Scene (NS) stimuli**, which present complex spatiotemporal correlations that challenge conventional modeling approaches.

We implemented and compared four distinct modeling paradigms — from classical linear statistics to biologically-inspired foveated deep learning — to determine which best predicts spike trains of mouse retinal ganglion cells (RGCs) under ecologically valid conditions.

| Path | Approach | Description | Status |
|:----:|:---------|:------------|:------:|
| **A** | Linear Statistical Model | GLM/CBEM with Poisson regression | Complete |
| **B** | Biophysical ODE Model | Retinomorphic circuit simulation | Complete |
| **C** | Deep Learning | Spatiotemporal CNN architecture | Complete |
| **D** | Foveated Drift CNN | Micro-drift + two-stream foveated processing | Complete |

---

## Data Pipeline

The project leverages the **Karamanlis & Gollisch 2021** dataset featuring Mouse Retinal Ganglion Cell recordings under natural image stimulation.

**Pipeline Location:** `src/01_Data_Prep/`

### Core Script: `prepare_ns_dataset_v2.py`

- Reconstructs the random sequence of natural images using the original `ran1` algorithm
- Upsamples images to video frames at **60 Hz**
- Synchronizes spike trains with **millisecond precision**
- Outputs processed H5 files (`training_dataset_ns_full.h5`) ready for model training

---

## Implemented Models

### Path A — Statistical Model (CBEM/GLM)

**Location:** `src/Path_A_CBEM/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Generalized Linear Model (Poisson Regression) |
| **Logic** | Predicts firing rate via linear combination of past frames (20-frame history) |
| **Result** | Failed to capture natural scene dynamics — **R = -0.02** |

> Linear models prove insufficient for complex visual stimuli with rich spatiotemporal correlations.

---

### Path B — Biophysical Model (Retinomorphic ODE)

**Location:** `src/Path_B_Retinomorphic/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Ordinary Differential Equations (RK4 solver) |
| **Logic** | Simulates cellular electrical circuits (RC) with bandpass filtering and adaptation |
| **Result** | Fixed mathematical structure proved too rigid — **R = 0.00** |

> Despite parameter optimization, the deterministic ODE structure cannot capture the dynamic range of natural videos.

---

### Path C — Deep Learning Model (Deep Retina CNN)

**Location:** `src/Path_C_DeepLearning/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Spatiotemporal Convolutional Neural Network (PyTorch) |
| **Logic** | Learns non-linear filters from raw video input (50x50 crop, 40-frame history) |
| **Result** | **State-of-the-art performance — R = 0.29** |

> Successfully predicts spike timing and burst events where classical models fail. Serves as the full-resolution baseline for efficiency comparisons.

---

### Path D — Foveated Drift CNN

**Location:** `src/Path_D_Drift_CNN/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Two-stream Foveated CNN with biological micro-drift (PyTorch) |
| **Concept** | Inspired by Tiezzi et al. (2022) and primate foveal/peripheral asymmetry |
| **Architecture** | **Fovea stream** — high-res center crop (20x20, full resolution); **Peripheral stream** — full-field downsampled via 2x2 avg pooling (25x25). Streams are concatenated and fed to a shared regressor. |
| **Micro-Drift** | Each training sample applies simulated fixational eye movement (Gaussian random walk, std=0.25 px/frame, clipped to ±3 px) across a 40-frame temporal history, spatially jittering the receptive field crop on every frame. |
| **Result** | **R ~ 0.20** (peak 0.22) with **92.1% fewer MACs** than the baseline CNN |

> By splitting the visual field into a high-acuity foveal center and a coarse peripheral surround, the Foveated Drift CNN retains ~69% of the baseline correlation while requiring only **7.9%** of the compute. This demonstrates that biologically-inspired spatial subsampling is a viable strategy for efficient retinal encoding models.

---

## Key Results

Our comparative analysis revealed a decisive advantage for Deep Learning in natural scene decoding, while the Foveated architecture demonstrated that biological design principles can dramatically reduce compute without catastrophic accuracy loss.

| Model | Approach | Correlation (R) | Compute (MACs) | Status |
|:------|:---------|:---------------:|:--------------:|:------:|
| Path A | GLM | -0.02 | — | Failed |
| Path B | ODE | 0.00 | — | Failed |
| **Path C** | **Standard CNN** | **0.291** | **Baseline (1.0x)** | **Success** |
| **Path D** | **Foveated Drift CNN** | **~0.20** | **0.079x (92.1% reduction)** | **Success** |

**Key Takeaways:**

1. Classical statistical and biophysical models (Paths A & B) cannot capture the nonlinear spatiotemporal structure of natural scenes.
2. The standard CNN (Path C) achieves the highest raw correlation, confirming that learned nonlinear features are essential.
3. The Foveated Drift CNN (Path D) trades a modest accuracy reduction (~31%) for a massive compute reduction (~92%), validating biologically-plausible efficient processing.

---

## Quick Start

### 1. Setup Environment

```bash
# Create and activate conda environment
conda create -n retina_env python=3.11
conda activate retina_env

# Install core dependencies
pip install numpy scipy matplotlib torch h5py scikit-learn thop
```

### 2. Prepare Data

```bash
cd src/01_Data_Prep
python prepare_ns_dataset_v2.py
```

This reconstructs the Karamanlis & Gollisch stimulus sequence and produces the aligned `training_dataset_ns_full.h5` file.

### 3. Find Receptive Field Center (Path D prerequisite)

```bash
python src/Path_D_Drift_CNN/find_rf_center.py
```

Computes the spike-triggered average to locate the RGC receptive field center used for cropping.

### 4. Train Models

```bash
# Path C — Standard CNN (baseline)
python src/Path_C_DeepLearning/train_cnn_model.py

# Path D — Baseline Drift CNN (micro-drift, no foveation)
python src/Path_D_Drift_CNN/train_drift_model.py

# Path D — Foveated Drift CNN (micro-drift + foveated two-stream)
python src/Path_D_Drift_CNN/train_foveated_model.py
```

### 5. Run Final Comparison

```bash
python src/Analysis/run_all_models.py
```

### Reference Model Libraries

The `Models/` directory contains cloned reference implementations used during early exploration:

```bash
# CBEM (Conductance-based Encoding Model)
cd Models/CBEM-master
../../Models/.conda/bin/python -m jupyter notebook exampleCBEMfitting.ipynb

# Deep-Retina (McIntosh et al.)
cd Models/deep-retina-master/scripts
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model BN_CNN

# Wu Nature Communications 2024
cd Models/wu-nature-comms-2024-master
../.conda/bin/jupyter notebook SUBMISSION_jitter_reconstruction_demo.ipynb
```

### Verify Installation

```bash
Models/.conda/bin/python --version   # Should be 3.11+
```

---

## Project Structure

```
Retina-Comp-Project/
│
├── data/                              # Raw recordings & processed H5 files
│   └── 10.12751_g-node.2j3d2i/       # Karamanlis & Gollisch dataset
│       └── processed_data/            # training_dataset_ns_full.h5
│
├── checkpoints/                       # Saved model weights & training curves
│
├── Models/                            # Reference implementations & conda env
│   ├── .conda/                        # Local Python 3.11 environment
│   ├── CBEM-master/                   # Conductance-based encoding model
│   ├── deep-retina-master/            # McIntosh et al. CNN
│   ├── foveated_cnn/                  # Early foveated CNN prototypes
│   └── wu-nature-comms-2024-master/   # Wu et al. jitter reconstruction
│
├── src/                               # Source code
│   ├── 01_Data_Prep/                  # Data reconstruction pipeline
│   ├── Path_A_CBEM/                   # GLM model scripts
│   ├── Path_B_Retinomorphic/          # ODE simulation scripts
│   ├── Path_C_DeepLearning/           # Standard CNN model scripts
│   ├── Path_D_Drift_CNN/              # Foveated Drift CNN (final model)
│   │   ├── find_rf_center.py          # STA-based RF localization
│   │   ├── drift_dataset.py           # Micro-drift dataset (baseline)
│   │   ├── foveated_drift_dataset.py  # Two-stream foveated dataset
│   │   ├── train_drift_model.py       # Baseline drift CNN training
│   │   ├── train_foveated_model.py    # Foveated drift CNN training
│   │   └── visualize_drift.py         # Drift trajectory visualization
│   ├── Analysis/                      # Final comparison & plotting
│   └── config.py                      # Global configuration
│
└── Written-materials/                 # Reports & papers
```

---

## References

1. **Karamanlis D & Gollisch T** (2021). *Nonlinear spatial integration underlies the diversity of retinal ganglion cell responses to natural images.* Journal of Neuroscience.

2. **McIntosh L et al.** (2016). *Deep Learning Models of the Retinal Response to Natural Scenes.* NIPS.

3. **Tiezzi M et al.** (2022). *Foveated Neural Computation.* ECML PKDD.

4. **Wu EG et al.** (2024). *Fixational eye movements enable robust edge computation in neural circuits.* Nature Communications.

---

## Authors

**Jonathan Dadcha** · **Adar Shapira**

---

*Built for computational neuroscience research.*
