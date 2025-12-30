# 🧠 Comparative Analysis of Retinal Neural Networks

**A computational neuroscience framework exploring how Statistical, Biophysical, and Deep Learning approaches predict retinal ganglion cell responses to Natural Scenes.**

---

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

---

## Overview

This repository hosts a **Unified Comparative Framework** designed to evaluate retinal encoding models. Unlike traditional studies using white noise stimuli, this project focuses on **Natural Scene (NS) stimuli**, which present complex spatiotemporal correlations that challenge conventional modeling approaches.

We implemented and compared four distinct modeling paradigms:

| Path | Approach | Description |
|:----:|:---------|:------------|
| **A** | Linear Statistical Model | GLM/CBEM with Poisson regression |
| **B** | Biophysical ODE Model | Retinomorphic circuit simulation |
| **C** | Deep Learning | Spatiotemporal CNN architecture |
| **D** | Foveated CNN | Location-dependent processing *(In Progress)* |

---

## Data Pipeline

The project leverages the **Karamanlis & Gollisch 2021** dataset featuring Mouse Retinal Ganglion Cell recordings.

**Pipeline Location:** `src/01_Data_Prep/`

### Core Script: `prepare_ns_dataset_v2.py`

- Reconstructs the random sequence of natural images using the original `ran1` algorithm
- Upsamples images to video frames at **60 Hz**
- Synchronizes spike trains with **millisecond precision**
- Outputs processed H5 files (`training_dataset_ns_full.h5`) ready for model training

---

## Implemented Models

### Path A — Statistical Model (CBEM/GLM)

📁 **Location:** `src/Path_A_CBEM/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Generalized Linear Model (Poisson Regression) |
| **Logic** | Predicts firing rate via linear combination of past frames (20-frame history) |
| **Result** | Failed to capture natural scene dynamics — *R ≈ 0* |

> Linear models prove insufficient for complex visual stimuli with rich spatiotemporal correlations.

---

### Path B — Biophysical Model (Retinomorphic ODE)

📁 **Location:** `src/Path_B_Retinomorphic/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Ordinary Differential Equations (RK4 solver) |
| **Logic** | Simulates cellular electrical circuits (RC) with bandpass filtering and adaptation |
| **Result** | Fixed mathematical structure proved too rigid — *R ≈ 0* |

> Despite parameter optimization, the deterministic ODE structure cannot capture the dynamic range of natural videos.

---

### Path C — Deep Learning Model (Deep Retina CNN) 🏆

📁 **Location:** `src/Path_C_DeepLearning/`

| Attribute | Details |
|:----------|:--------|
| **Type** | Spatiotemporal Convolutional Neural Network (PyTorch) |
| **Logic** | Learns non-linear filters from raw video input (50×50 crop, 40-frame history) |
| **Result** | **State-of-the-art performance — R ≈ 0.29** |

> Successfully predicts spike timing and burst events where classical models fail.

---

### Path D — Foveated CNN *(In Progress)*

📁 **Location:** `Models/foveated_cnn/`

| Attribute | Details |
|:----------|:--------|
| **Concept** | Based on Tiezzi et al. (2022) |
| **Goal** | Improve computational efficiency by mimicking biological fovea |
| **Architecture** | Custom `FoveatedConv2d` layers with Foveal + Peripheral streams and Gaussian blending |

---

## Key Results

Our comparative analysis (`src/Analysis/run_all_models.py`) revealed a decisive advantage for Deep Learning in natural scene decoding:

| Model | Approach | Correlation (R) | Status |
|:------|:---------|:---------------:|:------:|
| Path A | GLM | -0.02 | ❌ Failed |
| Path B | ODE | 0.00 | ❌ Failed |
| **Path C** | **CNN** | **0.291** | ✅ **Success** |

---

## Quick Start

### 1. Setup Environment

```bash
conda create -n retina_env python=3.11
conda activate retina_env
pip install numpy scipy matplotlib torch h5py scikit-learn
```

### 2. Prepare Data

```bash
cd src/01_Data_Prep
python prepare_ns_dataset_v2.py
```

### 3. Train Models

```bash
# Train the CNN (Path C)
python src/Path_C_DeepLearning/train_cnn_model.py

# Train the Foveated Model (Path D)
python Models/foveated_cnn/train_foveated.py
```

### 4. Run Final Comparison

```bash
# Generates the comparison plot (Figure 1 of the report)
python src/Analysis/run_all_models.py
```

---

## Project Structure

```
Retina-Comp-Project/
│
├── data/                          # Raw recordings & Processed H5 files
│
├── Models/                        # External Repos & New Architectures
│   └── foveated_cnn/              # Path D: Foveated Neural Network
│
├── src/                           # Source Code
│   ├── 01_Data_Prep/              # Data reconstruction pipeline
│   ├── Path_A_CBEM/               # GLM Model scripts
│   ├── Path_B_Retinomorphic/      # ODE Simulation scripts
│   ├── Path_C_DeepLearning/       # CNN Model scripts
│   ├── Analysis/                  # Final comparison & plotting
│   └── config.py                  # Global configuration
│
└── Written-materials/             # Reports & Papers
```

---

## References

1. **Karamanlis D & Gollisch T** (2021). *Nonlinear spatial integration underlies the diversity of retinal ganglion cell responses to natural images.* Journal of Neuroscience.

2. **McIntosh L et al.** (2016). *Deep Learning Models of the Retinal Response to Natural Scenes.* NIPS.

3. **Tiezzi M et al.** (2022). *Foveated Neural Computation.* ECML PKDD.

---

## Authors

**Jonathan Dadcha** · **Adar Shapira**

---

*Built with ❤️ for computational neuroscience*
