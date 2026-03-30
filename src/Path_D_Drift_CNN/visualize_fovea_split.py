"""
Visualize the foveated two-stream input decomposition.

Produces a 1x3 panel figure showing:
  1. Full 50x50 peripheral scene with a red FOA bounding box
  2. The 20x20 foveal crop extracted at the FOA
  3. The 50x50 peripheral input fed to the dilated-conv stream

Self-contained: loads the H5 data directly and replicates the
saliency-guided FOA logic from UnifiedFoveatedDataset so that
only numpy, h5py, scipy, and matplotlib are required (no torch).
"""

import os
import h5py
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SPIKES_PATH = (
    "/Users/jonathandadcha/Desktop/Retina-Comp-Project/"
    "data/10.12751_g-node.2j3d2i/processed_data/training_dataset_ns_full.h5"
)

HISTORY_SIZE = 40
CROP_SIZE = 50
FOVEA_SIZE = 20
HALF_FOVEA = FOVEA_SIZE // 2
HALF_CROP = CROP_SIZE // 2
CENTER_X, CENTER_Y = 27, 47

ALPHA = 0.4
JITTER_STD = 0.1
MAX_STEP = 3.0
SALIENCY_WINDOW = 5

FOA_MIN = float(HALF_FOVEA)
FOA_MAX = float(CROP_SIZE - HALF_FOVEA)


def _compute_saliency(frame):
    f64 = frame.astype(np.float64)
    local_mean = uniform_filter(f64, size=SALIENCY_WINDOW)
    local_sq_mean = uniform_filter(f64 ** 2, size=SALIENCY_WINDOW)
    return np.maximum(local_sq_mean - local_mean ** 2, 0.0)


def _saliency_gradient(saliency, x, y):
    h, w = saliency.shape
    ix = int(np.clip(round(x), 0, w - 1))
    iy = int(np.clip(round(y), 0, h - 1))
    x_lo, x_hi = max(ix - 1, 0), min(ix + 1, w - 1)
    y_lo, y_hi = max(iy - 1, 0), min(iy + 1, h - 1)
    dx = (saliency[iy, x_hi] - saliency[iy, x_lo]) / max(x_hi - x_lo, 1)
    dy = (saliency[y_hi, ix] - saliency[y_lo, ix]) / max(y_hi - y_lo, 1)
    return dx, dy


def to_01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def extract_sample(X, idx=0):
    """Replicate UnifiedFoveatedDataset.__getitem__ for a single sample."""
    raw_clip = X[idx: idx + HISTORY_SIZE]

    H, W = raw_clip.shape[1], raw_clip.shape[2]
    cx = max(HALF_CROP, min(W - HALF_CROP, CENTER_X))
    cy = max(HALF_CROP, min(H - HALF_CROP, CENTER_Y))

    crops = raw_clip[:,
                     cy - HALF_CROP: cy + HALF_CROP,
                     cx - HALF_CROP: cx + HALF_CROP]

    foa_x = float(CROP_SIZE) / 2.0
    foa_y = float(CROP_SIZE) / 2.0

    np.random.seed(42)

    fovea_frames = []
    for t in range(HISTORY_SIZE):
        frame = crops[t]
        saliency = _compute_saliency(frame)
        grad_x, grad_y = _saliency_gradient(saliency, foa_x, foa_y)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
        grad_x /= grad_mag
        grad_y /= grad_mag

        step_x = ALPHA * grad_x + np.random.randn() * JITTER_STD
        step_y = ALPHA * grad_y + np.random.randn() * JITTER_STD

        step_mag = np.sqrt(step_x ** 2 + step_y ** 2)
        if step_mag > MAX_STEP:
            scale = MAX_STEP / step_mag
            step_x *= scale
            step_y *= scale

        foa_x = np.clip(foa_x + step_x, FOA_MIN, FOA_MAX)
        foa_y = np.clip(foa_y + step_y, FOA_MIN, FOA_MAX)

        fx = int(np.clip(round(foa_x), HALF_FOVEA, CROP_SIZE - HALF_FOVEA))
        fy = int(np.clip(round(foa_y), HALF_FOVEA, CROP_SIZE - HALF_FOVEA))

        fovea_frame = frame[fy - HALF_FOVEA: fy + HALF_FOVEA,
                            fx - HALF_FOVEA: fx + HALF_FOVEA]
        fovea_frames.append(fovea_frame)

    last_fovea = fovea_frames[-1]
    last_peripheral = crops[-1]

    return last_fovea, last_peripheral, foa_x, foa_y


def main(output_dir=None):
    if not os.path.exists(SPIKES_PATH):
        raise FileNotFoundError(f"Data file not found: {SPIKES_PATH}")

    print("Loading data...")
    with h5py.File(SPIKES_PATH, "r") as f:
        X = f["X"][:]

    mean, std = np.mean(X), np.std(X)
    X = (X - mean) / (std + 1e-6)
    print(f"Loaded {X.shape[0]} frames, z-score normalized.")

    sample_idx = min(5000, len(X) - HISTORY_SIZE - 1)
    fovea_frame, periph_frame, foa_px_x, foa_px_y = extract_sample(X, idx=sample_idx)
    print(f"Using sample idx={sample_idx}")

    periph_disp = to_01(periph_frame)
    fovea_disp = to_01(fovea_frame)

    box_x = foa_px_x - HALF_FOVEA
    box_y = foa_px_y - HALF_FOVEA

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(periph_disp, cmap="gray", vmin=0, vmax=1)
    rect = patches.Rectangle(
        (box_x, box_y), FOVEA_SIZE, FOVEA_SIZE,
        linewidth=1.5, edgecolor="red", facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Full Scene with FOA", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(fovea_disp, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Foveal Crop (20\u00d720)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(periph_disp, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Peripheral Input (50\u00d750)", fontsize=11)
    axes[2].axis("off")

    fig.tight_layout()

    out = output_dir if output_dir else os.path.dirname(__file__)
    out_path = os.path.join(out, "foveated_split_visualization.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
