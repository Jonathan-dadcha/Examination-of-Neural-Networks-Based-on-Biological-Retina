"""
Optimal Stimulus / Receptive Field Visualization for ALL Four Models.

Produces a single 2x3 matplotlib figure:
  Row 0: Real Input | (A) GLM RF | (B) ODE RF
  Row 1: (C) DriftCNN Optimal | (D) FoveatedCNN Optimal | [empty]

Output: results/all_models_optimal_stimuli.png
"""

import sys
import os
import json
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_drift_model import DriftCNN
from train_unified_foveated_model import (
    UnifiedFoveatedCNN,
    UnifiedFoveatedDataset,
    IMAGES_PATH,
    SPIKES_PATH,
)

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CHECKPOINTS = os.path.join(PROJECT_ROOT, "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cpu")

GLM_HISTORY = 20
GLM_CROP_SIZE = 25
CNN_HISTORY = 40
NUM_STEPS = 500
LR = 0.05
BLUR_EVERY = 10
BLUR_SIGMA = 0.5
BLUR_KSIZE = 3
L2_DECAY = 1e-3


# ---------------------------------------------------------------------------
# Gaussian blur helper
# ---------------------------------------------------------------------------

def _make_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g


def blur_tensor(t: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    g1d = _make_gaussian_kernel(ksize, sigma).to(t.device)
    C = t.shape[1]
    pad = ksize // 2
    k_h = g1d.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
    k_w = g1d.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
    out = F.conv2d(t, k_h, padding=(pad, 0), groups=C)
    out = F.conv2d(out, k_w, padding=(0, pad), groups=C)
    return out


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Real input — high-activity frame from dataset
# ---------------------------------------------------------------------------

def load_real_input():
    """Return the last peripheral frame from the highest-firing-rate window."""
    print("  [Input] Scanning Y for peak firing rate ...")
    with h5py.File(SPIKES_PATH, "r") as f:
        Y = f["Y"][:]

    peak_frame = int(np.argmax(Y))
    dataset_idx = max(peak_frame - CNN_HISTORY + 1, 0)
    print(f"      Peak Y = {Y[peak_frame]:.4f} at frame {peak_frame}")

    dataset = UnifiedFoveatedDataset(IMAGES_PATH, SPIKES_PATH)
    _, peripheral, _, target = dataset[dataset_idx]
    return peripheral[-1, 0].numpy(), target.item()


# ---------------------------------------------------------------------------
# (A) GLM — spatial weights via PCA inverse transform
# ---------------------------------------------------------------------------

def generate_glm_rf():
    import joblib

    path = os.path.join(CHECKPOINTS, "glm_pipeline.joblib")
    print(f"  [A] Loading GLM pipeline from {path}")
    pipeline = joblib.load(path)

    pca = pipeline.named_steps["pca"]
    glm = pipeline.named_steps["glm"]

    weights_pca = glm.coef_.reshape(1, -1)
    weights_pixel = pca.inverse_transform(weights_pca).reshape(
        GLM_HISTORY, GLM_CROP_SIZE, GLM_CROP_SIZE
    )

    spatial = weights_pixel[-1]
    print(f"      coef shape: {glm.coef_.shape}  ->  spatial filter: {spatial.shape}")
    return spatial


# ---------------------------------------------------------------------------
# (B) ODE — center-surround receptive field mask
# ---------------------------------------------------------------------------

def generate_ode_rf():
    path = os.path.join(CHECKPOINTS, "ode_params.json")
    print(f"  [B] Loading ODE params from {path}")
    with open(path) as f:
        params = json.load(f)

    polarity = params["polarity"]
    print(f"      Polarity: {polarity}-cell")

    size = 50
    c_h, s_h = 1, 4
    cx, cy = size // 2, size // 2

    rf = np.zeros((size, size), dtype=np.float64)

    sy1, sy2 = cy - s_h, cy + s_h + 1
    sx1, sx2 = cx - s_h, cx + s_h + 1
    surround_sign = -1.0 if polarity == "ON" else 1.0
    rf[sy1:sy2, sx1:sx2] = surround_sign * 0.5

    cy1, cy2 = cy - c_h, cy + c_h + 1
    cx1, cx2 = cx - c_h, cx + c_h + 1
    center_sign = 1.0 if polarity == "ON" else -1.0
    rf[cy1:cy2, cx1:cx2] = center_sign * 1.0

    return rf, polarity


# ---------------------------------------------------------------------------
# (C) DriftCNN — activation maximization
# ---------------------------------------------------------------------------

def generate_drift_cnn_stimulus():
    path = os.path.join(CHECKPOINTS, "drift_model_final.pth")
    print(f"  [C] Loading DriftCNN from {path}")

    model = _freeze(DriftCNN(history_size=CNN_HISTORY).to(DEVICE))
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=False)
    )
    model.eval()

    noise = torch.randn(1, CNN_HISTORY, 50, 50,
                         device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([noise], lr=LR)

    print(f"      Running activation maximization ({NUM_STEPS} steps) ...")
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        loss = -model(noise).squeeze()
        loss.backward()
        optimizer.step()

        if step % BLUR_EVERY == 0:
            with torch.no_grad():
                noise.data = blur_tensor(noise.data, BLUR_KSIZE, BLUR_SIGMA)
                noise.data *= (1 - L2_DECAY)

        if step % 100 == 0:
            print(f"        step {step:>4d}  |  pred = {-loss.item():.4f}")

    return noise[0, -1].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# (D) UnifiedFoveatedCNN — activation maximization
# ---------------------------------------------------------------------------

def generate_foveated_stimulus():
    path = os.path.join(CHECKPOINTS, "unified_foveated_model_final.pth")
    print(f"  [D] Loading UnifiedFoveatedCNN from {path}")

    model = _freeze(UnifiedFoveatedCNN(history_size=CNN_HISTORY).to(DEVICE))
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    fovea_img = torch.randn(1, CNN_HISTORY, 1, 20, 20,
                             device=DEVICE, requires_grad=True)
    periph_img = torch.randn(1, CNN_HISTORY, 1, 50, 50,
                              device=DEVICE, requires_grad=True)
    foa = torch.zeros(1, 2, device=DEVICE)

    fy = torch.linspace(-1, 1, 20, device=DEVICE)
    fx = torch.linspace(-1, 1, 20, device=DEVICE)
    fy_map, fx_map = torch.meshgrid(fy, fx, indexing='ij')
    fov_coords = torch.stack([fx_map, fy_map]).unsqueeze(0).unsqueeze(0)
    fov_coords = fov_coords.expand(1, CNN_HISTORY, -1, -1, -1)

    py = torch.linspace(-1, 1, 50, device=DEVICE)
    px = torch.linspace(-1, 1, 50, device=DEVICE)
    py_map, px_map = torch.meshgrid(py, px, indexing='ij')
    per_coords = torch.stack([px_map, py_map]).unsqueeze(0).unsqueeze(0)
    per_coords = per_coords.expand(1, CNN_HISTORY, -1, -1, -1)

    optimizer = torch.optim.Adam([fovea_img, periph_img], lr=LR)

    print(f"      Running activation maximization ({NUM_STEPS} steps) ...")
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        fovea = torch.cat([fovea_img, fov_coords], dim=2)
        periph = torch.cat([periph_img, per_coords], dim=2)
        loss = -model(fovea, periph, foa).squeeze()
        loss.backward()
        optimizer.step()

        if step % BLUR_EVERY == 0:
            with torch.no_grad():
                T = CNN_HISTORY
                f_flat = fovea_img.data.reshape(T, 1, 20, 20)
                fovea_img.data = blur_tensor(f_flat, BLUR_KSIZE, BLUR_SIGMA
                                             ).reshape(1, T, 1, 20, 20)
                p_flat = periph_img.data.reshape(T, 1, 50, 50)
                periph_img.data = blur_tensor(p_flat, BLUR_KSIZE, BLUR_SIGMA
                                              ).reshape(1, T, 1, 50, 50)
                fovea_img.data *= (1 - L2_DECAY)
                periph_img.data *= (1 - L2_DECAY)

        if step % 100 == 0:
            print(f"        step {step:>4d}  |  pred = {-loss.item():.4f}")

    return periph_img[0, -1, 0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Composite 2x3 figure
# ---------------------------------------------------------------------------

def plot_all(real_frame, real_target, glm_rf, ode_rf, ode_polarity,
             drift_stim, foveated_stim, out_path):
    print("  Composing 2x3 figure ...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Real input
    ax = axes[0, 0]
    ax.imshow(real_frame, cmap="gray")
    ax.set_title(f"Real Input (peak Y = {real_target:.2f})",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # (A) GLM
    ax = axes[0, 1]
    vmax = max(abs(glm_rf.min()), abs(glm_rf.max()))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(glm_rf, cmap="RdBu_r", norm=norm)
    ax.set_title("(A) GLM — Spatial RF Weights", fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (B) ODE
    ax = axes[0, 2]
    vmax_ode = max(abs(ode_rf.min()), abs(ode_rf.max()))
    norm_ode = TwoSlopeNorm(vcenter=0, vmin=-vmax_ode, vmax=vmax_ode)
    im = ax.imshow(ode_rf, cmap="RdBu_r", norm=norm_ode)
    ax.set_title(f"(B) ODE — {ode_polarity}-Cell Center-Surround RF",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (C) DriftCNN
    ax = axes[1, 0]
    ax.imshow(drift_stim, cmap="gray")
    ax.set_title("(C) DriftCNN — Optimal Stimulus", fontsize=11,
                 fontweight="bold")
    ax.axis("off")

    # (D) UnifiedFoveatedCNN
    ax = axes[1, 1]
    ax.imshow(foveated_stim, cmap="gray")
    ax.set_title("(D) UnifiedFoveatedCNN — Optimal Stimulus", fontsize=11,
                 fontweight="bold")
    ax.axis("off")

    # Hide empty cell
    axes[1, 2].axis("off")

    fig.suptitle("Receptive Fields & Optimal Stimuli — All Models",
                 fontsize=14, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  All Models — Optimal Stimuli / Receptive Fields")
    print("=" * 60)

    real_frame, real_target = load_real_input()
    glm_rf = generate_glm_rf()
    ode_rf, ode_polarity = generate_ode_rf()
    drift_stim = generate_drift_cnn_stimulus()
    foveated_stim = generate_foveated_stimulus()

    out_path = os.path.join(OUT_DIR, "all_models_optimal_stimuli.png")
    plot_all(real_frame, real_target, glm_rf, ode_rf, ode_polarity,
             drift_stim, foveated_stim, out_path)

    print("\n  Done.")


if __name__ == "__main__":
    main()
