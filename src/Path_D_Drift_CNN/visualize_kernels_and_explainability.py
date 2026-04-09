"""
Kernel Visualization and Grad-CAM Explainability for UnifiedFoveatedCNN.

Generates two figures for reviewer response:
  1. kernel_weights.png              – First-layer spatial filters (most recent frame)
  2. gradcam_regression_heatmap.png  – Center-surround architecture proof:
        Row 0: Fovea input | Gaussian mask | Masked peripheral | Grad-CAM heatmap
        Row 1: Peripheral feature maps (per channel)
        Row 2: Grad-CAM overlay | Radial decay profile (Gaussian vs Grad-CAM)

The Grad-CAM sample is drawn from the dataset (default idx=20000).
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_unified_foveated_model import (
    UnifiedFoveatedCNN,
    UnifiedFoveatedDataset,
    IMAGES_PATH,
    SPIKES_PATH,
)

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "unified_foveated_model_final.pth"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEVICE = torch.device("cpu")

DEFAULT_GRADCAM_IDX = 20000


def load_model():
    """Load trained UnifiedFoveatedCNN from checkpoint (CPU)."""
    model = UnifiedFoveatedCNN()
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    print(f"  Loaded checkpoint: {CHECKPOINT}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — First-Layer Kernel Weights
# ═══════════════════════════════════════════════════════════════════════════════


def plot_kernels(model, out_dir=None):
    """
    Extract the spatial kernel weights for the MOST RECENT time step
    (channel index -1) from the first Conv2d in each stream and plot them
    in a grid with a diverging colormap centred at zero.
    """
    dest = out_dir if out_dir else OUT_DIR

    periph_w = model.peripheral_stream[0].weight.detach().cpu()  # (8, 3, 5, 5)
    fovea_w = model.fovea_stream[0].weight.detach().cpu()        # (16, 3, 9, 9)

    periph_k = periph_w[:, 0]  # (8, 5, 5) — image channel
    fovea_k = fovea_w[:, 0]    # (16, 9, 9) — image channel

    n_p, n_f = periph_k.shape[0], fovea_k.shape[0]
    ncols = 8
    nrows_p = 1
    nrows_f = (n_f + ncols - 1) // ncols
    nrows = nrows_p + nrows_f

    vmax = max(periph_k.abs().max().item(), fovea_k.abs().max().item())
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))

    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis("off")

    for j in range(n_p):
        ax = axes[0, j]
        ax.imshow(periph_k[j].numpy(), cmap="RdBu_r", norm=norm)
        ax.set_title(f"Periph {j}", fontsize=8)

    for i in range(n_f):
        r, c = nrows_p + i // ncols, i % ncols
        ax = axes[r, c]
        ax.imshow(fovea_k[i].numpy(), cmap="RdBu_r", norm=norm)
        ax.set_title(f"Fovea {i}", fontsize=8)

    fig.suptitle(
        "First Layer Kernels (Most Recent Frame)", fontsize=13, fontweight="bold"
    )

    sm = cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Weight value", fontsize=9)

    fig.subplots_adjust(right=0.88, top=0.92, hspace=0.4)

    path = os.path.join(dest, "kernel_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Center-Surround Architecture Proof (Grad-CAM)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_gradcam(model, sample_idx=None, out_dir=None):
    """
    For a single validation sample, produce a comprehensive figure
    proving the center-surround architecture:

      Row 0  Fovea input | Gaussian mask | Masked peripheral | Grad-CAM heatmap
      Row 1  Peripheral feature maps (one per channel)
      Row 2  Grad-CAM overlay | Radial decay profile

    The radial profile plots the designed Gaussian attenuation alongside
    the learned Grad-CAM intensity as a function of distance from center,
    visually proving that computation drops toward the periphery.
    """
    idx = sample_idx if sample_idx is not None else DEFAULT_GRADCAM_IDX
    dest = out_dir if out_dir else OUT_DIR

    print(f"    Using validation sample idx = {idx}")

    dataset = UnifiedFoveatedDataset(IMAGES_PATH, SPIKES_PATH)
    fovea, peripheral, foa_coords, target = dataset[idx]

    fovea_in = fovea.unsqueeze(0).to(DEVICE)
    peripheral_in = peripheral.unsqueeze(0).to(DEVICE)
    foa_in = foa_coords.unsqueeze(0).to(DEVICE)

    # ── Forward hook on ReLU after last Conv2d in peripheral_stream ───
    # Sequential: Conv2d[0] BN[1] ReLU[2] Conv2d[3] BN[4] ReLU[5] Pool[6]
    activation_store = {}

    def _fwd_hook(_module, _inp, out):
        activation_store["A"] = out
        out.retain_grad()

    handle = model.peripheral_stream[5].register_forward_hook(_fwd_hook)

    output = model(fovea_in, peripheral_in, foa_in)
    pred = output.item()

    model.zero_grad()
    output.backward()

    A = activation_store["A"].detach().cpu().squeeze(0)       # (4, 30, 30)
    dA = activation_store["A"].grad.detach().cpu().squeeze(0)
    handle.remove()

    # ── Grad-CAM: weight feature maps by global-avg-pooled gradients ──
    alpha = dA.mean(dim=(1, 2))
    cam = sum(alpha[k] * A[k] for k in range(A.shape[0]))
    cam = F.relu(cam)

    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(50, 50),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()

    lo, hi = cam_up.min(), cam_up.max()
    cam_up = (cam_up - lo) / (hi - lo + 1e-8)

    # ── Extract frames for visualisation ──────────────────────────────
    periph_frame = peripheral_in[0, -1, 0].detach().cpu().numpy()   # (50, 50)
    fovea_frame = fovea_in[0, -1, 0].detach().cpu().numpy()       # (20, 20)

    # ── Reconstruct Gaussian mask (matches dataset implementation) ────
    crop_size = 50
    sigma = crop_size / 4.0
    cx, cy = crop_size / 2.0, crop_size / 2.0
    ys, xs = np.mgrid[0:crop_size, 0:crop_size].astype(np.float64)
    gauss_mask = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))

    # ── Composite figure (3 rows × 4 cols) ────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)

    # Row 0 — architecture inputs & Grad-CAM heatmap
    ax_fov = fig.add_subplot(gs[0, 0])
    ax_fov.imshow(fovea_frame, cmap="gray")
    ax_fov.set_title("Fovea Input\n(Full Resolution 20\u00d720)", fontsize=10,
                     fontweight="bold")
    ax_fov.axis("off")

    ax_mask = fig.add_subplot(gs[0, 1])
    im_mask = ax_mask.imshow(gauss_mask, cmap="hot", vmin=0, vmax=1)
    ax_mask.set_title(f"Gaussian Decay Mask\n(\u03c3 = {sigma:.1f} px)",
                      fontsize=10, fontweight="bold")
    ax_mask.axis("off")
    plt.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)

    ax_per = fig.add_subplot(gs[0, 2])
    ax_per.imshow(periph_frame, cmap="gray")
    ax_per.set_title("Peripheral Input\n(After Gaussian Masking)", fontsize=10,
                     fontweight="bold")
    ax_per.axis("off")

    ax_cam = fig.add_subplot(gs[0, 3])
    im_cam = ax_cam.imshow(cam_up, cmap="jet", vmin=0, vmax=1)
    ax_cam.set_title("Grad-CAM Heatmap\n(Peripheral Stream)", fontsize=10,
                     fontweight="bold")
    ax_cam.axis("off")
    plt.colorbar(im_cam, ax=ax_cam, fraction=0.046, pad=0.04)

    # Row 1 — peripheral feature maps
    n_ch = A.shape[0]
    for ch in range(n_ch):
        ax = fig.add_subplot(gs[1, ch])
        ax.imshow(A[ch].numpy(), cmap="viridis")
        ax.set_title(f"Periph Feature Map {ch}", fontsize=9)
        ax.axis("off")

    # Row 2 — overlay (left half) + radial profile (right half)
    ax_ov = fig.add_subplot(gs[2, 0:2])
    ax_ov.imshow(periph_frame, cmap="gray")
    ax_ov.imshow(cam_up, cmap="jet", alpha=0.5)
    ax_ov.set_title(
        f"Grad-CAM Overlay  |  Pred: {pred:.3f}  |  Target: {target.item():.3f}",
        fontsize=11, fontweight="bold",
    )
    ax_ov.axis("off")

    # Radial decay profile — quantitative center-surround proof
    ax_rad = fig.add_subplot(gs[2, 2:4])
    center = np.array([cx, cy])
    distances = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    max_r = distances.max()
    n_bins = 30
    r_bins = np.linspace(0, max_r, n_bins + 1)

    gauss_profile, cam_profile = [], []
    for i in range(n_bins):
        ring = (distances >= r_bins[i]) & (distances < r_bins[i + 1])
        if ring.any():
            gauss_profile.append(gauss_mask[ring].mean())
            cam_profile.append(cam_up[ring].mean())
        else:
            gauss_profile.append(0.0)
            cam_profile.append(0.0)

    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0

    ax_rad.plot(r_centers, gauss_profile, "r-", linewidth=2.5,
                label="Gaussian Mask (designed attenuation)")
    ax_rad.plot(r_centers, cam_profile, "b-", linewidth=2.5,
                label="Grad-CAM (learned attention)")
    ax_rad.fill_between(r_centers, 0, gauss_profile, color="red", alpha=0.12)
    ax_rad.fill_between(r_centers, 0, cam_profile, color="blue", alpha=0.12)
    ax_rad.set_xlabel("Distance from Center (pixels)", fontsize=11)
    ax_rad.set_ylabel("Normalized Intensity", fontsize=11)
    ax_rad.set_title("Radial Decay Profile: Center \u2192 Periphery", fontsize=11,
                     fontweight="bold")
    ax_rad.legend(fontsize=9, loc="upper right")
    ax_rad.grid(True, alpha=0.3)
    ax_rad.set_xlim(0, max_r)
    ax_rad.set_ylim(0, 1.05)

    fig.suptitle(
        "Center\u2013Surround Architecture: "
        "Foveated Computation with Peripheral Gaussian Decay\n"
        "(Dilated convolutions + spatial mask \u2192 "
        "computation drops with distance from center)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    path = os.path.join(dest, "gradcam_regression_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════

def main(output_dir=None, sample_idx=None):
    print("=" * 60)
    print("  Kernel & Grad-CAM Visualization — UnifiedFoveatedCNN")
    print("=" * 60)

    model = load_model()

    print("\n  [Part 1] First-layer kernel weights …")
    plot_kernels(model, out_dir=output_dir)

    print("\n  [Part 2] Center-surround Grad-CAM proof …")
    plot_gradcam(model, sample_idx=sample_idx, out_dir=output_dir)

    print("\n  All visualizations complete.")


if __name__ == "__main__":
    main()
