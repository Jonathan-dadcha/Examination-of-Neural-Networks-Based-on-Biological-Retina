"""
Master Evaluation Pipeline
==========================
Single-command orchestrator that:
  0. Computes Pearson r over a large contiguous evaluation window.
  1. Generates GT-vs-Prediction plots (titled with global r).
  2. Generates foveated split visualisation.
  3. Generates input stimulus video.
  4. Generates kernel weights & Grad-CAM (center-surround proof).
  5. Prints output summary.

Usage
-----
    python src/Analysis/run_all_evaluations.py
"""

import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PATH_D_DIR = os.path.join(PROJECT_ROOT, "src", "Path_D_Drift_CNN")

sys.path.insert(0, PATH_D_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

DRIFT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "drift_model_final.pth")
FOVEATED_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "unified_foveated_model_final.pth")

SPIKES_PATH = os.path.join(
    PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i",
    "processed_data", "training_dataset_ns_full.h5",
)
IMAGES_PATH = os.path.join(
    PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i",
    "processed_data", "natural_scenes.h5",
)

# ── Evaluation window (contiguous region within biologically active data) ─
EVAL_START = 20000
EVAL_END = 50000

# ── 300-frame window for plots ────────────────────────────────────────────
PLOT_WINDOW_START = 20000
PLOT_WINDOW_SIZE = 300

# ── Grad-CAM sample ──────────────────────────────────────────────────────
GRADCAM_SAMPLE_IDX = 20000

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 256

SEPARATOR = "=" * 65


def _banner(msg: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {msg}")
    print(SEPARATOR)


def _pearson_r(preds, targets):
    if np.std(preds) < 1e-9 or np.std(targets) < 1e-9:
        return 0.0
    r = np.corrcoef(preds, targets)[0, 1]
    return r if not np.isnan(r) else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Step 0 — Global Validation Scoring
# ══════════════════════════════════════════════════════════════════════════

def compute_global_validation_scores() -> tuple:
    """
    Load both models and compute Pearson r over a large contiguous
    evaluation window (indices EVAL_START … EVAL_END).

    Returns (global_r_standard, global_r_foveated).
    """
    from train_drift_model import DriftCNN
    from train_unified_foveated_model import UnifiedFoveatedCNN, UnifiedFoveatedDataset
    from drift_dataset import DriftSimulationDataset

    _banner("Step 0/5 — Global Evaluation Scoring")
    print(f"  Evaluation range : indices {EVAL_START} → {EVAL_END}")
    print(f"  Expected samples : {EVAL_END - EVAL_START}")
    print(f"  Device           : {DEVICE}\n")

    # ── DriftCNN ──────────────────────────────────────────────────────────
    print("  [1/2] DriftCNN — loading dataset …")
    drift_ds = DriftSimulationDataset(IMAGES_PATH, SPIKES_PATH)
    val_end = min(EVAL_END, len(drift_ds))
    val_indices = list(range(EVAL_START, val_end))
    n_val = len(val_indices)

    model_std = DriftCNN().to(DEVICE)
    model_std.load_state_dict(
        torch.load(DRIFT_CHECKPOINT, map_location=DEVICE, weights_only=False)
    )
    model_std.eval()

    drift_loader = DataLoader(
        Subset(drift_ds, val_indices),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    preds_s, tgts_s = [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, (x, y) in enumerate(drift_loader):
            out = model_std(x.to(DEVICE))
            preds_s.extend(out.cpu().numpy().flatten())
            tgts_s.extend(y.numpy().flatten())
            if (bi + 1) % 25 == 0:
                print(f"    DriftCNN: {len(preds_s):>6}/{n_val} samples …")

    global_r_std = _pearson_r(np.array(preds_s), np.array(tgts_s))
    print(f"  DriftCNN done — {time.time() - t0:.1f}s  "
          f"({n_val} samples, r = {global_r_std:.4f})")

    del model_std, drift_ds, drift_loader

    # ── UnifiedFoveatedCNN ────────────────────────────────────────────────
    print("\n  [2/2] UnifiedFoveatedCNN — loading dataset …")
    fov_ds = UnifiedFoveatedDataset(IMAGES_PATH, SPIKES_PATH)
    val_end_fov = min(EVAL_END, len(fov_ds))
    val_indices_fov = list(range(EVAL_START, val_end_fov))
    n_val_fov = len(val_indices_fov)

    model_fov = UnifiedFoveatedCNN().to(DEVICE)
    model_fov.load_state_dict(
        torch.load(FOVEATED_CHECKPOINT, map_location=DEVICE, weights_only=False)
    )
    model_fov.eval()

    fov_loader = DataLoader(
        Subset(fov_ds, val_indices_fov),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    preds_f, tgts_f = [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, (fovea, peripheral, foa_coords, y) in enumerate(fov_loader):
            out = model_fov(
                fovea.to(DEVICE),
                peripheral.to(DEVICE),
                foa_coords.to(DEVICE),
            )
            preds_f.extend(out.cpu().numpy().flatten())
            tgts_f.extend(y.numpy().flatten())
            if (bi + 1) % 25 == 0:
                print(f"    FoveatedCNN: {len(preds_f):>6}/{n_val_fov} samples …")

    global_r_fov = _pearson_r(np.array(preds_f), np.array(tgts_f))
    print(f"  UnifiedFoveatedCNN done — {time.time() - t0:.1f}s  "
          f"({n_val_fov} samples, r = {global_r_fov:.4f})")

    del model_fov, fov_ds, fov_loader

    # ── Results banner ────────────────────────────────────────────────────
    star = "*" * 65
    print(f"\n{star}")
    print(f"  GLOBAL EVALUATION RESULTS  (indices {EVAL_START} – {val_end})")
    print(f"{star}")
    print(f"  Standard DriftCNN        Pearson r = {global_r_std:.4f}")
    print(f"  Unified Foveated CNN     Pearson r = {global_r_fov:.4f}")
    print(f"{star}\n")

    return global_r_std, global_r_fov


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint guard
# ══════════════════════════════════════════════════════════════════════════

def ensure_checkpoints() -> None:
    """Train models whose checkpoint files are missing."""
    _banner("Checking model checkpoints")

    if os.path.isfile(DRIFT_CHECKPOINT):
        print(f"  [OK] DriftCNN checkpoint found: {DRIFT_CHECKPOINT}")
    else:
        print("  [!!] DriftCNN checkpoint NOT found — launching training …")
        saved_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            from train_drift_model import main as train_drift
            train_drift()
        finally:
            os.chdir(saved_cwd)
        print("  [OK] DriftCNN training complete.")

    if os.path.isfile(FOVEATED_CHECKPOINT):
        print(f"  [OK] UnifiedFoveatedCNN checkpoint found: {FOVEATED_CHECKPOINT}")
    else:
        print("  [!!] UnifiedFoveatedCNN checkpoint NOT found — launching training …")
        saved_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            from train_unified_foveated_model import main as train_foveated
            train_foveated()
        finally:
            os.chdir(saved_cwd)
        print("  [OK] UnifiedFoveatedCNN training complete.")


# ══════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline() -> None:
    """Execute all evaluation steps, directing output to results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n  All outputs will be saved to: {RESULTS_DIR}\n")

    # ------------------------------------------------------------------
    # Step 0 — Global validation scoring (the TRUE scores)
    # ------------------------------------------------------------------
    t0_global = time.time()
    global_r_std, global_r_fov = compute_global_validation_scores()
    print(f"  Global scoring finished in {time.time() - t0_global:.1f}s")

    # ------------------------------------------------------------------
    # Step 1 — GT vs Prediction plots (validation window, global r title)
    # ------------------------------------------------------------------
    _banner("Step 1/5 — CNN Prediction Plots (Validation Window)")
    print("  Generating 1-D Ground-Truth vs Prediction curves …")
    t0 = time.time()

    import plot_cnn_predictions
    plot_cnn_predictions.main(
        output_dir=RESULTS_DIR,
        global_r_standard=global_r_std,
        global_r_foveated=global_r_fov,
        plot_start=PLOT_WINDOW_START,
        plot_size=PLOT_WINDOW_SIZE,
    )

    print(f"  Finished in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2 — Foveated split visualisation (1x3 panel)
    # ------------------------------------------------------------------
    _banner("Step 2/5 — Foveated Split Visualisation")
    print("  Generating 1x3 FOA decomposition subplot …")
    t0 = time.time()

    import visualize_fovea_split
    visualize_fovea_split.main(output_dir=RESULTS_DIR)

    print(f"  Finished in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 3 — Input stimulus video (GIF + MP4)
    # ------------------------------------------------------------------
    _banner("Step 3/5 — Input Stimulus Video")
    print("  Generating animated peripheral input stimulus …")
    t0 = time.time()

    import generate_input_video
    generate_input_video.main(output_dir=RESULTS_DIR)

    print(f"  Finished in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 4 — Kernel weights & Grad-CAM (center-surround proof)
    # ------------------------------------------------------------------
    _banner("Step 4/5 — Kernel Weights & Grad-CAM (Center-Surround Proof)")
    print("  Generating first-layer kernels and Grad-CAM overlay …")
    t0 = time.time()

    import visualize_kernels_and_explainability
    visualize_kernels_and_explainability.main(
        output_dir=RESULTS_DIR,
        sample_idx=GRADCAM_SAMPLE_IDX,
    )

    print(f"  Finished in {time.time() - t0:.1f}s")


def print_summary() -> None:
    """List every file written to results/."""
    _banner("Step 5/5 — Pipeline Complete — Output Summary")
    files = sorted(os.listdir(RESULTS_DIR))
    if not files:
        print("  (no files found in results/)")
        return
    for f in files:
        full = os.path.join(RESULTS_DIR, f)
        size_kb = os.path.getsize(full) / 1024
        print(f"  {f:45s}  {size_kb:8.1f} KB")
    print(f"\n  Total: {len(files)} file(s) in {RESULTS_DIR}")


def main() -> None:
    _banner("Master Evaluation Pipeline")
    t_start = time.time()

    ensure_checkpoints()
    run_pipeline()
    print_summary()

    elapsed = time.time() - t_start
    print(f"\n  Total pipeline time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
