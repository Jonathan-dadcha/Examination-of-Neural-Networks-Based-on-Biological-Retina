"""
CNN Prediction Plots — Ground Truth vs Model Output
====================================================
Generates 1-D time-series plots comparing ground-truth spike rates
with model predictions.  The 300-frame plotting window is drawn from
the VALIDATION set, and the title shows the GLOBAL Pearson r computed
over the entire validation range (passed in from the orchestrator).
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PATH_C_DIR = os.path.join(PROJECT_ROOT, "src", "Path_C_DeepLearning")
PATH_D_DIR = os.path.join(PROJECT_ROOT, "src", "Path_D_Drift_CNN")

sys.path.insert(0, PATH_C_DIR)
sys.path.insert(0, PATH_D_DIR)

from train_cnn_model import DriftCNN
from train_unified_foveated_model import UnifiedFoveatedCNN, UnifiedFoveatedDataset
from drift_dataset import DriftSimulationDataset

SPIKES_PATH = os.path.join(
    PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i",
    "processed_data", "training_dataset_ns_full.h5",
)
IMAGES_PATH = os.path.join(
    PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i",
    "processed_data", "natural_scenes.h5",
)
DRIFT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "drift_model_final.pth")
FOVEATED_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "unified_foveated_model_final.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

DEFAULT_PLOT_START = 20000
DEFAULT_PLOT_SIZE = 300


def pearson_r(preds, targets):
    if np.std(preds) < 1e-9 or np.std(targets) < 1e-9:
        return 0.0
    r = np.corrcoef(preds, targets)[0, 1]
    return r if not np.isnan(r) else 0.0


def evaluate_standard_cnn(dataset, checkpoint_path, device, start_idx, num_samples):
    """Load DriftCNN and run inference on a contiguous window."""
    model = DriftCNN().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for i in range(start_idx, start_idx + num_samples):
            x, y = dataset[i]
            out = model(x.unsqueeze(0).to(device))
            preds.append(out.item())
            targets.append(y.item())

    return np.array(preds), np.array(targets)


def evaluate_unified_foveated(dataset, checkpoint_path, device, start_idx, num_samples):
    """Load UnifiedFoveatedCNN and run inference on a contiguous window."""
    model = UnifiedFoveatedCNN().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for i in range(start_idx, start_idx + num_samples):
            fovea, peripheral, y = dataset[i]
            out = model(
                fovea.unsqueeze(0).to(device),
                peripheral.unsqueeze(0).to(device),
            )
            preds.append(out.item())
            targets.append(y.item())

    return np.array(preds), np.array(targets)


def plot_prediction(targets, preds, title, save_path, global_r=None):
    """
    Plot GT vs prediction.

    If *global_r* is provided it is shown prominently in the title
    (the "true" score computed over the full validation set).  The local
    window r is still printed to the console for reference.
    """
    local_r = pearson_r(preds, targets)
    display_r = global_r if global_r is not None else local_r
    label = "Global Validation Pearson r" if global_r is not None else "Pearson r"

    plt.figure(figsize=(14, 5))
    plt.plot(targets, color="black", linewidth=1, label="Ground Truth (spikes)")
    plt.plot(preds, color="orange", linewidth=1, alpha=0.8, label="CNN Prediction")
    plt.title(f"{title}  |  {label} = {display_r:.4f}", fontsize=13)
    plt.xlabel("Validation Sample Index")
    plt.ylabel("Firing Rate")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Saved: {save_path}")
    if global_r is not None:
        print(f"    Local window r = {local_r:.4f}  |  Global r = {global_r:.4f}")
    else:
        print(f"    Pearson r = {local_r:.4f}")
    return local_r


def main(output_dir=None, global_r_standard=None, global_r_foveated=None,
         plot_start=None, plot_size=None):
    out = output_dir if output_dir else SCRIPT_DIR
    start = plot_start if plot_start is not None else DEFAULT_PLOT_START
    size = plot_size if plot_size is not None else DEFAULT_PLOT_SIZE

    print(f"  Running CNN Prediction Evaluation on {DEVICE}")
    print(f"  Plot window: validation indices {start} → {start + size}\n")

    # ── Standard (Drift) CNN ──────────────────────────────────────────
    print("  --- Standard CNN (DriftCNN) ---")
    drift_dataset = DriftSimulationDataset(IMAGES_PATH, SPIKES_PATH)
    preds_std, targets_std = evaluate_standard_cnn(
        drift_dataset, DRIFT_CHECKPOINT, DEVICE, start, size,
    )
    plot_prediction(
        targets_std, preds_std,
        "Standard CNN (Drift)",
        os.path.join(out, "standard_cnn_prediction_plot.png"),
        global_r=global_r_standard,
    )

    # ── Unified Foveated CNN ──────────────────────────────────────────
    print("\n  --- Unified Foveated CNN ---")
    foveated_dataset = UnifiedFoveatedDataset(IMAGES_PATH, SPIKES_PATH)
    preds_fov, targets_fov = evaluate_unified_foveated(
        foveated_dataset, FOVEATED_CHECKPOINT, DEVICE, start, size,
    )
    plot_prediction(
        targets_fov, preds_fov,
        "Unified Foveated CNN",
        os.path.join(out, "foveated_prediction_plot.png"),
        global_r=global_r_foveated,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
