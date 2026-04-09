"""
master_pipeline_24h.py — Master Training + Evaluation Pipeline
===============================================================
Sequential training of all 4 retina models, followed by unified evaluation
on a contiguous 300-sample test window with synchronized video.

Models:
  A. GLM  (Path A) — sklearn PoissonRegressor via StandardScaler -> PCA -> GLM
  B. ODE  (Path B) — Leaky-integrator optimized with scipy differential_evolution
  C. DriftCNN (Path C) — Standard CNN with biological micro-drift augmentation
  D. UnifiedFoveatedCNN (Path D) — Two-stream foveated architecture (dilated + NM-FCL)

Outputs:
  checkpoints/glm_pipeline.joblib
  checkpoints/ode_params.json
  checkpoints/drift_model_final.pth
  checkpoints/unified_foveated_model_final.pth
  results/all_models_comparison.png
  results/test_stimulus_sync.mp4

Usage:
    python src/Analysis/master_pipeline_24h.py
"""

import sys
import os
import json
import time
import copy

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor

from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Path setup — import model classes from existing codebase
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PATH_D_DIR = os.path.join(PROJECT_ROOT, "src", "Path_D_Drift_CNN")
sys.path.insert(0, PATH_D_DIR)

from train_drift_model import DriftCNN  # noqa: E402  # type: ignore[import-not-found]
from train_unified_foveated_model import UnifiedFoveatedCNN, UnifiedFoveatedDataset  # noqa: E402  # type: ignore[import-not-found]
from drift_dataset import DriftSimulationDataset  # noqa: E402  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPIKES_PATH = os.path.join(
    PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i",
    "processed_data", "training_dataset_ns_full.h5",
)
IMAGES_PATH = os.path.join(
    PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i",
    "processed_data", "natural_scenes.h5",
)
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

TEST_START = 20000
NUM_SAMPLES = 300

CNN_HISTORY = 40
BATCH_SIZE = 256
DRIFTCNN_EPOCHS = 100
FOVEATED_EPOCHS = 120
LR = 1e-3

TRAIN_SPLIT_RATIO = 0.80

GLM_HISTORY = 20
GLM_CROP_HALF = 12

CENTER_X = 27
CENTER_Y = 47
DT = 1.0 / 60.0
ODE_TRAIN_END = 144_480  # 80% of 180600

SEPARATOR = "=" * 70


def _banner(msg: str) -> None:
    print(f"\n{SEPARATOR}\n  {msg}\n{SEPARATOR}")


def pearson_r(preds, targets):
    if np.std(preds) < 1e-9 or np.std(targets) < 1e-9:
        return 0.0
    r = np.corrcoef(preds, targets)[0, 1]
    return r if not np.isnan(r) else 0.0


# ===================================================================
# Section 1 — Data Loading & Inspection
# ===================================================================
def load_raw_data():
    """Load raw X, Y arrays from HDF5 (full dataset, no truncation)."""
    _banner("Section 1 — Loading Raw Data")
    print(f"  File: {SPIKES_PATH}")

    with h5py.File(SPIKES_PATH, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 keys: {keys}")
        X = f["X"][:]
        Y = f["Y"][:]

    print(f"  X shape: {X.shape}   dtype: {X.dtype}")
    print(f"  Y shape: {Y.shape}   dtype: {Y.dtype}")
    print(f"  X value range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Y value range: [{Y.min():.4f}, {Y.max():.4f}]")

    gt_slice = Y[TEST_START : TEST_START + NUM_SAMPLES]
    print(f"\n  Evaluation ground-truth slice: Y[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    print(f"  GT mean={gt_slice.mean():.4f}  std={gt_slice.std():.4f}  "
          f"min={gt_slice.min():.4f}  max={gt_slice.max():.4f}")

    return X, Y


# ===================================================================
# Section 2 — Train GLM (Path A)
# ===================================================================
def train_glm(X_raw, Y):
    _banner("Section 2 — Training Model A: GLM (PoissonRegressor)")

    cx, cy = CENTER_X, CENTER_Y
    ch = GLM_CROP_HALF
    x_s, x_e = max(cx - ch, 0), min(cx + ch + 1, X_raw.shape[2])
    y_s, y_e = max(cy - ch, 0), min(cy + ch + 1, X_raw.shape[1])
    X_crop = X_raw[:, y_s:y_e, x_s:x_e].astype(np.float32)

    print(f"  Crop: y[{y_s}:{y_e}], x[{x_s}:{x_e}] -> patch {X_crop.shape[1:]}")
    print(f"  GLM history: {GLM_HISTORY} frames")

    X_flat = X_crop.reshape(X_crop.shape[0], -1)
    pixel_dim = X_flat.shape[1]

    train_end = ODE_TRAIN_END
    print(f"  Building design matrix for indices [{GLM_HISTORY}, {train_end}) ...")

    t0 = time.time()
    n_train = train_end - GLM_HISTORY
    X_train = np.empty((n_train, GLM_HISTORY * pixel_dim), dtype=np.float32)
    y_train = np.empty(n_train, dtype=np.float32)

    for i, t in enumerate(range(GLM_HISTORY, train_end)):
        X_train[i] = X_flat[t - GLM_HISTORY : t].ravel()
        y_train[i] = Y[t]

    print(f"  Design matrix: {X_train.shape}  ({X_train.nbytes / 1e6:.1f} MB)")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50)),
        ("glm", PoissonRegressor(alpha=1.0, max_iter=1000)),
    ])

    print("  Fitting: StandardScaler -> PCA(50) -> PoissonRegressor ...")
    pipeline.fit(X_train, y_train)

    explained = pipeline.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained:.2%}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    try:
        import joblib
        ckpt = os.path.join(CHECKPOINTS_DIR, "glm_pipeline.joblib")
        joblib.dump(pipeline, ckpt)
        print(f"  Saved: {ckpt}")
    except ImportError:
        print("  [WARN] joblib not installed — GLM not persisted to disk")

    print(f"  GLM training finished in {time.time() - t0:.1f}s")
    return pipeline, X_crop


# ===================================================================
# Section 3 — Train ODE (Path B)
# ===================================================================
def _simulate_ode(stimulus, tau, gain, bias, dt):
    """Vectorized leaky-integrator followed by softplus nonlinearity."""
    alpha = np.exp(-dt / tau)
    y = lfilter([1 - alpha], [1, -alpha], stimulus)
    return np.log1p(np.exp(gain * y + bias))


def train_ode(X_raw, Y):
    _banner("Section 3 — Training Model B: Retinomorphic ODE")

    c_h, s_h = 1, 4
    center = np.mean(
        X_raw[:, CENTER_Y - c_h : CENTER_Y + c_h + 1,
              CENTER_X - c_h : CENTER_X + c_h + 1],
        axis=(1, 2),
    )
    surround = np.mean(
        X_raw[:, CENTER_Y - s_h : CENTER_Y + s_h + 1,
              CENTER_X - s_h : CENTER_X + s_h + 1],
        axis=(1, 2),
    )

    contrast_on = center - surround
    contrast_off = surround - center
    contrast_on = (contrast_on - contrast_on.mean()) / (contrast_on.std() + 1e-6)
    contrast_off = (contrast_off - contrast_off.mean()) / (contrast_off.std() + 1e-6)

    spikes = Y.astype(np.float64)
    train_end = ODE_TRAIN_END

    print(f"  Center pixel: ({CENTER_X}, {CENTER_Y})")
    print(f"  Center half-width: {c_h},  Surround half-width: {s_h}")
    print(f"  Training on frames [0, {train_end})")

    bounds = [(0.001, 0.5), (0.1, 10.0), (-5.0, 5.0)]

    def objective(params, stimulus, y):
        tau, gain, bias = params
        pred = _simulate_ode(stimulus, tau, gain, bias, DT)
        if np.std(pred) < 1e-9:
            return 0.0
        corr = np.corrcoef(pred, y)[0, 1]
        return -corr if not np.isnan(corr) else 0.0

    t0 = time.time()
    best_result, best_corr, best_label = None, -np.inf, None

    for label, stim in [("ON", contrast_on), ("OFF", contrast_off)]:
        print(f"  Optimizing {label} polarity (differential_evolution) ...")
        result = differential_evolution(
            objective,
            bounds=bounds,
            args=(stim[:train_end], spikes[:train_end]),
            seed=42, maxiter=200, tol=1e-6, polish=True,
        )
        corr = -result.fun
        tau, gain, bias = result.x
        print(f"    {label}: r={corr:.4f}  tau={tau:.4f}  gain={gain:.4f}  bias={bias:.4f}")
        if corr > best_corr:
            best_corr, best_result, best_label = corr, result, label

    tau, gain, bias = best_result.x
    print(f"  Best polarity: {best_label}-Cell")

    best_stim = contrast_on if best_label == "ON" else contrast_off

    ode_params = {
        "polarity": best_label,
        "tau": float(tau), "gain": float(gain), "bias": float(bias),
        "train_corr": float(best_corr),
    }
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINTS_DIR, "ode_params.json")
    with open(ckpt, "w") as f:
        json.dump(ode_params, f, indent=2)
    print(f"  Saved: {ckpt}")
    print(f"  ODE training finished in {time.time() - t0:.1f}s")

    return ode_params, best_stim


# ===================================================================
# Section 4 — Train DriftCNN (Path C)
# ===================================================================
def _run_dl_training_loop(model, train_loader, val_loader, model_name, max_epochs,
                          checkpoint_path=None):
    """Adam + CosineAnnealing, with optional checkpoint resume every 10 epochs."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.PoissonNLLLoss(log_input=False)

    best_val_loss = float("inf")
    best_state = None
    is_foveated = "Foveated" in model_name
    start_epoch = 1

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 2  # ckpt['epoch'] is 0-based, loop is 1-based
        best_val_loss = ckpt['best_val_loss']
        best_state = ckpt['best_state']
        print(f"  Resumed from epoch {start_epoch}, "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

    t0 = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_loader:
            if is_foveated:
                fovea, periph, foa, targets = batch
                fovea, periph = fovea.to(DEVICE), periph.to(DEVICE)
                foa, targets = foa.to(DEVICE), targets.to(DEVICE)
                outputs = model(fovea, periph, foa)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        model.eval()
        val_loss, v_batches = 0.0, 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_foveated:
                    fovea, periph, foa, targets = batch
                    fovea, periph = fovea.to(DEVICE), periph.to(DEVICE)
                    foa, targets = foa.to(DEVICE), targets.to(DEVICE)
                    outputs = model(fovea, periph, foa)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item()
                v_batches += 1

                all_val_preds.extend(outputs.cpu().numpy().flatten())
                all_val_targets.extend(targets.cpu().numpy().flatten())

        avg_val = val_loss / max(v_batches, 1)

        vp = np.array(all_val_preds)
        vt = np.array(all_val_targets)
        if np.std(vp) < 1e-9 or np.std(vt) < 1e-9:
            val_r = 0.0
        else:
            val_r, _ = pearsonr(vt, vp)
            if np.isnan(val_r):
                val_r = 0.0

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

        if epoch <= 5 or epoch % 10 == 0 or improved:
            flag = " *" if improved else ""
            print(f"  Epoch {epoch:4d}/{max_epochs} | "
                  f"Train {avg_train:.5f} | Val {avg_val:.5f} | "
                  f"Correlation: {val_r:.4f}{flag}")

        if checkpoint_path and epoch % 10 == 0:
            torch.save({
                'epoch': epoch - 1,  # store 0-based for compat with standalone
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_state': best_state,
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch}")

    elapsed = time.time() - t0
    print(f"  Best val loss: {best_val_loss:.5f}")
    print(f"  {model_name} training finished in {elapsed:.1f}s")

    return best_state


def train_driftcnn():
    _banner("Section 4 — Training Model C: DriftCNN")

    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {DRIFTCNN_EPOCHS}")
    print(f"  Optimizer: Adam(lr={LR})")
    print("  Loss: PoissonNLLLoss(log_input=False)")
    print("  Split: 80/20 random")

    dataset = DriftSimulationDataset(IMAGES_PATH, SPIKES_PATH)
    n = len(dataset)
    train_size = int(TRAIN_SPLIT_RATIO * n)
    test_size = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, test_size])

    print(f"  Random split: train {len(train_ds)} | val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = DriftCNN(history_size=CNN_HISTORY).to(DEVICE)
    best_state = _run_dl_training_loop(model, train_loader, val_loader,
                                       "DriftCNN", DRIFTCNN_EPOCHS)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINTS_DIR, "drift_model_final.pth")
    torch.save(best_state, ckpt)
    print(f"  Saved: {ckpt}")

    return dataset


# ===================================================================
# Section 5 — Train UnifiedFoveatedCNN (Path D)
# ===================================================================
def train_unified_foveated():
    _banner("Section 5 — Training Model D: UnifiedFoveatedCNN")

    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {FOVEATED_EPOCHS}")
    print(f"  Optimizer: Adam(lr={LR})")
    print("  Loss: PoissonNLLLoss(log_input=False)")
    print("  Split: 80/20 random")

    dataset = UnifiedFoveatedDataset(IMAGES_PATH, SPIKES_PATH)
    n = len(dataset)
    train_size = int(TRAIN_SPLIT_RATIO * n)
    test_size = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, test_size])

    print(f"  Random split: train {len(train_ds)} | val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = UnifiedFoveatedCNN(history_size=CNN_HISTORY).to(DEVICE)
    ckpt_resume = os.path.join(CHECKPOINTS_DIR, "unified_foveated_checkpoint.pth")
    best_state = _run_dl_training_loop(model, train_loader, val_loader,
                                       "UnifiedFoveatedCNN", FOVEATED_EPOCHS,
                                       checkpoint_path=ckpt_resume)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINTS_DIR, "unified_foveated_model_final.pth")
    torch.save(best_state, ckpt)
    print(f"  Saved: {ckpt}")

    return dataset


# ===================================================================
# Section 6 — Unified Evaluation (all 4 models, same contiguous window)
# ===================================================================
def evaluate_all(X_raw, Y, glm_pipeline, glm_crop, ode_params, ode_stim,
                 drift_dataset, foveated_dataset):
    _banner("Section 6 — Unified Evaluation on Contiguous Window")

    gt = Y[TEST_START : TEST_START + NUM_SAMPLES].astype(np.float64)

    print(f"  Ground truth: Y[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    print(f"  GT shape: {gt.shape}  mean={gt.mean():.4f}  std={gt.std():.4f}")
    print(f"  Strictly contiguous indices: {TEST_START}, {TEST_START+1}, ..., "
          f"{TEST_START + NUM_SAMPLES - 1}")

    results = {}

    # ----- A. GLM -----
    print(f"\n  [A] GLM — Predicting Y[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    X_flat = glm_crop.reshape(glm_crop.shape[0], -1)
    pixel_dim = X_flat.shape[1]
    X_test_glm = np.empty((NUM_SAMPLES, GLM_HISTORY * pixel_dim), dtype=np.float32)
    for i, t in enumerate(range(TEST_START, TEST_START + NUM_SAMPLES)):
        X_test_glm[i] = X_flat[t - GLM_HISTORY : t].ravel()

    print(f"      Design matrix: {X_test_glm.shape}")
    preds_glm = glm_pipeline.predict(X_test_glm)
    r_glm = pearson_r(preds_glm, gt)
    print(f"      Pearson r = {r_glm:.4f}")
    results["GLM"] = preds_glm

    # ----- B. ODE -----
    print(f"\n  [B] ODE ({ode_params['polarity']}-Cell) — "
          f"Predicting Y[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    tau, gain, bias = ode_params["tau"], ode_params["gain"], ode_params["bias"]
    full_pred = _simulate_ode(ode_stim, tau, gain, bias, DT)
    preds_ode = full_pred[TEST_START : TEST_START + NUM_SAMPLES]
    r_ode = pearson_r(preds_ode, gt)
    print(f"      Pearson r = {r_ode:.4f}")
    results["ODE"] = preds_ode

    # ----- C. DriftCNN -----
    first_ds_idx = TEST_START - CNN_HISTORY + 1
    print(f"\n  [C] DriftCNN — Predicting Y[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    print(f"      Dataset indices: [{first_ds_idx}, {first_ds_idx + NUM_SAMPLES})")

    drift_ckpt = os.path.join(CHECKPOINTS_DIR, "drift_model_final.pth")
    model_c = DriftCNN(history_size=CNN_HISTORY).to(DEVICE)
    model_c.load_state_dict(
        torch.load(drift_ckpt, map_location=DEVICE, weights_only=False)
    )
    model_c.eval()

    np.random.seed(0)
    preds_cnn = []
    with torch.no_grad():
        for t in range(TEST_START, TEST_START + NUM_SAMPLES):
            ds_idx = t - CNN_HISTORY + 1
            x_input, _ = drift_dataset[ds_idx]
            out = model_c(x_input.unsqueeze(0).to(DEVICE))
            preds_cnn.append(out.item())

    preds_cnn = np.array(preds_cnn)
    r_cnn = pearson_r(preds_cnn, gt)
    print(f"      Pearson r = {r_cnn:.4f}")
    results["DriftCNN"] = preds_cnn

    # ----- D. UnifiedFoveatedCNN -----
    print(f"\n  [D] UnifiedFoveatedCNN — Predicting Y[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    print(f"      Dataset indices: [{first_ds_idx}, {first_ds_idx + NUM_SAMPLES})")

    fov_ckpt = os.path.join(CHECKPOINTS_DIR, "unified_foveated_model_final.pth")
    model_d = UnifiedFoveatedCNN(history_size=CNN_HISTORY).to(DEVICE)
    model_d.load_state_dict(
        torch.load(fov_ckpt, map_location=DEVICE, weights_only=False)
    )
    model_d.eval()

    np.random.seed(0)
    preds_fov = []
    with torch.no_grad():
        for t in range(TEST_START, TEST_START + NUM_SAMPLES):
            ds_idx = t - CNN_HISTORY + 1
            fovea, peripheral, foa_coords, _ = foveated_dataset[ds_idx]
            out = model_d(
                fovea.unsqueeze(0).to(DEVICE),
                peripheral.unsqueeze(0).to(DEVICE),
                foa_coords.unsqueeze(0).to(DEVICE),
            )
            preds_fov.append(out.item())

    preds_fov = np.array(preds_fov)
    r_fov = pearson_r(preds_fov, gt)
    print(f"      Pearson r = {r_fov:.4f}")
    results["UnifiedFoveatedCNN"] = preds_fov

    correlations = {
        "GLM": r_glm, "ODE": r_ode,
        "DriftCNN": r_cnn, "UnifiedFoveatedCNN": r_fov,
    }
    return gt, results, correlations


# ===================================================================
# Section 7 — 4-Panel Comparison Plot
# ===================================================================
def plot_comparison(gt, results, correlations):
    _banner("Section 7 — 4-Panel Comparison Plot")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(
        f"All Models — Ground Truth vs Prediction  "
        f"(Y[{TEST_START}:{TEST_START + NUM_SAMPLES}], contiguous)",
        fontsize=13, fontweight="bold",
    )

    panels = [
        ("GLM",                 "tab:orange"),
        ("ODE",                 "tab:green"),
        ("DriftCNN",            "tab:blue"),
        ("UnifiedFoveatedCNN",  "tab:red"),
    ]

    for idx, (name, color) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        ax.plot(gt, color="black", linewidth=0.8, alpha=0.7, label="Ground Truth")
        ax.plot(results[name], color=color, linewidth=1.0, alpha=0.85,
                label=f"{name}")
        r = correlations[name]
        ax.set_title(f"{name}  |  Pearson r = {r:.4f}", fontsize=11)
        ax.set_ylabel("Firing Rate")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[1, 0].set_xlabel("Sample (contiguous)")
    axes[1, 1].set_xlabel("Sample (contiguous)")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "all_models_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ===================================================================
# Section 8 — Synchronized Stimulus Video
# ===================================================================
def generate_sync_video(X_raw):
    _banner("Section 8 — Synchronized Stimulus Video")

    crop_half = 25
    cx = min(max(CENTER_X, crop_half), X_raw.shape[2] - crop_half)
    cy = min(max(CENTER_Y, crop_half), X_raw.shape[1] - crop_half)

    frames = X_raw[
        TEST_START : TEST_START + NUM_SAMPLES,
        cy - crop_half : cy + crop_half,
        cx - crop_half : cx + crop_half,
    ]

    print(f"  Raw frame indices: X[{TEST_START}:{TEST_START + NUM_SAMPLES}]")
    print(f"  Crop: 50x50 centred on ({cx}, {cy})")
    print(f"  Frames shape (before uint8): {frames.shape}")

    lo, hi = float(frames.min()), float(frames.max())
    if hi - lo < 1e-8:
        print("  WARNING: near-constant pixel values — video may appear blank")
        frames_u8 = np.zeros(frames.shape, dtype=np.uint8)
    else:
        frames_u8 = ((frames - lo) / (hi - lo) * 255).astype(np.uint8)

    print(f"  uint8 range: [{frames_u8.min()}, {frames_u8.max()}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    mp4_path = os.path.join(RESULTS_DIR, "test_stimulus_sync.mp4")

    writer = imageio.get_writer(
        mp4_path, format="FFMPEG", fps=30, macro_block_size=1, codec="libx264",
    )
    for frame in frames_u8:
        writer.append_data(frame)
    writer.close()

    print(f"  Saved: {mp4_path}  ({NUM_SAMPLES} frames @ 30 FPS)")


# ===================================================================
# Main Orchestrator
# ===================================================================
def main():
    _banner("MASTER PIPELINE — 24-Hour Training + Evaluation")
    t_total = time.time()

    # ---- wipe stale final checkpoints (preserve in-progress *_checkpoint.pth) ----
    import glob as _glob
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    stale = []
    for ext in ("*_final.pth", "*.json", "*.joblib"):
        stale.extend(_glob.glob(os.path.join(CHECKPOINTS_DIR, ext)))
    for fp in stale:
        os.remove(fp)
    if stale:
        print(f"  Cleared {len(stale)} old final checkpoint(s).")
    else:
        print("  Checkpoints directory is clean — nothing to remove.")
    preserved = _glob.glob(os.path.join(CHECKPOINTS_DIR, "*_checkpoint.pth"))
    if preserved:
        print(f"  Preserved {len(preserved)} in-progress checkpoint(s) for resume.")

    print(f"  Device:        {DEVICE}")
    print(f"  Test window:   Y[{TEST_START}:{TEST_START + NUM_SAMPLES}] "
          f"({NUM_SAMPLES} contiguous samples)")
    print(f"  Data file:     {SPIKES_PATH}")
    print(f"  Checkpoints:   {CHECKPOINTS_DIR}")
    print(f"  Results:       {RESULTS_DIR}")

    # ---- shared raw data (for GLM, ODE, and video) ----
    X_raw, Y = load_raw_data()

    # ---- sequential training ----
    glm_pipeline, glm_crop = train_glm(X_raw, Y)
    ode_params, ode_stim = train_ode(X_raw, Y)
    drift_dataset = train_driftcnn()
    foveated_dataset = train_unified_foveated()

    # ---- evaluation + outputs ----
    gt, results, correlations = evaluate_all(
        X_raw, Y, glm_pipeline, glm_crop,
        ode_params, ode_stim,
        drift_dataset, foveated_dataset,
    )

    plot_comparison(gt, results, correlations)
    generate_sync_video(X_raw)

    # ---- summary ----
    _banner("PIPELINE COMPLETE")
    print(f"  {'Model':<30s}  Pearson r")
    print(f"  {'-' * 45}")
    for name, r in correlations.items():
        print(f"  {name:<30s}  {r:.4f}")

    print(f"\n  Output files in {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        sz = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1024
        print(f"    {f:45s} {sz:8.1f} KB")

    print(f"\n  Total pipeline time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
