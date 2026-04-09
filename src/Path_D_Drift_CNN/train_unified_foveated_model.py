"""
Unified Foveated CNN v3-Final — High-Generalisation Architecture
================================================================
Dynamic architecture with no hardcoded dims, advanced regularisation
(GaussianNoise, Dropout2d, activity reg, grad clipping), OneCycleLR,
and early stopping on validation Pearson correlation.
"""

import os
import copy
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DATA_DIR = os.path.join(
    os.environ.get("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i")),
    "processed_data",
)

IMAGES_PATH = os.path.join(_DATA_DIR, "natural_scenes.h5")
SPIKES_PATH = os.path.join(_DATA_DIR, "training_dataset_ns_full.h5")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
HISTORY_SIZE = 40
BATCH_SIZE = 64
EPOCHS = 120
EARLY_STOP_PATIENCE = 10
ACTIVITY_LAMBDA = 0.01
GAUSSIAN_NOISE_STD = 0.05


# ---------------------------------------------------------------------------
# Shared utility layers
# ---------------------------------------------------------------------------
class GaussianNoise(nn.Module):
    """Adds Gaussian noise during training only."""
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


# ---------------------------------------------------------------------------
# Unified Foveated Dataset v3 — 1-channel, no coordinate maps
# ---------------------------------------------------------------------------
class UnifiedFoveatedDataset(Dataset):
    """
    Two-stream foveated dataset with saliency-guided Focus of Attention.

    Returns per sample:
        fovea:      (history_size, 1, fovea_size, fovea_size)
        peripheral: (history_size, 1, crop_size, crop_size)
        target:     (1,)
    """

    def __init__(self, images_path, spikes_path, history_size=40, crop_size=50,
                 fovea_size=20, alpha=0.4, jitter_std=0.1, max_step=3.0,
                 saliency_window=5, shift_range=3, training=True):
        self.history_size = history_size
        self.crop_size = crop_size
        self.fovea_size = fovea_size
        self.alpha = alpha
        self.jitter_std = jitter_std
        self.max_step = max_step
        self.saliency_window = saliency_window
        self.shift_range = shift_range
        self.training = training

        self.half_fovea = fovea_size // 2
        self.half_crop = crop_size // 2

        self.foa_min = float(self.half_fovea)
        self.foa_max = float(crop_size - self.half_fovea)

        print(f"Loading Synced Data from: {os.path.basename(spikes_path)}")

        if not os.path.exists(spikes_path):
            raise FileNotFoundError(f"File not found: {spikes_path}")

        with h5py.File(spikes_path, 'r') as f:
            self.X = f['X'][:]
            self.Y = f['Y'][:]

        self.center_x = 27
        self.center_y = 47

        mean = np.mean(self.X)
        std = np.std(self.X)
        self.X = (self.X - mean) / (std + 1e-6)

        print(f"Dataset Loaded: {self.X.shape[0]} frames.")
        print(f"   -> v3-Final: 1-channel, history={history_size}")
        print(f"   -> fovea {fovea_size}x{fovea_size}, "
              f"peripheral {crop_size}x{crop_size}")

    def _compute_saliency(self, frame):
        f64 = frame.astype(np.float64)
        local_mean = uniform_filter(f64, size=self.saliency_window)
        local_sq_mean = uniform_filter(f64 ** 2, size=self.saliency_window)
        return np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    def _saliency_gradient(self, saliency, x, y):
        h, w = saliency.shape
        ix = int(np.clip(round(x), 0, w - 1))
        iy = int(np.clip(round(y), 0, h - 1))
        x_lo, x_hi = max(ix - 1, 0), min(ix + 1, w - 1)
        y_lo, y_hi = max(iy - 1, 0), min(iy + 1, h - 1)
        dx = (saliency[iy, x_hi] - saliency[iy, x_lo]) / max(x_hi - x_lo, 1)
        dy = (saliency[y_hi, ix] - saliency[y_lo, ix]) / max(y_hi - y_lo, 1)
        return dx, dy

    def __len__(self):
        return len(self.Y) - self.history_size

    def __getitem__(self, idx):
        raw_clip = self.X[idx: idx + self.history_size]
        H, W = raw_clip.shape[1], raw_clip.shape[2]

        if self.training:
            dx = np.random.randint(-self.shift_range, self.shift_range + 1)
            dy = np.random.randint(-self.shift_range, self.shift_range + 1)
        else:
            dx, dy = 0, 0

        cx = max(self.half_crop, min(W - self.half_crop, self.center_x + dx))
        cy = max(self.half_crop, min(H - self.half_crop, self.center_y + dy))

        crops = raw_clip[:,
                         cy - self.half_crop: cy + self.half_crop,
                         cx - self.half_crop: cx + self.half_crop]

        foa_x = float(self.crop_size) / 2.0
        foa_y = float(self.crop_size) / 2.0

        fovea_frames = []
        periph_frames = []

        for t in range(self.history_size):
            frame = crops[t]
            saliency = self._compute_saliency(frame)
            grad_x, grad_y = self._saliency_gradient(saliency, foa_x, foa_y)

            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
            grad_x /= grad_mag
            grad_y /= grad_mag

            step_x = self.alpha * grad_x + np.random.randn() * self.jitter_std
            step_y = self.alpha * grad_y + np.random.randn() * self.jitter_std
            step_mag = np.sqrt(step_x ** 2 + step_y ** 2)
            if step_mag > self.max_step:
                scale = self.max_step / step_mag
                step_x *= scale
                step_y *= scale

            foa_x = np.clip(foa_x + step_x, self.foa_min, self.foa_max)
            foa_y = np.clip(foa_y + step_y, self.foa_min, self.foa_max)

            fx = int(np.clip(round(foa_x), self.half_fovea,
                             self.crop_size - self.half_fovea))
            fy = int(np.clip(round(foa_y), self.half_fovea,
                             self.crop_size - self.half_fovea))

            fovea_img = frame[fy - self.half_fovea: fy + self.half_fovea,
                              fx - self.half_fovea: fx + self.half_fovea]
            fovea_frames.append(fovea_img[np.newaxis, :, :])
            periph_frames.append(frame[np.newaxis, :, :])

        fovea = np.array(fovea_frames, dtype=np.float32)
        peripheral = np.array(periph_frames, dtype=np.float32)
        y_val = self.Y[idx + self.history_size - 1]

        return (torch.FloatTensor(fovea),
                torch.FloatTensor(peripheral),
                torch.FloatTensor([y_val]))


# ---------------------------------------------------------------------------
# Legacy: Baseline DriftCNN (for MACs comparison only)
# ---------------------------------------------------------------------------
class DriftCNN(nn.Module):
    def __init__(self, history_size=40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(history_size, 16, kernel_size=15),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=9),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.flat_dim = 8 * 28 * 28
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# ---------------------------------------------------------------------------
# Legacy: Previous FoveatedDriftCNN (for MACs comparison only)
# ---------------------------------------------------------------------------
class FoveatedDriftCNN(nn.Module):
    def __init__(self, history_size=40):
        super().__init__()
        self.fovea_stream = nn.Sequential(
            nn.Conv2d(history_size, 16, kernel_size=9),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.fovea_flat = 8 * 8 * 8
        self.peripheral_stream = nn.Sequential(
            nn.Conv2d(history_size, 8, kernel_size=9),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=7),
            nn.BatchNorm2d(4), nn.ReLU(),
        )
        self.periph_flat = 4 * 11 * 11
        self.regressor = nn.Sequential(
            nn.Linear(self.fovea_flat + self.periph_flat, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Softplus(),
        )

    def forward(self, fovea, peripheral):
        f = self.fovea_stream(fovea).view(fovea.size(0), -1)
        p = self.peripheral_stream(peripheral).view(peripheral.size(0), -1)
        return self.regressor(torch.cat([f, p], dim=1))


# ---------------------------------------------------------------------------
# Unified Foveated CNN v3-Final
# ---------------------------------------------------------------------------
class UnifiedFoveatedCNN(nn.Module):
    """
    v3-Final: Dynamic dims, GaussianNoise, Dropout2d, AdaptiveAvgPool,
    Kaiming init, ~80-100k params.

    Pipeline:
      GaussianNoise -> fovea Conv -> BN -> ReLU -> Dropout2d -> Conv -> BN
        -> ReLU -> Dropout2d -> AdaptiveAvgPool2d(4) -> flatten
      GaussianNoise -> periph dilated Conv -> BN -> ReLU -> Dropout2d -> Conv
        -> BN -> ReLU -> Dropout2d -> AdaptiveAvgPool2d(5) -> flatten
      Concat -> (B, T, combined_dim)
      GRU(combined_dim, gru_hidden) -> last hidden -> Dropout
      Linear -> ReLU -> Dropout -> Linear -> Softplus
    """

    def __init__(self, history_size=40, gru_hidden=64,
                 dropout_conv=0.2, dropout_fc=0.5,
                 noise_std=0.05, fovea_size=20, periph_size=50):
        super().__init__()
        self.history_size = history_size

        self.fovea_noise = GaussianNoise(std=noise_std)
        self.fovea_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=9),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout2d(dropout_conv),
            nn.Conv2d(16, 8, kernel_size=5),
            nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout2d(dropout_conv),
            nn.AdaptiveAvgPool2d(4),
        )

        self.periph_noise = GaussianNoise(std=noise_std)
        self.peripheral_stream = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, dilation=4, padding=0),
            nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout2d(dropout_conv),
            nn.Conv2d(8, 4, kernel_size=3, dilation=2, padding=0),
            nn.BatchNorm2d(4), nn.ReLU(), nn.Dropout2d(dropout_conv),
            nn.AdaptiveAvgPool2d(5),
        )

        with torch.no_grad():
            fov_dummy = torch.zeros(1, 1, fovea_size, fovea_size)
            self.fovea_flat = self.fovea_stream(fov_dummy).numel()
            per_dummy = torch.zeros(1, 1, periph_size, periph_size)
            self.periph_flat = self.peripheral_stream(per_dummy).numel()

        combined_dim = self.fovea_flat + self.periph_flat

        self.gru = nn.GRU(
            input_size=combined_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.drop_post_gru = nn.Dropout(dropout_fc)

        self.regressor = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.regressor[-2].bias.data.fill_(0.01)

    def print_debug_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print("  UnifiedFoveatedCNN v3-Final — Debug Summary")
        print(f"{'='*60}")
        print(f"  Total params:     {total:>10,}")
        print(f"  Trainable params: {trainable:>10,}")
        print(f"  Fovea flat dim:   {self.fovea_flat}")
        print(f"  Periph flat dim:  {self.periph_flat}")
        print(f"  GRU input dim:    {self.fovea_flat + self.periph_flat}")
        print(f"  GRU hidden:       {self.gru.hidden_size}")
        print(f"{'-'*60}")
        for name, p in self.named_parameters():
            print(f"  {name:45s} {str(list(p.shape)):20s} {p.numel():>8,}")
        print(f"{'='*60}\n")

    def forward(self, fovea, peripheral):
        B, T, C, Hf, Wf = fovea.shape
        _, _, _, Hp, Wp = peripheral.shape

        fov = fovea.reshape(B * T, C, Hf, Wf)
        per = peripheral.reshape(B * T, C, Hp, Wp)

        fov = self.fovea_noise(fov)
        per = self.periph_noise(per)

        fov = self.fovea_stream(fov).reshape(B * T, -1)
        per = self.peripheral_stream(per).reshape(B * T, -1)

        combined = torch.cat([fov, per], dim=1).reshape(B, T, -1)

        _, h_n = self.gru(combined)
        features = self.drop_post_gru(h_n.squeeze(0))

        return self.regressor(features)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for fovea, peripheral, targets in loader:
            fovea = fovea.to(device)
            peripheral = peripheral.to(device)
            outputs = model(fovea, peripheral)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    if np.std(all_preds) < 1e-9:
        return 0.0
    r = np.corrcoef(all_preds, all_targets)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Main training loop — v3-Final
# ---------------------------------------------------------------------------
def main():
    print(f"Initializing Unified Foveated CNN v3-Final on {DEVICE}")
    print(f"   HISTORY={HISTORY_SIZE}  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}  "
          f"PATIENCE={EARLY_STOP_PATIENCE}")
    print(f"   ACTIVITY_LAMBDA={ACTIVITY_LAMBDA}  NOISE_STD={GAUSSIAN_NOISE_STD}")

    full_dataset = UnifiedFoveatedDataset(
        IMAGES_PATH, SPIKES_PATH,
        history_size=HISTORY_SIZE,
        training=True,
    )

    n = len(full_dataset)
    train_end = int(0.8 * n)
    train_ds = Subset(full_dataset, list(range(train_end)))
    val_ds = Subset(full_dataset, list(range(train_end, n)))

    use_cuda = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and True,
    )

    print(f"Chronological split: {len(train_ds)} train / {len(val_ds)} val "
          f"(contiguous block, indices {train_end}..{n-1})")

    model = UnifiedFoveatedCNN(
        history_size=HISTORY_SIZE,
        noise_std=GAUSSIAN_NOISE_STD,
    ).to(DEVICE)
    model.print_debug_summary()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )
    criterion = nn.PoissonNLLLoss(log_input=False)

    os.makedirs(os.path.join(_PROJECT_ROOT, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(
        _PROJECT_ROOT, "checkpoints", "unified_foveated_checkpoint.pth",
    )

    best_val_corr = -float("inf")
    best_state = None
    patience_counter = 0
    train_loss_history = []
    val_corr_history = []

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        full_dataset.training = True
        model.train()
        epoch_loss = 0.0
        epoch_activity_loss = 0.0

        for batch_idx, (fovea, peripheral, targets) in enumerate(train_loader):
            fovea = fovea.to(DEVICE)
            peripheral = peripheral.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(fovea, peripheral)

            poisson_loss = criterion(outputs, targets)
            activity_loss = ACTIVITY_LAMBDA * torch.abs(
                outputs.mean() - targets.mean()
            )
            loss = poisson_loss + activity_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_activity_loss += activity_loss.item()

            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}: Loss {loss.item():.4f} "
                      f"(activity: {activity_loss.item():.5f})")

        n_batches = len(train_loader)
        avg_train_loss = epoch_loss / n_batches
        avg_activity = epoch_activity_loss / n_batches

        full_dataset.training = False
        val_corr = evaluate_model(model, val_loader, DEVICE)

        train_loss_history.append(avg_train_loss)
        val_corr_history.append(val_corr)

        current_lr = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = " *best*"
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Activity: {avg_activity:.5f} | "
              f"Val Corr: {val_corr:.4f}{improved} | "
              f"LR: {current_lr:.6f} | "
              f"Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_corr": best_val_corr,
                "best_state": best_state,
                "train_loss_history": train_loss_history,
                "val_corr_history": val_corr_history,
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    final_state = best_state if best_state is not None else model.state_dict()
    final_path = os.path.join(
        _PROJECT_ROOT, "checkpoints", "unified_foveated_model_final.pth",
    )
    torch.save(final_state, final_path)
    print(f"Best model saved to {final_path}  (val corr = {best_val_corr:.4f})")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(train_loss_history, color="tab:blue", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Correlation (Pearson)", color="tab:orange")
    ax2.plot(val_corr_history, color="tab:orange", label="Val Correlation")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("Unified Foveated CNN v3-Final: Loss vs Correlation")
    fig.tight_layout()
    curve_path = os.path.join(
        _PROJECT_ROOT, "checkpoints", "unified_foveated_training_curve.png",
    )
    plt.savefig(curve_path, dpi=150)
    print(f"Training curve saved to {curve_path}")


if __name__ == "__main__":
    main()
