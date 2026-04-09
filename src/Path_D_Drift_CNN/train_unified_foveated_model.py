import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
HISTORY_SIZE = 60
BATCH_SIZE = 64
EPOCHS = 120


# ---------------------------------------------------------------------------
# Unified Foveated Dataset — Saliency-Guided FOA (Gravitational Attention)
# ---------------------------------------------------------------------------
class UnifiedFoveatedDataset(Dataset):
    """
    Two-stream foveated dataset with saliency-guided Focus of Attention
    and early-fusion FOA coordinate maps.

    Returns per sample:
        fovea:      (history_size, 3, 20, 20) — [image, x_coord, y_coord]
        peripheral: (history_size, 3, 50, 50) — [image, x_coord, y_coord]
        foa_coords: (2,)                      — normalized FOA position (kept for compat)
        target:     (1,)                      — spike count
    """

    def __init__(self, images_path, spikes_path, history_size=60, crop_size=50,
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
        print(f"   -> Unified Foveated mode: fovea {fovea_size}x{fovea_size}, "
              f"peripheral {crop_size}x{crop_size} (full-res, dilated)")
        print(f"   -> Saliency-guided FOA (alpha={alpha}, jitter={jitter_std}, "
              f"max_step={max_step}px)")
        print(f"   -> Spatial augmentation: shift_range=±{shift_range}px "
              f"(training={training})")

    def _compute_saliency(self, frame):
        """Local variance via sliding window — contrast/saliency proxy."""
        f64 = frame.astype(np.float64)
        local_mean = uniform_filter(f64, size=self.saliency_window)
        local_sq_mean = uniform_filter(f64 ** 2, size=self.saliency_window)
        return np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    def _saliency_gradient(self, saliency, x, y):
        """Finite-difference gradient of saliency at integer position (x, y)."""
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

        fovea_grid_y = np.linspace(-1, 1, self.fovea_size, dtype=np.float32)
        fovea_grid_x = np.linspace(-1, 1, self.fovea_size, dtype=np.float32)
        fovea_ymap, fovea_xmap = np.meshgrid(fovea_grid_y, fovea_grid_x, indexing='ij')

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
            fovea_3ch = np.stack([fovea_img, fovea_xmap, fovea_ymap], axis=0)
            fovea_frames.append(fovea_3ch)

            py_grid = np.linspace(0, self.crop_size - 1, self.crop_size, dtype=np.float32)
            px_grid = np.linspace(0, self.crop_size - 1, self.crop_size, dtype=np.float32)
            py_map, px_map = np.meshgrid(py_grid, px_grid, indexing='ij')
            px_map = (px_map - foa_x) / (self.crop_size / 2.0)
            py_map = (py_map - foa_y) / (self.crop_size / 2.0)

            periph_3ch = np.stack([frame, px_map, py_map], axis=0)
            periph_frames.append(periph_3ch)

        fovea = np.array(fovea_frames, dtype=np.float32)
        peripheral = np.array(periph_frames, dtype=np.float32)

        norm_x = (foa_x - self.crop_size / 2.0) / (self.crop_size / 2.0)
        norm_y = (foa_y - self.crop_size / 2.0) / (self.crop_size / 2.0)
        foa_coords = np.array([norm_x, norm_y], dtype=np.float32)

        y_val = self.Y[idx + self.history_size - 1]

        return (torch.FloatTensor(fovea),
                torch.FloatTensor(peripheral),
                torch.FloatTensor(foa_coords),
                torch.FloatTensor([y_val]))


# ---------------------------------------------------------------------------
# Baseline DriftCNN (for MACs comparison only)
# ---------------------------------------------------------------------------
class DriftCNN(nn.Module):
    def __init__(self, history_size=60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(history_size, 16, kernel_size=15),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=9),
            nn.BatchNorm2d(8),
            nn.ReLU(),
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
# Previous FoveatedDriftCNN (for MACs comparison only)
# ---------------------------------------------------------------------------
class FoveatedDriftCNN(nn.Module):
    def __init__(self, history_size=60):
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
# Causal Conv1d block — strict causal padding (no future leakage)
# ---------------------------------------------------------------------------
class CausalConv1dBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, groups=4):
        super().__init__()
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0, groups=groups)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))


# ---------------------------------------------------------------------------
# Unified Foveated CNN v2 — (2+1)D + GRU + Causal Attention + Early FOA
# ---------------------------------------------------------------------------
class UnifiedFoveatedCNN(nn.Module):
    """
    Architecture v2 with four components:

    1. Per-frame dual-stream CNN (TimeDistributed):
       - Fovea stream:  3-ch 20x20 -> Conv layers -> 512-d per frame
       - Peripheral stream: 3-ch 50x50 -> dilated Conv + pool -> 100-d per frame
       Input channels: [image, x_coord_map, y_coord_map] (early-fusion FOA).
       Produces a (B, T, 612) feature sequence.

    2. Causal temporal Conv1d:
       - Two CausalConv1dBlock layers (k=3, grouped) on the (B, T, 612) sequence.
       - Frame t only depends on frames t, t-1, t-2 (no future leakage).

    3. GRU + Causal Self-Attention:
       - GRU (input=612, hidden=128, 1 layer).
       - Multi-head self-attention (4 heads) with causal mask.
       - Residual connection + LayerNorm.
       - Last timestep -> (B, 128).

    4. Regressor head: Linear -> Softplus.
    """

    def __init__(self, history_size=60):
        super().__init__()
        self.history_size = history_size

        self.fovea_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.fovea_flat = 8 * 8 * 8   # 512

        self.peripheral_stream = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, dilation=4, padding=0),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, dilation=2, padding=0),
            nn.BatchNorm2d(4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),
        )
        self.periph_flat = 4 * 5 * 5  # 100

        combined_dim = self.fovea_flat + self.periph_flat  # 612

        self.temporal_conv = nn.Sequential(
            CausalConv1dBlock(combined_dim, kernel_size=3, groups=4),
            CausalConv1dBlock(combined_dim, kernel_size=3, groups=4),
        )

        self.gru = nn.GRU(
            input_size=combined_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(128)

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, fovea, peripheral, foa_coords):
        B, T, C, Hf, Wf = fovea.shape
        _, _, _, Hp, Wp = peripheral.shape

        fov = fovea.reshape(B * T, C, Hf, Wf)
        per = peripheral.reshape(B * T, C, Hp, Wp)

        fov = self.fovea_stream(fov).reshape(B * T, -1)
        per = self.peripheral_stream(per).reshape(B * T, -1)

        combined = torch.cat([fov, per], dim=1)
        combined = combined.reshape(B, T, -1)

        combined = combined.permute(0, 2, 1)
        combined = self.temporal_conv(combined)
        combined = combined.permute(0, 2, 1)

        gru_out, _ = self.gru(combined)

        causal_mask = torch.triu(
            torch.ones(T, T, device=gru_out.device), diagonal=1
        ).bool()

        attn_out, _ = self.attention(
            gru_out, gru_out, gru_out,
            attn_mask=causal_mask,
        )
        attn_out = self.attn_norm(attn_out + gru_out)
        features = attn_out[:, -1, :]

        return self.regressor(features)


# ---------------------------------------------------------------------------
# MACs / FLOPS comparison — three-way table
# ---------------------------------------------------------------------------
def compare_flops(history_size=60):
    """Print MACs comparison: Baseline vs Previous Foveated vs Unified."""
    try:
        from thop import profile
    except ImportError:
        print("[!] `thop` not installed — skipping MACs comparison.")
        return

    baseline = DriftCNN(history_size)
    prev_fov = FoveatedDriftCNN(history_size)
    unified = UnifiedFoveatedCNN(history_size)

    b_in = (torch.randn(1, history_size, 50, 50),)
    pf_in = (torch.randn(1, history_size, 20, 20),
             torch.randn(1, history_size, 25, 25))
    u_in = (torch.randn(1, history_size, 3, 20, 20),
            torch.randn(1, history_size, 3, 50, 50),
            torch.randn(1, 2))

    b_macs, b_params = profile(baseline, inputs=b_in, verbose=False)
    pf_macs, pf_params = profile(prev_fov, inputs=pf_in, verbose=False)

    try:
        u_macs, u_params = profile(unified, inputs=u_in, verbose=False)
    except Exception:
        u_macs, u_params = 0, sum(p.numel() for p in unified.parameters())

    print("\n" + "=" * 70)
    print("  FLOPS / MACs COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<30} {'MACs':>15} {'Params':>12}")
    print("-" * 70)
    print(f"  {'Baseline DriftCNN':<30} {b_macs:>15,.0f} {b_params:>12,.0f}")
    print(f"  {'Prev FoveatedDriftCNN':<30} {pf_macs:>15,.0f} {pf_params:>12,.0f}")
    print(f"  {'Unified CNN+LSTM':<30} {u_macs:>15,.0f} {u_params:>12,.0f}")
    print("-" * 70)

    if u_macs > 0 and b_macs > 0:
        savings_vs_base = (1 - u_macs / b_macs) * 100
        savings_vs_prev = (1 - u_macs / pf_macs) * 100 if pf_macs > 0 else 0
        print(f"  Unified vs Baseline: {b_macs / u_macs:.1f}x  "
              f"({savings_vs_base:.1f}% fewer MACs)")
        print(f"  Unified vs Previous: {pf_macs / u_macs:.1f}x  "
              f"({savings_vs_prev:+.1f}% MACs)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for fovea, peripheral, foa_coords, targets in loader:
            fovea = fovea.to(device)
            peripheral = peripheral.to(device)
            foa_coords = foa_coords.to(device)
            outputs = model(fovea, peripheral, foa_coords)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    if np.std(all_preds) < 1e-9:
        return 0.0

    return np.corrcoef(all_preds, all_targets)[0, 1]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    print(f"Initializing Unified Foveated CNN v2 Experiment on {DEVICE}")
    print(f"   HISTORY_SIZE={HISTORY_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")

    compare_flops(HISTORY_SIZE)

    try:
        full_dataset = UnifiedFoveatedDataset(
            IMAGES_PATH, SPIKES_PATH,
            history_size=HISTORY_SIZE,
            training=True,
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Data Split: {len(train_ds)} training / {len(test_ds)} test samples.")

    except Exception as e:
        print(f"Data Error: {e}")
        return

    model = UnifiedFoveatedCNN(history_size=HISTORY_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.PoissonNLLLoss(log_input=False)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/unified_foveated_checkpoint.pth"

    start_epoch = 0
    best_val_loss = float('inf')
    best_state = None
    train_loss_history = []
    test_corr_history = []

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_state = ckpt['best_state']
        train_loss_history = ckpt.get('train_loss_history', [])
        test_corr_history = ckpt.get('test_corr_history', [])
        print(f"  Resumed from epoch {start_epoch}, "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

    print("Starting Training Loop...")

    for epoch in range(start_epoch, EPOCHS):
        full_dataset.training = True
        model.train()
        epoch_loss = 0

        for batch_idx, (fovea, peripheral, foa_coords, targets) in enumerate(train_loader):
            fovea = fovea.to(DEVICE)
            peripheral = peripheral.to(DEVICE)
            foa_coords = foa_coords.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(fovea, peripheral, foa_coords)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}: Loss {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        full_dataset.training = False
        current_correlation = evaluate_model(model, test_loader, DEVICE)

        train_loss_history.append(avg_train_loss)
        test_corr_history.append(current_correlation)

        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            best_state = model.state_dict()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Test Correlation: {current_correlation:.4f} | "
              f"LR: {current_lr:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_state': best_state,
                'train_loss_history': train_loss_history,
                'test_corr_history': test_corr_history,
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, "checkpoints/unified_foveated_model_final.pth")
    print("Model saved to checkpoints/unified_foveated_model_final.pth")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(train_loss_history, color="tab:blue", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Correlation (Pearson)", color="tab:orange")
    ax2.plot(test_corr_history, color="tab:orange", label="Test Correlation")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Unified Foveated CNN v2: Loss vs Correlation")
    fig.tight_layout()
    plt.savefig("checkpoints/unified_foveated_training_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
