"""
Path C — DriftCNN v3-Final Training
====================================
Dynamic architecture with no hardcoded dims, advanced regularisation
(GaussianNoise, Dropout2d, activity reg, grad clipping), OneCycleLR,
and early stopping on validation Pearson correlation.

Uses DriftSimulationDataset (merged from Path D).
"""

import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "Path_D_Drift_CNN"))
from drift_dataset import DriftSimulationDataset  # noqa: E402

_DATA_DIR = os.path.join(
    os.environ.get("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i")),
    "processed_data",
)

IMAGES_PATH = os.path.join(_DATA_DIR, "natural_scenes.h5")
SPIKES_PATH = os.path.join(_DATA_DIR, "training_dataset_ns_full.h5")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
HISTORY_SIZE = 40
BATCH_SIZE = 256
EPOCHS = 120
EARLY_STOP_PATIENCE = 10
ACTIVITY_LAMBDA = 0.01
GAUSSIAN_NOISE_STD = 0.05


# ---------------------------------------------------------------------------
# Shared utility layer
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
# DriftCNN v3-Final — Dynamic dims, GaussianNoise, Dropout2d, Kaiming
# ---------------------------------------------------------------------------
class DriftCNN(nn.Module):
    def __init__(self, history_size=40, crop_size=50,
                 dropout_conv=0.2, dropout_fc=0.5, noise_std=0.05):
        super().__init__()

        self.noise = GaussianNoise(std=noise_std)

        self.features = nn.Sequential(
            nn.Conv2d(history_size, 16, kernel_size=15),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout2d(dropout_conv),
            nn.Conv2d(16, 8, kernel_size=9),
            nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout2d(dropout_conv),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, history_size, crop_size, crop_size)
            self.flat_dim = self.features(dummy).numel()

        self.regressor = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(self.flat_dim, 1),
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
        print("  DriftCNN v3-Final — Debug Summary")
        print(f"{'='*60}")
        print(f"  Total params:     {total:>10,}")
        print(f"  Trainable params: {trainable:>10,}")
        print(f"  Flat dim:         {self.flat_dim}")
        print(f"{'-'*60}")
        for name, p in self.named_parameters():
            print(f"  {name:45s} {str(list(p.shape)):20s} {p.numel():>8,}")
        print(f"{'='*60}\n")

    def forward(self, x):
        x = self.noise(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    if np.std(all_preds) < 1e-9:
        return 0.0
    r = np.corrcoef(all_preds, all_targets)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Training loop — v3-Final
# ---------------------------------------------------------------------------
def main():
    print(f"Initializing DriftCNN v3-Final on {DEVICE}")
    print(f"   HISTORY={HISTORY_SIZE}  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}  "
          f"PATIENCE={EARLY_STOP_PATIENCE}")
    print(f"   ACTIVITY_LAMBDA={ACTIVITY_LAMBDA}  NOISE_STD={GAUSSIAN_NOISE_STD}")

    full_dataset = DriftSimulationDataset(IMAGES_PATH, SPIKES_PATH,
                                          history_size=HISTORY_SIZE)

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

    model = DriftCNN(
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

    checkpoints_dir = os.path.join(_PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_val_corr = -float("inf")
    best_state = None
    patience_counter = 0
    train_loss_history = []
    val_corr_history = []

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_activity_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

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
            }, os.path.join(checkpoints_dir, "drift_checkpoint.pth"))
            print(f"  Checkpoint saved at epoch {epoch+1}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    final_state = best_state if best_state is not None else model.state_dict()
    final_path = os.path.join(checkpoints_dir, "drift_model_final.pth")
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
    plt.title("DriftCNN v3-Final: Loss vs Correlation")
    fig.tight_layout()
    curve_path = os.path.join(checkpoints_dir, "drift_training_curve.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Training curve saved to {curve_path}")


if __name__ == "__main__":
    main()
