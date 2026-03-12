import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

from foveated_drift_dataset import FoveatedDriftDataset

IMAGES_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/natural_scenes.h5"
SPIKES_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/training_dataset_ns_full.h5"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 120

# ---------------------------------------------------------------------------
# Baseline DriftCNN (imported here for FLOPS comparison only)
# ---------------------------------------------------------------------------
class DriftCNN(nn.Module):
    def __init__(self, history_size=40):
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
# Foveated two-stream architecture
# ---------------------------------------------------------------------------
class FoveatedDriftCNN(nn.Module):
    def __init__(self, history_size=40):
        super().__init__()

        # Fovea stream — high-res center (history_size, 20, 20)
        self.fovea_stream = nn.Sequential(
            nn.Conv2d(history_size, 16, kernel_size=9),   # -> 16 x 12 x 12
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5),              # -> 8 x 8 x 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.fovea_flat = 8 * 8 * 8  # 512

        # Peripheral stream — downsampled full-field (history_size, 25, 25)
        self.peripheral_stream = nn.Sequential(
            nn.Conv2d(history_size, 8, kernel_size=9),    # -> 8 x 17 x 17
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=7),               # -> 4 x 11 x 11
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.periph_flat = 4 * 11 * 11  # 484

        self.regressor = nn.Sequential(
            nn.Linear(self.fovea_flat + self.periph_flat, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, fovea, peripheral):
        f = self.fovea_stream(fovea)
        f = f.view(f.size(0), -1)

        p = self.peripheral_stream(peripheral)
        p = p.view(p.size(0), -1)

        combined = torch.cat([f, p], dim=1)
        return self.regressor(combined)


# ---------------------------------------------------------------------------
# FLOPS / MACs comparison
# ---------------------------------------------------------------------------
def compare_flops(history_size=40):
    """Print a MACs comparison table for Baseline vs Foveated models."""
    try:
        from thop import profile
    except ImportError:
        print("[!] `thop` not installed — falling back to manual estimates.")
        print_manual_flops()
        return

    baseline = DriftCNN(history_size)
    foveated = FoveatedDriftCNN(history_size)

    baseline_input = torch.randn(1, history_size, 50, 50)
    fovea_input = torch.randn(1, history_size, 20, 20)
    periph_input = torch.randn(1, history_size, 25, 25)

    baseline_macs, baseline_params = profile(baseline, inputs=(baseline_input,), verbose=False)
    foveated_macs, foveated_params = profile(foveated, inputs=(fovea_input, periph_input), verbose=False)

    print("\n" + "=" * 60)
    print("  FLOPS / MACs COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<25} {'MACs':>15} {'Params':>12}")
    print("-" * 60)
    print(f"  {'Baseline DriftCNN':<25} {baseline_macs:>15,.0f} {baseline_params:>12,.0f}")
    print(f"  {'Foveated DriftCNN':<25} {foveated_macs:>15,.0f} {foveated_params:>12,.0f}")
    print("-" * 60)

    if foveated_macs > 0:
        reduction = baseline_macs / foveated_macs
        savings = (1 - foveated_macs / baseline_macs) * 100
        print(f"  Reduction factor: {reduction:.1f}x  ({savings:.1f}% fewer MACs)")
    print("=" * 60 + "\n")


def print_manual_flops():
    """Fallback manual MAC estimates when thop is unavailable."""
    b_conv1 = 16 * (40 * 15 * 15) * 36 * 36
    b_conv2 = 8 * (16 * 9 * 9) * 28 * 28
    b_linear = 8 * 28 * 28
    baseline_total = b_conv1 + b_conv2 + b_linear

    f_conv1 = 16 * (40 * 9 * 9) * 12 * 12
    f_conv2 = 8 * (16 * 5 * 5) * 8 * 8
    p_conv1 = 8 * (40 * 9 * 9) * 17 * 17
    p_conv2 = 4 * (8 * 7 * 7) * 11 * 11
    fov_linear = (512 + 484) * 128 + 128
    foveated_total = f_conv1 + f_conv2 + p_conv1 + p_conv2 + fov_linear

    print("\n" + "=" * 60)
    print("  FLOPS / MACs COMPARISON  (manual estimate)")
    print("=" * 60)
    print(f"  Baseline DriftCNN:  {baseline_total:>15,} MACs")
    print(f"  Foveated DriftCNN:  {foveated_total:>15,} MACs")
    reduction = baseline_total / foveated_total
    savings = (1 - foveated_total / baseline_total) * 100
    print(f"  Reduction: {reduction:.1f}x  ({savings:.1f}% fewer MACs)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for fovea, peripheral, targets in loader:
            fovea = fovea.to(device)
            peripheral = peripheral.to(device)
            outputs = model(fovea, peripheral)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    if np.std(all_preds) < 1e-9:
        return 0.0

    return np.corrcoef(all_preds, all_targets)[0, 1]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    print(f"Initializing Foveated Drift Experiment on {DEVICE}")

    # -- FLOPS comparison before training --
    compare_flops()

    # 1. Prepare Data
    try:
        full_dataset = FoveatedDriftDataset(IMAGES_PATH, SPIKES_PATH)

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Data Split: {len(train_ds)} training / {len(test_ds)} test samples.")

    except Exception as e:
        print(f"Data Error: {e}")
        return

    # 2. Setup Model
    model = FoveatedDriftCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.PoissonNLLLoss(log_input=False)

    train_loss_history = []
    test_corr_history = []

    # 3. Train
    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, (fovea, peripheral, targets) in enumerate(train_loader):
            fovea = fovea.to(DEVICE)
            peripheral = peripheral.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(fovea, peripheral)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}: Loss {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        current_correlation = evaluate_model(model, test_loader, DEVICE)

        train_loss_history.append(avg_train_loss)
        test_corr_history.append(current_correlation)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Test Correlation: {current_correlation:.4f} | "
              f"LR: {current_lr:.6f}")

    # 4. Save & Plot
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/foveated_drift_model_final.pth")
    print("Model saved to checkpoints/foveated_drift_model_final.pth")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(train_loss_history, color="tab:blue", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Correlation (Pearson)", color="tab:orange")
    ax2.plot(test_corr_history, color="tab:orange", label="Test Correlation")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Foveated Drift CNN: Loss vs Correlation")
    fig.tight_layout()
    plt.savefig("checkpoints/foveated_training_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
