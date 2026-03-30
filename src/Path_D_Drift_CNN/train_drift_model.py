import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

from drift_dataset import DriftSimulationDataset

IMAGES_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/natural_scenes.h5"
SPIKES_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/training_dataset_ns_full.h5"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 100

# --- ARCHITECTURE ---
class DriftCNN(nn.Module):
    def __init__(self, history_size=40):
        super(DriftCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(history_size, 16, kernel_size=15),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=9),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.flat_dim = 8 * 28 * 28
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# --- HELPER: CALCULATE CORRELATION ---
def evaluate_model(model, loader, device):
    model.eval() 
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    if np.std(all_preds) < 1e-9: 
        return 0.0
    
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    return correlation

# --- TRAINING LOOP ---
def main():
    print(f"🚀 Initializing Drift Experiment on {DEVICE}")
    
    try:
        full_dataset = DriftSimulationDataset(IMAGES_PATH, SPIKES_PATH)
        
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"✅ Data Split: {len(train_ds)} Training samples, {len(test_ds)} Test samples.")
        
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    model = DriftCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    train_loss_history = []
    test_corr_history = []

    print("💪 Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        model.train() 
        epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
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
        
        print(f"🏁 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | 🏆 Test Correlation: {current_correlation:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/drift_model_final.pth")
    print("💾 Model saved.")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_loss_history, color='tab:blue', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Correlation (Pearson)', color='tab:orange')  
    ax2.plot(test_corr_history, color='tab:orange', label='Test Correlation')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Drift CNN: Loss vs Correlation")
    fig.tight_layout()  
    plt.show()

if __name__ == "__main__":
    main()
    
