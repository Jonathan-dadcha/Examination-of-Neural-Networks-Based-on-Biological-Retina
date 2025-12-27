import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import BASE_PATH, SESSION

DATA_FILE = os.path.join(BASE_PATH, SESSION, 'processed_data', 'training_dataset_wn.h5')

# Hyperparameters
HISTORY_SIZE = 30   # How many frames back to look (e.g., 30 frames = 1 second)
BATCH_SIZE = 128    # How many samples to process at once
LEARNING_RATE = 0.001
EPOCHS = 20         # How many times to go over the data
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Use Mac GPU if available

print(f"🚀 Running on device: {DEVICE}")

# --- 2. DATASET CLASS ---
class RetinaDataset(Dataset):
    """
    Creates samples (X, Y) where:
    X = A clip of 'history_size' frames
    Y = The spike count at the end of that clip
    """
    def __init__(self, h5_file, history_size):
        self.h5_file = h5_file
        self.history_size = history_size
        
        with h5py.File(h5_file, 'r') as f:
            # Load full data into memory (it's small enough)
            self.X_full = f['X'][:]  # (TotalFrames, H, W)
            self.Y_full = f['Y'][:]  # (TotalFrames,)
            
        # Normalize X to be centered around 0 (helps learning)
        # White noise is 0/1, so mean is approx 0.5
        self.X_full = self.X_full - 0.5
        
        self.num_samples = self.X_full.shape[0] - history_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # We take a slice of frames [t-history : t]
        # Flatten input to 1D vector for Linear Model
        x_clip = self.X_full[idx : idx + self.history_size] # Shape: (History, H, W)
        y_val = self.Y_full[idx + self.history_size - 1]    # Shape: scalar
        
        # Flatten X for the linear layer
        return torch.FloatTensor(x_clip.flatten()), torch.FloatTensor([y_val])

# --- 3. MODEL DEFINITION (The "LN" Model) ---
class LNModel(nn.Module):
    def __init__(self, input_dim):
        super(LNModel, self).__init__()
        # Linear Stage (The "Receptive Field")
        self.linear = nn.Linear(input_dim, 1)
        
        # Batch Normalization to stabilize outputs before nonlinearity
        self.bn = nn.BatchNorm1d(1)

        # Nonlinear Stage (Softplus is like a smooth ReLU)
        # Ensures firing rate is always positive
        self.nonlinearity = nn.Softplus()

    def forward(self, x):
        linear_out = self.linear(x)
        # Add dimension for BN (Batch, 1) if output is (Batch, 1) or reshape if needed
        # nn.Linear output is (Batch, 1), BN expects (Batch, C) or (Batch, C, L)
        # For 1D input (Batch, Features), BN1d works on (Batch, Features)
        
        # Apply BN to keep variance non-zero
        normalized = self.bn(linear_out)
        
        firing_rate = self.nonlinearity(normalized)
        return firing_rate

# --- 4. MAIN TRAINING LOOP ---
def train_model():
    # Load Data
    if not os.path.exists(DATA_FILE):
        print("❌ Dataset not found.")
        return

    print("Loading Dataset...")
    dataset = RetinaDataset(DATA_FILE, HISTORY_SIZE)
    
    # Split Train/Test (80% / 20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate input dimension (History * H * W)
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    H, W = 100, 75 # We know this from previous steps
    
    print(f"Input Dimension: {input_dim} ({HISTORY_SIZE} frames x {H}x{W})")
    
    # Initialize Model
    model = LNModel(input_dim).to(DEVICE)
    
    # Loss Function: Poisson NLL (Standard for spike counts)
    criterion = nn.PoissonNLLLoss(log_input=False) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print("\n--- Starting Training ---")
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Poisson loss expects firing rate, targets are counts
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    print("✅ Training Complete.")

    # --- 5. EVALUATION & VISUALIZATION ---
    # Let's verify correlation on Test Set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    # Calculate Pearson Correlation with Epsilon to avoid NaN
    preds_flat = np.array(all_preds).flatten()
    targets_flat = np.array(all_targets).flatten()
    
    # Check for zero variance
    if np.std(preds_flat) == 0:
        print("⚠️ Warning: Model predictions have zero variance (constant output).")
    
    # Use robust correlation calculation
    # Adding a tiny epsilon to denominator is a common trick, but numpy handles it if we are careful
    try:
        pcc = np.corrcoef(preds_flat, targets_flat)[0, 1]
    except Exception as e:
        print(f"Correlation calculation failed: {e}")
        pcc = 0.0
        
    print(f"\n🏆 Test Set Pearson Correlation (Accuracy): {pcc:.4f}")

    # --- 6. PLOT THE LEARNED FILTERS ---
    # We extract the weights from the linear layer. 
    # This IS the Receptive Field the model learned!
    weights = model.linear.weight.data.cpu().numpy().flatten()
    
    # Reshape back to (Time, Height, Width)
    rf_movie = weights.reshape(HISTORY_SIZE, H, W)
    
    # Find the frame with strongest weights (peak response)
    peak_frame_idx = np.argmax(np.abs(rf_movie).sum(axis=(1, 2)))
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Spatial RF
    plt.subplot(1, 3, 1)
    plt.imshow(rf_movie[peak_frame_idx], cmap='RdBu_r')
    plt.title(f"Learned Spatial Filter\n(Lag: {peak_frame_idx} frames)")
    plt.colorbar()
    
    # Plot 2: Temporal Profile (center pixel)
    # Find center of RF
    cy, cx = np.unravel_index(np.argmax(np.abs(rf_movie[peak_frame_idx])), (H, W))
    plt.subplot(1, 3, 2)
    plt.plot(rf_movie[:, cy, cx], marker='o')
    plt.title(f"Learned Temporal Filter\n(at x={cx}, y={cy})")
    plt.xlabel("Time Lags")
    plt.grid(True)
    
    # Plot 3: Loss Curve
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()