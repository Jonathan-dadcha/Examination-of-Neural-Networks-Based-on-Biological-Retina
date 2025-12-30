import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import BASE_PATH, SESSION

DATA_FILE = os.path.join(BASE_PATH, SESSION, 'processed_data', 'training_dataset_wn.h5')

# --- CONFIGURATION (UPDATED) ---
CENTER_X = 27
CENTER_Y = 47
CROP_SIZE = 40      # הרחבתי קצת ל-40 כדי להיות בטוחים שאנחנו לא מפספסים את הקצה
HISTORY_SIZE = 30   
BATCH_SIZE = 64     # הקטנתי Batch Size לאימון יציב יותר
LEARNING_RATE = 5e-4 # הורדתי את ה-LR (0.0005) למניעת קפיצות ב-Loss
EPOCHS = 20         
REGULARIZATION = 1e-4 # הורדתי קצת את הענישה כדי לא לחנוק את המודל
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 Running on device: {DEVICE}")

# --- DATASET ---
class RetinaDataset(Dataset):
    def __init__(self, h5_file, history_size, crop_size, cx, cy):
        self.history_size = history_size
        self.crop_half = crop_size // 2
        
        with h5py.File(h5_file, 'r') as f:
            X_full = f['X'][:]
            self.Y_full = f['Y'][:]
            
        # Crop Logic
        y1 = max(0, cy - self.crop_half)
        y2 = min(X_full.shape[1], cy + self.crop_half)
        x1 = max(0, cx - self.crop_half)
        x2 = min(X_full.shape[2], cx + self.crop_half)
        
        print(f"✂️ Cropping Input: y[{y1}:{y2}], x[{x1}:{x2}] around center ({cx},{cy})")
        self.X_cropped = X_full[:, y1:y2, x1:x2]
        
        # Consistent Normalization (Use fixed centering like before)
        self.X_cropped = self.X_cropped - 0.5 
        
        self.num_samples = self.X_cropped.shape[0] - history_size
        self.input_shape = self.X_cropped.shape[1:] 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_clip = self.X_cropped[idx : idx + self.history_size] 
        y_val = self.Y_full[idx + self.history_size - 1]    
        return torch.FloatTensor(x_clip.flatten()), torch.FloatTensor([y_val])

# --- MODEL (RESTORED BATCH NORM) ---
class LNModel(nn.Module):
    def __init__(self, input_dim):
        super(LNModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        # החזרתי את ה-Batch Norm! זה קריטי לייצוב ה-Poisson Loss
        self.bn = nn.BatchNorm1d(1) 
        self.nonlinearity = nn.Softplus()

    def forward(self, x):
        linear_out = self.linear(x)
        normalized = self.bn(linear_out) # Normalize before activation
        firing_rate = self.nonlinearity(normalized)
        return firing_rate

# --- TRAINING LOOP ---
def train_model():
    if not os.path.exists(DATA_FILE):
        print("❌ Dataset not found.")
        return

    print("Loading Dataset...")
    dataset = RetinaDataset(DATA_FILE, HISTORY_SIZE, CROP_SIZE, CENTER_X, CENTER_Y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    print(f"Input Dimension: {input_dim}")
    
    model = LNModel(input_dim).to(DEVICE)
    criterion = nn.PoissonNLLLoss(log_input=False) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    
    print("\n--- Starting Training ---")
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # --- EVALUATION ---
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    preds_flat = np.array(all_preds).flatten()
    targets_flat = np.array(all_targets).flatten()
    
    pcc = np.corrcoef(preds_flat, targets_flat)[0, 1]
    print(f"\n🏆 Test Set Pearson Correlation: {pcc:.4f}")

    # Visualization
    H_crop, W_crop = dataset.input_shape
    weights = model.linear.weight.data.cpu().numpy().flatten()
    rf_movie = weights.reshape(HISTORY_SIZE, H_crop, W_crop)
    peak_frame_idx = np.argmax(np.abs(rf_movie).sum(axis=(1, 2)))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rf_movie[peak_frame_idx], cmap='RdBu_r')
    plt.title(f"Spatial RF (Cropped)\nLag: {peak_frame_idx}")
    plt.colorbar()
    
    cy, cx = H_crop//2, W_crop//2 
    plt.subplot(1, 3, 2)
    plt.plot(rf_movie[:, cy, cx], marker='o')
    plt.title("Temporal Profile")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title("Loss")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()