import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import h5py
import torch.nn.functional as F

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
sys.path.append(src_dir)

try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(src_dir, '..'))

# --- Hyperparameters ---
DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'processed_data', 'foveated_cnn_checkpoint.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

HISTORY_SIZE = 40
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3 # Standard LR for Path C success

# ==========================================
# 1. Bio-Inspired Foveated CNN (Bio-Blur)
# ==========================================
class FoveatedRetina(nn.Module):
    def __init__(self, history_size, spatial_shape):
        super(FoveatedRetina, self).__init__()
        H, W = spatial_shape
        
        # Spatial Mask: Center=1, Edges=0 (Gaussian)
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        # Sigma 0.6 focuses on the central 30x30 area
        mask = torch.exp(-(dist**2) / (2 * 0.6**2))
        self.register_buffer('fovea_mask', mask.view(1, 1, H, W))
        
        # Gaussian Blur Kernel (Fixed)
        k_size = 7
        sigma_blur = 2.0
        k = torch.linspace(-(k_size // 2), k_size // 2, k_size)
        gauss = torch.exp(-k**2 / (2 * sigma_blur**2))
        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d /= kernel_2d.sum()
        self.register_buffer('blur_kernel', kernel_2d.view(1, 1, k_size, k_size))

        # CNN Backbone (Same as Path C for guaranteed learning)
        self.conv1 = nn.Conv2d(history_size, 16, kernel_size=15)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 8, kernel_size=9)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        
        h_out = (H - 15 + 1) - 9 + 1
        w_out = (W - 15 + 1) - 9 + 1
        self.flat_dim = 8 * h_out * w_out
        self.dense = nn.Linear(self.flat_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Apply Foveated Pre-processing
        B, C, H, W = x.shape
        x_reshaped = x.view(B*C, 1, H, W)
        # Peripheral blur
        x_blurred = F.conv2d(x_reshaped, self.blur_kernel, padding=3).view(B, C, H, W)
        # Combine: Sharp Center + Blurred Periphery
        x_foveated = (x * self.fovea_mask) + (x_blurred * (1 - self.fovea_mask))
        
        # Neural Processing
        x = self.relu1(self.bn1(self.conv1(x_foveated)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.softplus(self.dense(x))

# ==========================================
# 2. Robust Data Loading
# ==========================================
def load_data():
    print("📂 [Data] Loading H5 Dataset...")
    with h5py.File(DATA_FILE, 'r') as f:
        X, Y = f['X'][:], f['Y'][:]
    CENTER_X, CENTER_Y = 27, 47
    crop = 50
    c_h = crop // 2
    X = X[:, CENTER_Y-c_h:CENTER_Y+c_h, CENTER_X-c_h:CENTER_X+c_h]
    X = (X - np.mean(X)) / (np.std(X) + 1e-6)
    return X, Y

def create_sequences(X, Y, history):
    num_samples = len(X) - history
    X_seq = np.zeros((num_samples, history, X.shape[1], X.shape[2]), dtype=np.float32)
    Y_seq = Y[history:]
    for i in range(num_samples):
        X_seq[i] = X[i:i+history]
    return torch.FloatTensor(X_seq), torch.FloatTensor(Y_seq)

def find_active_block(Y, train_len, test_len):
    print("🔍 [Search] Finding active block...")
    best_start, max_spikes = 0, 0
    total = train_len + test_len
    for start in range(0, len(Y) - total, 2500):
        s = np.sum(Y[start : start + total])
        # Check if test set also has action
        if s > max_spikes and np.sum(Y[start+train_len : start+total]) > 500:
            max_spikes, best_start = s, start
    return best_start

# ==========================================
# 3. Main
# ==========================================
def main():
    X_raw, Y_raw = load_data()
    train_len, test_len = 50000, 10000
    best_start = find_active_block(Y_raw, train_len, test_len)
    
    X_train, Y_train = create_sequences(X_raw[best_start:best_start+train_len], Y_raw[best_start:best_start+train_len], HISTORY_SIZE)
    X_test, Y_test = create_sequences(X_raw[best_start+train_len:best_start+train_len+test_len], Y_raw[best_start+train_len:best_start+train_len+test_len], HISTORY_SIZE)
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model = FoveatedRetina(HISTORY_SIZE, (50, 50)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.PoissonNLLLoss(log_input=False)

    print(f"🚀 Training Foveated Bio-Blur CNN on {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_b).squeeze()
            loss = criterion(outputs, Y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(DEVICE)).squeeze().cpu().numpy()
            targets = Y_test.numpy()
            corr = np.corrcoef(preds, targets)[0, 1] if np.std(preds) > 1e-6 else 0.0
            
        print(f"Epoch {epoch+1:02d}/50 | Loss: {epoch_loss/len(train_loader):.4f} | Test Corr: {corr:.4f}")

if __name__ == "__main__":
    main()