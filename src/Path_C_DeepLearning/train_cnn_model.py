import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import BASE_PATH, SESSION
except ImportError:
    from config import BASE_PATH, SESSION

DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'processed_data', 'cnn_checkpoint.pth')

CENTER_X = 27
CENTER_Y = 47
CROP_SIZE = 50      
HISTORY_SIZE = 40   
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 35           
REGULARIZATION = 1e-5 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 Training CNN on device: {DEVICE}")

# --- 1. DATASET ---
class DeepRetinaDataset(Dataset):
    def __init__(self, h5_file, history_size, crop_size, cx, cy):
        self.history_size = history_size
        self.crop_half = crop_size // 2
        
        print("📂 Loading H5 Dataset into memory...")
        with h5py.File(h5_file, 'r') as f:
            X_full = f['X'][:]  
            self.Y_full = f['Y'][:]
            
        y1 = max(0, cy - self.crop_half)
        y2 = min(X_full.shape[1], cy + self.crop_half)
        x1 = max(0, cx - self.crop_half)
        x2 = min(X_full.shape[2], cx + self.crop_half)
        
        print(f"✂️ Cropping Input: y[{y1}:{y2}], x[{x1}:{x2}] around center ({cx},{cy})")
        self.X_cropped = X_full[:, y1:y2, x1:x2]
        
        mean = np.mean(self.X_cropped)
        std = np.std(self.X_cropped)
        self.X_cropped = (self.X_cropped - mean) / (std + 1e-6)
        
        self.num_samples = self.X_cropped.shape[0] - history_size
        self.spatial_shape = self.X_cropped.shape[1:]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_clip = self.X_cropped[idx : idx + self.history_size] 
        y_val = self.Y_full[idx + self.history_size - 1]
        return torch.FloatTensor(x_clip), torch.FloatTensor([y_val])

# --- 2. CNN ARCHITECTURE ---
class DeepRetina(nn.Module):
    def __init__(self, history_size, spatial_shape):
        super(DeepRetina, self).__init__()
        H, W = spatial_shape
        
        self.conv1 = nn.Conv2d(in_channels=history_size, out_channels=16, kernel_size=15)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        
        h_out = (H - 15 + 1) - 9 + 1
        w_out = (W - 15 + 1) - 9 + 1
        self.flat_dim = 8 * h_out * w_out
        
        self.dense = nn.Linear(self.flat_dim, 1)
        self.softplus = nn.Softplus() 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        firing_rate = self.softplus(x)
        return firing_rate

# --- 3. TRAINING LOOP WITH CHECKPOINTS ---
def train_cnn():
    if not os.path.exists(DATA_FILE):
        print("❌ Dataset not found.")
        return

    dataset = DeepRetinaDataset(DATA_FILE, HISTORY_SIZE, CROP_SIZE, CENTER_X, CENTER_Y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DeepRetina(HISTORY_SIZE, dataset.spatial_shape).to(DEVICE)
    print(f"🧠 Model Architecture Created.")
    
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    
    start_epoch = 0
    train_losses = []
    test_correlations = []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Found checkpoint at {CHECKPOINT_PATH}. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        test_correlations = checkpoint.get('test_correlations', [])
        print(f"✅ Resuming training from epoch {start_epoch+1}")
    else:
        print("🆕 Starting training from scratch.")
    
    print("\n--- Starting Training (STABLE MODE + CHECKPOINTS) ---")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                preds = model(inputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())
        
        preds_arr = np.array(all_preds).flatten()
        targets_arr = np.array(all_targets).flatten()
        
        if np.std(preds_arr) < 1e-9:
            pcc = 0.0
        else:
            pcc = np.corrcoef(preds_arr, targets_arr)[0, 1]
            if np.isnan(pcc): pcc = 0.0
            
        test_correlations.append(pcc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Corr: {pcc:.4f}")

        if (epoch + 1) % 5 == 0:
            print(f"💾 Saving checkpoint to {CHECKPOINT_PATH}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_correlations': test_correlations
            }, CHECKPOINT_PATH)

    torch.save({
        'epoch': EPOCHS-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_correlations': test_correlations
    }, CHECKPOINT_PATH)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title("Training Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(test_correlations, color='orange', label='Test Correlation')
    plt.title(f"Final Correlation: {test_correlations[-1] if test_correlations else 0:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"\n🏆 FINAL RESULT: Test Correlation = {test_correlations[-1] if test_correlations else 0:.4f}")

if __name__ == "__main__":
    train_cnn()