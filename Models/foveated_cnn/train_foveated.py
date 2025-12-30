import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import h5py

# ייבוא השכבה החדשה (נמצאת באותה תיקייה)
from foveated_layers import FoveatedConv2d

# --- הגדרת נתיבים ---
# תיקון: מוצאים את תיקיית src ביחס למיקום הקובץ הנוכחי
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
sys.path.append(src_dir)

try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(src_dir, '..'))

# --- הגדרות ---
DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'processed_data', 'foveated_cnn_checkpoint.pth')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

HISTORY_SIZE = 40
BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 0.001

# ==========================================
# 1. המודל החדש: Foveated Deep Retina
# ==========================================
class FoveatedRetina(nn.Module):
    def __init__(self, history_size, spatial_shape):
        super(FoveatedRetina, self).__init__()
        H, W = spatial_shape
        
        # Foveated Layer: רדיוס 0.4 אומר ש-40% מהמרכז חד, השאר מטושטש
        self.conv1 = FoveatedConv2d(history_size, 16, kernel_size=15, img_size=(H, W), fovea_radius=0.4)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # חישוב גודל ביניים (Valid padding)
        h_1 = H - 14
        w_1 = W - 14
        
        # שכבה רגילה שנייה
        self.conv2 = nn.Conv2d(16, 8, kernel_size=9)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        
        # חישוב גודל סופי
        h_out = h_1 - 8
        w_out = w_1 - 8
        
        self.flat_dim = 8 * h_out * w_out
        self.dense = nn.Linear(self.flat_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        out = self.softplus(self.dense(x))
        return out

# ==========================================
# 2. פונקציות עזר
# ==========================================
def load_data():
    print("📂 Loading H5 Dataset...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find data file: {DATA_FILE}")
        
    with h5py.File(DATA_FILE, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    
    # חיתוך מרחבי 50x50
    CENTER_X, CENTER_Y = 27, 47
    crop = 50
    c_h = crop // 2
    
    y_start = max(0, CENTER_Y - c_h)
    y_end = min(X.shape[1], CENTER_Y + c_h)
    x_start = max(0, CENTER_X - c_h)
    x_end = min(X.shape[2], CENTER_X + c_h)
    
    X = X[:, y_start:y_end, x_start:x_end]
    X = (X - np.mean(X)) / (np.std(X) + 1e-6)
    return X, Y

def create_sequences(X, Y, history):
    num_samples = len(X) - history
    X_seq = np.zeros((num_samples, history, X.shape[1], X.shape[2]), dtype=np.float32)
    Y_seq = Y[history:]
    for i in range(num_samples):
        X_seq[i] = X[i:i+history]
    return torch.FloatTensor(X_seq), torch.FloatTensor(Y_seq)

def poisson_loss(pred, target):
    return (pred - target * torch.log(pred + 1e-6)).mean()

# ==========================================
# 3. Main Training Loop (עם מנגנון Resume)
# ==========================================
def main():
    print(f"🚀 Initializing Foveated CNN on: {DEVICE}")
    
    # 1. טעינת נתונים
    X_raw, Y_raw = load_data()
    
    # פיצול
    split_idx = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    Y_train_raw, Y_test_raw = Y_raw[:split_idx], Y_raw[split_idx:]
    
    train_limit = 30000 
    test_limit = 5000
    
    print("✂️ Creating sequences...")
    X_train_tensor, Y_train_tensor = create_sequences(
        X_train_raw[-train_limit:], Y_train_raw[-train_limit:], HISTORY_SIZE)
    X_test_tensor, Y_test_tensor = create_sequences(
        X_test_raw[:test_limit], Y_test_raw[:test_limit], HISTORY_SIZE)
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. אתחול מודל
    model = FoveatedRetina(HISTORY_SIZE, (50, 50)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- מנגנון RESUME חכם ---
    start_epoch = 0
    train_losses = []
    test_corrs = []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Found checkpoint at {CHECKPOINT_PATH}. Resuming training...")
        # טעינה בטוחה (weights_only=False כדי לטעון גם מספרים ורשימות)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # טעינת מצב האופטימייזר (חשוב ל-Momentum של Adam)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # שחזור האפוק וההיסטוריה
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"⏩ Continuing from Epoch {start_epoch + 1}")
        
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            test_corrs = checkpoint['test_corrs']
    else:
        print("🆕 No checkpoint found. Starting from scratch.")

    # 3. לולאת האימון
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = poisson_loss(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test_tensor.to(DEVICE)
            preds = model(X_test_dev).squeeze().cpu().numpy()
            targets = Y_test_tensor.numpy()
            
            if np.std(preds) > 1e-5 and np.std(targets) > 1e-5:
                corr = np.corrcoef(preds, targets)[0, 1]
            else:
                corr = 0.0
            test_corrs.append(corr)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Corr: {corr:.4f}")
        
        # שמירת צ'קפוינט מלא (כולל הכל)
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_corrs': test_corrs
            }, CHECKPOINT_PATH)
            print(f"💾 Checkpoint saved (Epoch {epoch+1}).")

    # 4. שרטוט סופי
    print(f"🏆 Final Result (Foveated): {test_corrs[-1] if test_corrs else 0:.4f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss')
    plt.title('Training Loss')
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_corrs, color='purple', label='Correlation')
    plt.title(f'Test Correlation (Max: {max(test_corrs) if test_corrs else 0:.4f})')
    plt.grid()
    
    save_fig_path = os.path.join(BASE_PATH, 'processed_data', 'Figure_Foveated_CNN.png')
    plt.savefig(save_fig_path)
    print(f"✅ Graph saved to {save_fig_path}")

if __name__ == "__main__":
    main()