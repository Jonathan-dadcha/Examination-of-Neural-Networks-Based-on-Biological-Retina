import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import PoissonRegressor

# הגדרת נתיבים
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- הגדרות ---
DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CNN_CHECKPOINT = os.path.join(BASE_PATH, 'processed_data', 'cnn_checkpoint.pth')
CENTER_X = 27
CENTER_Y = 47
HISTORY_SIZE = 40
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# בחירת אזור פעיל ראשוני
TEST_START = 20000
TEST_END = 21000

print("🚀 Starting FINAL Comparative Analysis (The Grand Finale)...")

# ==========================================
# 2. מודל A: CBEM (GLM)
# ==========================================
def run_cbem(X, Y):
    print("running Path A: CBEM (GLM)...")
    # אימון קצר
    X_train = X[:2000, CENTER_Y, CENTER_X].reshape(-1, 1)
    Y_train = Y[:2000]
    
    glm = PoissonRegressor(alpha=0.1, max_iter=100)
    glm.fit(X_train, Y_train)
    
    # חיזוי
    X_test_flat = X[TEST_START:TEST_END, CENTER_Y, CENTER_X].reshape(-1, 1)
    preds = glm.predict(X_test_flat)
    return preds

# ==========================================
# 3. מודל B: Retinomorphic ODE
# ==========================================
def run_ode(X_slice):
    print("running Path B: Retinomorphic ODE...")
    tau = 0.03
    gain = 1.1
    bias = -2.38
    dt = 1.0/60.0
    
    c_h, s_h = 1, 4
    center = np.mean(X_slice[:, CENTER_Y-c_h:CENTER_Y+c_h+1, CENTER_X-c_h:CENTER_X+c_h+1], axis=(1,2))
    surround = np.mean(X_slice[:, CENTER_Y-s_h:CENTER_Y+s_h+1, CENTER_X-s_h:CENTER_X+s_h+1], axis=(1,2))
    contrast = (surround - center)
    std_val = np.std(contrast)
    if std_val == 0: std_val = 1
    contrast = (contrast - np.mean(contrast)) / std_val
    
    y_state = 0.0
    preds = []
    for t in range(len(contrast)):
        I = contrast[t]
        dy = (I - y_state) / tau
        y_state += dy * dt
        val = gain * y_state + bias
        rate = np.log(1 + np.exp(val)) 
        preds.append(max(0, rate))
        
    return np.array(preds)

# ==========================================
# 4. מודל C: CNN (Deep Retina)
# ==========================================
class DeepRetina(nn.Module):
    def __init__(self, history_size, spatial_shape):
        super(DeepRetina, self).__init__()
        H, W = spatial_shape
        self.conv1 = nn.Conv2d(history_size, 16, 15)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 9)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        h_out = (H - 15 + 1) - 9 + 1
        w_out = (W - 15 + 1) - 9 + 1
        self.flat_dim = 8 * h_out * w_out
        self.dense = nn.Linear(self.flat_dim, 1)
        self.softplus = nn.Softplus() 

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.softplus(self.dense(x))

def run_cnn(X_full):
    print("running Path C: CNN (Deep Retina)...")
    
    if not os.path.exists(CNN_CHECKPOINT):
        return np.zeros(TEST_END - TEST_START)

    crop = 50
    c_half = crop // 2
    X_crop = X_full[:, CENTER_Y-c_half:CENTER_Y+c_half, CENTER_X-c_half:CENTER_X+c_half]
    X_crop = (X_crop - np.mean(X_crop)) / (np.std(X_crop) + 1e-6)
    
    preds = []
    model = DeepRetina(HISTORY_SIZE, (50,50)).to(DEVICE)
    
    checkpoint = torch.load(CNN_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_range = range(TEST_START, TEST_END)
    
    with torch.no_grad():
        for t in test_range:
            if t < HISTORY_SIZE:
                preds.append(0.0)
                continue
            x_clip = X_crop[t-HISTORY_SIZE:t]
            x_tensor = torch.FloatTensor(x_clip).unsqueeze(0).to(DEVICE)
            out = model(x_tensor)
            preds.append(out.item())
            
    return np.array(preds)

# ==========================================
# Main Execution
# ==========================================
def main():
    # כאן התיקון - מגדירים global בהתחלה
    global TEST_START, TEST_END 

    with h5py.File(DATA_FILE, 'r') as f:
        X_full = f['X'][:]
        Y_full = f['Y'][:]
    
    # בדיקת פעילות ראשונית
    Y_test = Y_full[TEST_START:TEST_END]
    
    if np.std(Y_test) == 0:
        print("⚠️ Region has NO spikes. Moving to backup region (50000-51000)...")
        TEST_START = 50000
        TEST_END = 51000
        Y_test = Y_full[TEST_START:TEST_END]

    print(f"🔬 Testing on frames {TEST_START} to {TEST_END} (Activity Check: std={np.std(Y_test):.4f})")

    pred_a = run_cbem(X_full, Y_full)
    pred_b = run_ode(X_full[TEST_START:TEST_END])
    pred_c = run_cnn(X_full)
    
    # יישור
    min_len = min(len(Y_test), len(pred_a), len(pred_b), len(pred_c))
    Y_test = Y_test[:min_len]
    pred_a = pred_a[:min_len]
    pred_b = pred_b[:min_len]
    pred_c = pred_c[:min_len]

    def get_corr(p, t):
        if np.std(p) < 1e-9 or np.std(t) < 1e-9: return 0.0
        return np.corrcoef(p, t)[0,1]

    corr_a = get_corr(pred_a, Y_test)
    corr_b = get_corr(pred_b, Y_test)
    corr_c = get_corr(pred_c, Y_test)
    
    print("\n" + "="*30)
    print("🏆 FINAL RESULTS 🏆")
    print(f"Path A (CBEM): Corr = {corr_a:.4f}")
    print(f"Path B (ODE):  Corr = {corr_b:.4f}")
    print(f"Path C (CNN):  Corr = {corr_c:.4f}")
    print("="*30)
    
    plt.figure(figsize=(12, 10))
    limit = 300
    
    plt.subplot(4,1,1)
    plt.plot(Y_test[:limit], 'k', linewidth=1.5, label='Biology (GT)')
    plt.title("Biological Retina Response")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    plt.subplot(4,1,2)
    plt.plot(pred_a[:limit], 'orange', label=f'A: GLM (Corr={corr_a:.3f})')
    plt.title("Path A: Linear Statistical Model")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

    plt.subplot(4,1,3)
    plt.plot(pred_b[:limit], 'green', label=f'B: ODE (Corr={corr_b:.3f})')
    plt.title("Path B: Biophysical Model")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

    plt.subplot(4,1,4)
    plt.plot(pred_c[:limit], 'blue', linewidth=1.2, label=f'C: CNN (Corr={corr_c:.3f})')
    plt.title("Path C: Deep Learning Model")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'processed_data', 'FINAL_COMPARISON.png'))
    plt.show()

if __name__ == "__main__":
    main()