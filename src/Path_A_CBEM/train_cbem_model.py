import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor

# הוספת התיקייה שמעל (src) לנתיב
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CENTER_X = 27
CENTER_Y = 47

HISTORY_SIZE = 20

print("🚀 Initializing CBEM (GLM Approach) - Path A...")

def load_data_for_glm():
    with h5py.File(DATA_FILE, 'r') as f:
        X_full = f['X'][:]
        Y_full = f['Y'][:]
    
    # חיתוך מרחבי 5x5
    c_h = 2
    x_s, x_e = CENTER_X - c_h, CENTER_X + c_h + 1
    y_s, y_e = CENTER_Y - c_h, CENTER_Y + c_h + 1
    X_crop = X_full[:, y_s:y_e, x_s:x_e]
    
    # נרמול
    X_crop = (X_crop - np.mean(X_crop)) / (np.std(X_crop) + 1e-6)
    return X_crop, Y_full

def create_design_matrix(X, Y, indices, history_size):
    """
    יצירת מטריצת פיצ'רים רק עבור אינדקסים ספציפיים
    """
    num_samples = len(indices)
    X_flat = X.reshape(X.shape[0], -1)
    
    features = []
    targets = []
    
    for t in indices:
        if t < history_size: continue
        stim_history = X_flat[t-history_size:t].flatten() 
        features.append(stim_history)
        targets.append(Y[t])
        
    return np.array(features), np.array(targets)

def train_cbem():
    X, Y = load_data_for_glm()
    print(f"Loaded Data. Total frames: {len(X)}")
    
    # --- הגדרת החלוקה: אימון ובדיקה עוקבים ---
    # אימון: פריימים 20 עד 20,000
    train_indices = range(HISTORY_SIZE, 20000)
    # בדיקה: פריימים 20,000 עד 25,000 (איפה שיש בטוח פעילות)
    test_indices = range(20000, 25000)
    
    print("📦 Preparing Training Data...")
    X_train, y_train = create_design_matrix(X, Y, train_indices, HISTORY_SIZE)
    print(f"📊 Training shape: {X_train.shape}")
    
    print("🧠 Fitting GLM (Poisson Regression)...")
    glm = PoissonRegressor(alpha=0.1, max_iter=300)
    glm.fit(X_train, y_train)
    
    print("🧪 Preparing Test Data...")
    X_test, y_test = create_design_matrix(X, Y, test_indices, HISTORY_SIZE)
    
    print("🔮 Predicting...")
    y_pred = glm.predict(X_test)
    
    # --- בדיקה והגנה מקריסה ---
    if np.std(y_test) == 0:
        print("⚠️ Warning: Test data (Biology) has zero variance (flat line). Cannot compute correlation.")
        pcc = 0.0
    else:
        pcc = np.corrcoef(y_pred, y_test)[0, 1]

    print(f"\n🏆 CBEM Model (Path A) Result: Correlation = {pcc:.4f}")
    
    # ויזואליזציה
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:200], 'k', label='Biology (Ground Truth)')
    plt.plot(y_pred[:200], 'orange', label='CBEM Prediction')
    plt.title(f"CBEM Model (GLM) | Correlation: {pcc:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    train_cbem()