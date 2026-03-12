import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CENTER_X = 27
CENTER_Y = 47
CROP_HALF = 12
HISTORY_SIZE = 20

print("Initializing CBEM (GLM Approach) - Path A...")


def load_data_for_glm():
    with h5py.File(DATA_FILE, 'r') as f:
        X_full = f['X'][:]
        Y_full = f['Y'][:]

    x_s = max(CENTER_X - CROP_HALF, 0)
    x_e = min(CENTER_X + CROP_HALF + 1, X_full.shape[2])
    y_s = max(CENTER_Y - CROP_HALF, 0)
    y_e = min(CENTER_Y + CROP_HALF + 1, X_full.shape[1])
    X_crop = X_full[:, y_s:y_e, x_s:x_e].astype(np.float32)

    return X_crop, Y_full


def create_design_matrix(X, Y, indices, history_size):
    X_flat = X.reshape(X.shape[0], -1)

    features = []
    targets = []

    for t in indices:
        if t < history_size:
            continue
        stim_history = X_flat[t - history_size:t].flatten()
        features.append(stim_history)
        targets.append(Y[t])

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_cbem():
    X, Y = load_data_for_glm()
    print(f"Loaded Data. Total frames: {len(X)}, Crop shape: {X.shape[1:]}")

    train_indices = range(HISTORY_SIZE, 20000)
    test_indices = range(20000, 25000)

    print("Preparing Training Data...")
    X_train, y_train = create_design_matrix(X, Y, train_indices, HISTORY_SIZE)
    print(f"Training design matrix shape: {X_train.shape}")

    print("Preparing Test Data...")
    X_test, y_test = create_design_matrix(X, Y, test_indices, HISTORY_SIZE)
    print(f"Test design matrix shape: {X_test.shape}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('glm', PoissonRegressor(alpha=1.0, max_iter=1000)),
    ])

    print("Fitting Pipeline (StandardScaler -> PCA -> PoissonRegressor)...")
    pipeline.fit(X_train, y_train)

    explained = pipeline.named_steps['pca'].explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained:.2%}")

    print("Predicting...")
    y_pred = pipeline.predict(X_test)

    if np.std(y_test) == 0 or np.std(y_pred) == 0:
        print("Warning: zero variance in predictions or ground truth. Correlation undefined.")
        pcc = 0.0
    else:
        pcc = np.corrcoef(y_pred, y_test)[0, 1]

    print(f"\nCBEM Model (Path A) Result: Pearson Correlation = {pcc:.4f}")

    plt.figure(figsize=(14, 5))
    plt.plot(y_test[:300], 'k', linewidth=1, label='Ground Truth (spikes)')
    plt.plot(y_pred[:300], 'orange', linewidth=1, label='GLM Prediction')
    plt.title(f"CBEM Model (GLM) | Pearson r = {pcc:.4f}")
    plt.xlabel("Test Sample")
    plt.ylabel("Firing Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_cbem()
