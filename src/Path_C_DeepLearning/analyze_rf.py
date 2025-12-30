import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config import BASE_PATH, SESSION


DATA_FILE = os.path.join(BASE_PATH, SESSION, 'processed_data', 'training_dataset_wn.h5')

def calculate_sta():
    if not os.path.exists(DATA_FILE):
        print("❌ Dataset not found. Run data_loader.py first.")
        return

    print(f"Loading data from {DATA_FILE}...")
    with h5py.File(DATA_FILE, 'r') as f:
        X = f['X'][:]  # Stimulus (Frames, Height, Width)
        Y = f['Y'][:]  # Spikes (Frames,)

    print(f"Data Loaded: X={X.shape}, Y={Y.shape}")
    print(f"Total Spikes: {np.sum(Y)}")

    # Parameters for STA
    fps = 30
    time_window = 0.5  # Look back 500ms
    num_lags = int(time_window * fps) # How many frames back
    
    H, W = X.shape[1], X.shape[2]
    sta = np.zeros((num_lags, H, W))
    spike_counts = 0

    print("Calculating Spike Triggered Average (STA)...")
    
    # Iterate through time
    # Start from 'num_lags' because we need history
    for t in range(num_lags, len(Y)):
        if Y[t] > 0: # If there was a spike at time t
            # Take the video clip preceding the spike
            clip = X[t-num_lags : t, :, :] 
            
            # Add to average, weighted by number of spikes (usually 1)
            sta += clip * Y[t]
            spike_counts += Y[t]

    if spike_counts == 0:
        print("❌ No spikes found in the analyzed window.")
        return

    # Normalize
    sta = sta / spike_counts
    
    # Remove the mean background (0.5 for binary noise) to see contrast
    sta = sta - np.mean(X)

    print("✅ STA Calculation Complete.")

    # --- VISUALIZATION ---
    # We want to find the frame with the strongest response (Max variance)
    variances = np.var(sta, axis=(1, 2))
    best_lag = np.argmax(variances)
    time_before_spike = (num_lags - best_lag) * (1000/fps)

    plt.figure(figsize=(10, 5))
    
    # Plot 1: The Spatial Receptive Field (Best Frame)
    plt.subplot(1, 2, 1)
    # Using 'RdBu' colormap: Red=ON, Blue=OFF, White=Background
    max_val = np.max(np.abs(sta[best_lag]))
    plt.imshow(sta[best_lag], cmap='RdBu_r', vmin=-max_val, vmax=max_val)
    plt.title(f"Receptive Field\n({time_before_spike:.0f}ms before spike)")
    plt.colorbar(label='Intensity change')
    plt.axis('off')

    # Plot 2: Temporal Course (Pixel at center of RF)
    # Find center of mass or max pixel
    center_y, center_x = np.unravel_index(np.argmax(np.abs(sta[best_lag])), (H, W))
    
    plt.subplot(1, 2, 2)
    temporal_profile = sta[:, center_y, center_x]
    time_axis = np.linspace(-time_window*1000, 0, num_lags)
    plt.plot(time_axis, temporal_profile, marker='o')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title(f"Temporal Profile (at x={center_x}, y={center_y})")
    plt.xlabel("Time before spike (ms)")
    plt.ylabel("Stimulus Intensity")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_sta()