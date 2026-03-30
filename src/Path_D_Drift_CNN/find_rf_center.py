import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/training_dataset_ns_full.h5"

def find_center():
    print(f"🔍 Loading data from: {os.path.basename(DATA_PATH)}")
    
    if not os.path.exists(DATA_PATH):
        print("❌ File not found!")
        return

    with h5py.File(DATA_PATH, 'r') as f:
        X = f['X'][:]  # (Frames, H, W)
        Y = f['Y'][:]  # Spikes
    
    print(f"✅ Data loaded. X: {X.shape}, Y sum: {np.sum(Y)}")
    
    H, W = X.shape[1], X.shape[2]
    accumulated_frame = np.zeros((H, W))
    spike_count = 0
    
    latency = 4 
    
    spike_indices = np.where(Y > 0)[0]
    
    print(f"⚡ Calculating RF center based on {len(spike_indices)} spikes...")
    
    for idx in spike_indices:
        if idx > latency:
            accumulated_frame += X[idx - latency]
            spike_count += 1
            
    sta_frame = accumulated_frame / spike_count
    
    sta_frame -= np.mean(X)
    

    y_center, x_center = np.unravel_index(np.argmax(np.abs(sta_frame)), sta_frame.shape)
    
    print("\n" + "="*40)
    print(f"🎯 FOUND CENTER: X={x_center}, Y={y_center}")
    print("="*40)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(sta_frame, cmap='RdBu_r')
    plt.scatter([x_center], [y_center], color='lime', marker='x', s=100, label='Center')
    plt.title(f"Receptive Field Location\nCenter: ({x_center}, {y_center})")
    plt.legend()
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    find_center()