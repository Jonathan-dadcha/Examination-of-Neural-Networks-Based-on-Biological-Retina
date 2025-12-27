import os
import h5py
import numpy as np
import pandas as pd
from config import BASE_PATH, SESSION

def prepare_training_data():
    processed_dir = os.path.join(BASE_PATH, SESSION, 'processed_data')
    spikes_dir = os.path.join(BASE_PATH, SESSION, 'spiketimes')
    frametimes_dir = os.path.join(BASE_PATH, SESSION, 'frametimes')
    
    print(f"--- Processing Data for Session: {SESSION} (SYNC FIX) ---")
    
    # 1. Load Real Frame Times
    # Looking for file starting with '5_checkerflicker' in frametimes
    ft_files = [f for f in os.listdir(frametimes_dir) if f.startswith('5_checker') and f.endswith('.mat')]
    if not ft_files:
        print("❌ Frametimes file not found!")
        return
    
    ft_path = os.path.join(frametimes_dir, ft_files[0])
    print(f"⏱️ Loading Frame Times from: {ft_files[0]}")
    
    try:
        with h5py.File(ft_path, 'r') as f:
            # Usually stored as 'ftimes' or similar
            list(f.keys())
            # Assuming the variable is named 'ftimes' based on previous checks
            real_times = f['ftimes'][:].flatten() 
            # Convert ms to seconds if needed (check values)
            if np.mean(real_times) > 10000: # Likely in microseconds or weird scale? 
                # Usually these are in ms. Let's assume ms and convert to seconds.
                real_times = real_times / 1000.0
            elif np.mean(real_times) > 1000: # In ms
                 real_times = real_times / 1000.0
            
            print(f"   - First frame time: {real_times[0]:.4f}s")
            print(f"   - Last frame time:  {real_times[-1]:.4f}s")
            print(f"   - Total Frames recorded: {len(real_times)}")
            
    except Exception as e:
        print(f"❌ Error reading .mat file: {e}")
        return

    # 2. Load White Noise Video (X)
    wn_path = os.path.join(processed_dir, 'white_noise_v2.h5')
    if os.path.exists(wn_path):
        with h5py.File(wn_path, 'r') as f:
            raw_stim = f['stimulus'][:]
            
            # Align Dimensions: Ensure (Time, Height, Width)
            if raw_stim.shape[0] != len(real_times):
                # Probably (Height, Width, Time) or (Height, Time, Width)
                # Find the dimension that is LARGER than len(real_times) (since we generated extra)
                time_dim = np.argmax(raw_stim.shape)
                
                # Move time to front
                X_full = np.moveaxis(raw_stim, time_dim, 0)
            else:
                X_full = raw_stim

            # CROP VIDEO TO MATCH REAL EXPERIMENT
            # We generated 75,000 frames, but maybe only 53,998 were shown.
            valid_frames = len(real_times)
            X = X_full[:valid_frames, :, :]
            
            print("✅ Video Synced & Cropped.")
            print(f"   - Old shape: {X_full.shape}")
            print(f"   - New shape: {X.shape} (Matches frametimes)")

    else:
        print("❌ White Noise file missing.")
        return

    # 3. Load & Bin Spikes (Y)
    print("\n--- Loading White Noise Spikes ---")
    spike_files = sorted([f for f in os.listdir(spikes_dir) if f.startswith('5_SP')])
    
    if spike_files:
        # Use the BEST cell we found
        cell_file = '5_SP_C8704.txt' 
        if cell_file not in spike_files:
             cell_file = spike_files[0]

        spikes_path = os.path.join(spikes_dir, cell_file)
        spikes_timestamp = pd.read_csv(spikes_path, header=None)[0].values
        
        print(f"✅ Loaded Cell {cell_file}")
        print(f"   - Total Spikes: {len(spikes_timestamp)}")
        
        # BINNING USING REAL FRAME TIMES
        # We define bins edges based on the frame onsets.
        # Spikes occurring between Frame 1 and Frame 2 belong to Bin 1.
        
        # Append one last edge to close the final bin (avg duration)
        avg_dt = np.mean(np.diff(real_times))
        bin_edges = np.append(real_times, real_times[-1] + avg_dt)
        
        Y, _ = np.histogram(spikes_timestamp, bins=bin_edges)
        
        print("✅ Spikes Binned (Synced).")
        print(f"   - Y shape: {Y.shape}")
        
        # Save Final Dataset
        save_path = os.path.join(processed_dir, 'training_dataset_wn.h5')
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('Y', data=Y)
        print(f"\n💾 DATASET READY: {save_path}")
        
    else:
        print("❌ No spike files found.")

if __name__ == "__main__":
    prepare_training_data()