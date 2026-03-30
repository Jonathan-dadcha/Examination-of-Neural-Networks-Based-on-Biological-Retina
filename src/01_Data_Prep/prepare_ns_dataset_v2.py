import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import numpy as np
import scipy.io as sio
import pandas as pd
from config import BASE_PATH, SESSION

PROCESSED_DIR = os.path.join(BASE_PATH, 'processed_data')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'training_dataset_ns_full.h5')
NATURAL_SCENES_FILE = os.path.join(PROCESSED_DIR, 'natural_scenes.h5')

# --- Ran1 RNG ---
class Ran1:
    def __init__(self, seed):
        self.IA = 16807; self.IM = 2147483647; self.AM = 1.0/self.IM; self.IQ = 127773; self.IR = 2836
        self.NTAB = 32; self.NDIV = (1+(self.IM-1)//self.NTAB); self.EPS = 1.2e-7; self.RNMX = 1.0-self.EPS
        self.idum = seed; self.iy = 0; self.iv = [0]*self.NTAB
        if self.idum <= 0 or self.iy == 0:
            if -self.idum < 1: self.idum = 1
            else: self.idum = -self.idum
            for j in range(self.NTAB+7, -1, -1):
                k = self.idum//self.IQ; self.idum = self.IA*(self.idum-k*self.IQ)-self.IR*k
                if self.idum < 0: self.idum += self.IM
                if j < self.NTAB: self.iv[j] = self.idum
            self.iy = self.iv[0]
    def get(self):
        k = self.idum//self.IQ; self.idum = self.IA*(self.idum-k*self.IQ)-self.IR*k
        if self.idum < 0: self.idum += self.IM
        j = self.iy//self.NDIV; self.iy = self.iv[j]; self.iv[j] = self.idum
        temp = self.AM*self.iy
        if temp > self.RNMX: return self.RNMX
        return temp

# --- Helper Functions (Loaders) ---
class DotDict(dict):
    __getattr__ = dict.get; __setattr__ = dict.__setitem__; __delattr__ = dict.__delitem__

def load_mat_v73(filepath, variable_name):
    params = DotDict()
    with h5py.File(filepath, 'r') as f:
        grp = f[variable_name] if variable_name in f else f
        if isinstance(grp, h5py.Group):
            for key in grp.keys():
                val = grp[key][()]
                if isinstance(val, np.ndarray) and val.size == 1: val = val.item()
                params[key] = val
        else: return grp[()]
    return params

def get_stimulus_parameters():
    stim_dir = os.path.join(BASE_PATH, SESSION, 'stimuli')
    files = [f for f in os.listdir(stim_dir) if ('imgseq' in f.lower() or 'imseq' in f.lower()) and f.endswith('.mat')]
    filepath = os.path.join(stim_dir, files[0])
    try: return sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)['stimpara']
    except NotImplementedError: return load_mat_v73(filepath, 'stimpara')

def get_frame_times():
    ft_dir = os.path.join(BASE_PATH, SESSION, 'frametimes')
    files = [f for f in os.listdir(ft_dir) if ('imgseq' in f.lower() or 'imseq' in f.lower()) and f.endswith('.mat')]
    filepath = os.path.join(ft_dir, files[0])
    try: ftimes = sio.loadmat(filepath, squeeze_me=True)['ftimes']
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f: ftimes = f['ftimes'][:].flatten()
    if np.mean(ftimes) > 1000: ftimes = ftimes / 1000.0
    return ftimes

# --- MAIN LOGIC ---
def prepare_dataset_v2():
    print(f"🚀 Starting Dataset Expansion (v2)...")
    
    # 1. Load images
    with h5py.File(NATURAL_SCENES_FILE, 'r') as f:
        train_imgs = f['train_images'][:]
        test_imgs = f['test_images'][:] if 'test_images' in f else np.empty((0,100,75))
        all_images = np.concatenate([train_imgs, test_imgs], axis=0)
        print(f"✅ Loaded {len(all_images)} total images (Train+Test).")

    # 2. Reconstruct presentation sequence
    stimpara = get_stimulus_parameters()
    seed = int(stimpara.seed)
    n_repeats = int(stimpara.nrepeats) if 'nrepeats' in stimpara else 10
    n_images = len(all_images)
    
    print(f"🎲 Reconstructing sequence (Seed: {seed}, Images: {n_images})...")
    rng = Ran1(-abs(seed))
    full_sequence = []
    for r in range(n_repeats):
        temp_seq = list(range(1, n_images + 1))
        for i in range(len(temp_seq), 1, -1):
            ridx = int(np.floor(rng.get() * i))
            temp_seq[i-1], temp_seq[ridx] = temp_seq[ridx], temp_seq[i-1]
        full_sequence.append(0) # Gray screen
        full_sequence.extend(temp_seq)

    # 3. Validate alignment with frame times
    ftimes = get_frame_times()
    num_events = len(ftimes)
    
    print(f"📊 Stats: Sequence Length={len(full_sequence)}, FrameTimes Length={len(ftimes)}")
    full_sequence = full_sequence[:num_events]

    # 4. Unroll full video
    trial_frames = int(stimpara.trialduration) if 'trialduration' in stimpara else 60
    total_video_frames = num_events * trial_frames
    
    print(f"🎞️ Expanding to FULL video: {num_events} events x {trial_frames} frames = {total_video_frames} frames")
    
    H, W = 100, 75
    X_movie = np.zeros((total_video_frames, H, W), dtype=np.uint8)
    gray_screen = np.full((H, W), 127, dtype=np.uint8)
    
    for i, img_id in enumerate(full_sequence):
        if img_id == 0: pic = gray_screen
        else: pic = all_images[img_id - 1]
        
        start_f = i * trial_frames
        end_f = start_f + trial_frames
        X_movie[start_f:end_f] = pic
        
        if i % 500 == 0: print(f"   Processed {i}/{num_events} events...")

    # 5. Sync spikes at full video resolution
    print("⚡ Syncing spikes to full video resolution...")
    spikes_dir = os.path.join(BASE_PATH, SESSION, 'spiketimes')
    cell_file = '5_SP_C8704.txt'
    if not os.path.exists(os.path.join(spikes_dir, cell_file)):
         cell_file = [f for f in os.listdir(spikes_dir) if 'SP' in f][0]
    
    spikes = pd.read_csv(os.path.join(spikes_dir, cell_file), header=None)[0].values
    
    avg_event_dur = np.mean(np.diff(ftimes))
    frame_dt = avg_event_dur / trial_frames
    
    print(f"   Frame DT calculated: {frame_dt*1000:.2f} ms ({1/frame_dt:.2f} Hz)")
    
    t_start = ftimes[0]
    all_frame_times = t_start + np.arange(total_video_frames + 1) * frame_dt
    
    Y_binned, _ = np.histogram(spikes, bins=all_frame_times)
    
    # 6. Save
    print(f"💾 Saving full dataset to {OUTPUT_FILE}...")
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('X', data=X_movie, compression="gzip")
        f.create_dataset('Y', data=Y_binned)
        
    print("✨ DONE! FULL High-Res Dataset ready.")

if __name__ == "__main__":
    prepare_dataset_v2()