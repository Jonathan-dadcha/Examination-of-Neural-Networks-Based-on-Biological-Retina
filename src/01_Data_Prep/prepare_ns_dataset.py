import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import numpy as np
import scipy.io as sio
import pandas as pd
from config import BASE_PATH, SESSION
# --- הגדרות ---
PROCESSED_DIR = os.path.join(BASE_PATH, 'processed_data')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'training_dataset_ns.h5')
NATURAL_SCENES_FILE = os.path.join(PROCESSED_DIR, 'natural_scenes.h5')

# --- RAN1 RNG Implementation ---
class Ran1:
    def __init__(self, seed):
        self.IA = 16807
        self.IM = 2147483647
        self.AM = 1.0 / self.IM
        self.IQ = 127773
        self.IR = 2836
        self.NTAB = 32
        self.NDIV = (1 + (self.IM - 1) // self.NTAB)
        self.EPS = 1.2e-7
        self.RNMX = 1.0 - self.EPS
        
        self.idum = seed
        self.iy = 0
        self.iv = [0] * self.NTAB
        
        if self.idum <= 0 or self.iy == 0:
            if -self.idum < 1: self.idum = 1
            else: self.idum = -self.idum
            
            for j in range(self.NTAB + 7, -1, -1):
                k = self.idum // self.IQ
                self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
                if self.idum < 0: self.idum += self.IM
                if j < self.NTAB: self.iv[j] = self.idum
            self.iy = self.iv[0]

    def get(self):
        k = self.idum // self.IQ
        self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
        if self.idum < 0: self.idum += self.IM
        j = self.iy // self.NDIV
        self.iy = self.iv[j]
        self.iv[j] = self.idum
        temp = self.AM * self.iy
        if temp > self.RNMX: return self.RNMX
        return temp

class DotDict(dict):
    """מאפשר גישה למילון כמו לאובייקט"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_mat_v73(filepath, variable_name):
    """קורא קבצי Matlab v7.3 באמצעות h5py"""
    params = DotDict()
    with h5py.File(filepath, 'r') as f:
        if variable_name not in f:
             grp = f
        else:
            grp = f[variable_name]
            
        if isinstance(grp, h5py.Group):
            for key in grp.keys():
                val = grp[key][()] 
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                params[key] = val
        else:
            val = grp[()]
            return val
    return params

def get_stimulus_parameters():
    stim_dir = os.path.join(BASE_PATH, SESSION, 'stimuli')
    # חיפוש גמיש יותר (imgseq או imseq)
    files = [f for f in os.listdir(stim_dir) if ('imgseq' in f.lower() or 'imseq' in f.lower()) and f.endswith('.mat')]
    
    if not files:
        raise FileNotFoundError(f"Could not find *imgseq*.mat in {stim_dir}")
    
    filepath = os.path.join(stim_dir, files[0])
    print(f"📖 Reading parameters from: {files[0]}")
    
    try:
        mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return mat['stimpara']
    except NotImplementedError:
        print("   (Detected MATLAB v7.3 file, using h5py reader)")
        return load_mat_v73(filepath, 'stimpara')

def get_frame_times():
    ft_dir = os.path.join(BASE_PATH, SESSION, 'frametimes')
    # התיקון: חיפוש גם של 'imseq' (בלי g)
    files = [f for f in os.listdir(ft_dir) if ('imgseq' in f.lower() or 'imseq' in f.lower()) and f.endswith('.mat')]
    
    if not files:
        # הדפסת תוכן התיקייה כדי להבין מה קורה אם עדיין נכשל
        print(f"❌ Files found in {ft_dir}: {os.listdir(ft_dir)}")
        raise FileNotFoundError("Could not find *imgseq*.mat or *imseq*.mat in frametimes folder")
        
    filepath = os.path.join(ft_dir, files[0])
    print(f"⏱️ Reading frame times from: {files[0]}")
    
    try:
        mat = sio.loadmat(filepath, squeeze_me=True)
        ftimes = mat['ftimes']
    except NotImplementedError:
        print("   (Detected MATLAB v7.3 file, using h5py reader)")
        with h5py.File(filepath, 'r') as f:
            ftimes = f['ftimes'][:].flatten()
    
    if np.mean(ftimes) > 1000:
        ftimes = ftimes / 1000.0
    return ftimes

def prepare_dataset():
    print(f"🚀 Starting Natural Scenes Dataset Preparation...")
    
    if not os.path.exists(NATURAL_SCENES_FILE):
        print("❌ natural_scenes.h5 missing. Run prepare_natural_scenes.py first.")
        return
        
    with h5py.File(NATURAL_SCENES_FILE, 'r') as f:
        images = f['train_images'][:] 
        print(f"✅ Loaded {len(images)} unique images.")

    # 2. שחזור סדר ההופעה
    stimpara = get_stimulus_parameters()
    
    seed = int(stimpara.seed)
    n_repeats = int(stimpara.nrepeats) if 'nrepeats' in stimpara else int(stimpara.nRepeats)
    n_images = 200 
    
    print(f"🎲 Reconstructing sequence (Seed: {seed}, Repeats: {n_repeats})...")
    
    rng = Ran1(-abs(seed)) 
    full_sequence = []
    
    for r in range(n_repeats):
        temp_seq = list(range(1, n_images + 1))
        for i in range(len(temp_seq), 1, -1):
            ridx = int(np.floor(rng.get() * i))
            temp_seq[i-1], temp_seq[ridx] = temp_seq[ridx], temp_seq[i-1]
        full_sequence.append(0) 
        full_sequence.extend(temp_seq)

    # 3. בניית הוידאו
    trial_duration = int(stimpara.trialduration) if 'trialduration' in stimpara else 60
    print(f"🎞️ Expanding video (Each image shown for {trial_duration} frames)...")
    
    ftimes = get_frame_times()
    total_frames = len(ftimes)
    
    H, W = images.shape[1], images.shape[2]
    X_movie = np.zeros((total_frames, H, W), dtype=np.uint8)
    gray_screen = np.full((H, W), 127, dtype=np.uint8)
    
    current_frame = 0
    for img_id in full_sequence:
        if current_frame >= total_frames:
            break
            
        if img_id == 0:
            frame_img = gray_screen
        else:
            frame_img = images[img_id - 1]
            
        end_frame = min(current_frame + trial_duration, total_frames)
        X_movie[current_frame : end_frame] = frame_img
        current_frame = end_frame

    print(f"✅ Video constructed. Shape: {X_movie.shape}")

    # 4. סנכרון ספייקים
    spikes_dir = os.path.join(BASE_PATH, SESSION, 'spiketimes')
    
    cell_file = '5_SP_C8704.txt' 
    if not os.path.exists(os.path.join(spikes_dir, cell_file)):
         possible = [f for f in os.listdir(spikes_dir) if f.endswith('.txt') and 'SP' in f]
         if possible: 
             cell_file = possible[0]
         else:
             print("❌ No spike files found!")
             return
    
    print(f"⚡ Loading spikes for cell: {cell_file}")
    spikes_path = os.path.join(spikes_dir, cell_file)
    spikes = pd.read_csv(spikes_path, header=None)[0].values
    
    avg_dt = np.mean(np.diff(ftimes))
    bin_edges = np.append(ftimes, ftimes[-1] + avg_dt)
    
    Y_binned, _ = np.histogram(spikes, bins=bin_edges)
    
    min_len = min(len(X_movie), len(Y_binned))
    X_movie = X_movie[:min_len]
    Y_binned = Y_binned[:min_len]
    
    print(f"✅ Spikes synced. Final Dataset: X={X_movie.shape}, Y={Y_binned.shape}")
    
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('X', data=X_movie)
        f.create_dataset('Y', data=Y_binned)
        
    print("✨ DONE! Dataset ready for CNN training.")

if __name__ == "__main__":
    prepare_dataset()