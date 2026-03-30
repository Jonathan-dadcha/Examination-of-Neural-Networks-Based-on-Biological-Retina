import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DriftSimulationDataset(Dataset):
    """
    Loads SYNCHRONIZED natural scenes video and spikes.
    Applies BIOLOGICAL MICRO-DRIFT (jitter) during the cropping process.
    """

    def __init__(self, images_path, spikes_path, history_size=40, crop_size=50):
        self.history_size = history_size
        self.crop_size = crop_size

        print(f"Loading Synced Data from: {os.path.basename(spikes_path)}")

        if not os.path.exists(spikes_path):
            raise FileNotFoundError(f"File not found: {spikes_path}")

        with h5py.File(spikes_path, 'r') as f:
            self.X = f['X'][:]
            self.Y = f['Y'][:]

        self.center_x = 27
        self.center_y = 47

        mean = np.mean(self.X)
        std = np.std(self.X)
        self.X = (self.X - mean) / (std + 1e-6)

        print(f"Dataset Loaded: {self.X.shape[0]} frames.")
        print("   -> Configuration: Micro-Drift (std=0.25, clip=3px)")

    def __len__(self):
        return len(self.Y) - self.history_size

    def __getitem__(self, idx):
        raw_clip = self.X[idx: idx + self.history_size]

        drift_x = np.cumsum(np.random.randn(self.history_size) * 0.25)
        drift_y = np.cumsum(np.random.randn(self.history_size) * 0.25)

        drift_x = np.clip(drift_x, -3, 3)
        drift_y = np.clip(drift_y, -3, 3)

        frames = []
        for t in range(self.history_size):
            cur_cx = int(self.center_x + drift_x[t])
            cur_cy = int(self.center_y + drift_y[t])

            max_x = raw_clip.shape[2] - self.crop_size // 2
            min_x = self.crop_size // 2
            max_y = raw_clip.shape[1] - self.crop_size // 2
            min_y = self.crop_size // 2

            cur_cx = max(min_x, min(max_x, cur_cx))
            cur_cy = max(min_y, min(max_y, cur_cy))

            x1 = cur_cx - self.crop_size // 2
            x2 = cur_cx + self.crop_size // 2
            y1 = cur_cy - self.crop_size // 2
            y2 = cur_cy + self.crop_size // 2

            frame = raw_clip[t, y1:y2, x1:x2]
            frames.append(frame)

        x_stack = np.array(frames)
        y_val = self.Y[idx + self.history_size - 1]

        return torch.FloatTensor(x_stack), torch.FloatTensor([y_val])
