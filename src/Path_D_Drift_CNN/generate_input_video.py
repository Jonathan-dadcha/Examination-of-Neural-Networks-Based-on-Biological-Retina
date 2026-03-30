"""
Generate an animated visualisation of the 50x50 peripheral input stimulus
that the Unified Foveated CNN receives during inference.

Loads raw frames from the HDF5 dataset, crops them to the same 50x50 region
used by UnifiedFoveatedDataset, denormalises to uint8, and saves the result
as both an animated GIF and an MP4 video.
"""

import os
import h5py
import numpy as np
import imageio
from PIL import Image

SPIKES_PATH = (
    "/Users/jonathandadcha/Desktop/Retina-Comp-Project/"
    "data/10.12751_g-node.2j3d2i/processed_data/training_dataset_ns_full.h5"
)

OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
)

CENTER_X = 27
CENTER_Y = 47
CROP_SIZE = 50
HALF_CROP = CROP_SIZE // 2

NUM_FRAMES = 250
FPS = 30


def main(output_dir=None):
    if not os.path.exists(SPIKES_PATH):
        raise FileNotFoundError(f"Data file not found: {SPIKES_PATH}")

    out = output_dir if output_dir else OUTPUT_DIR

    print("Loading raw frames from HDF5 …")
    with h5py.File(SPIKES_PATH, "r") as f:
        X = f["X"][:]
    print(f"  Loaded array with shape {X.shape}")

    N, H, W = X.shape
    cx = max(HALF_CROP, min(W - HALF_CROP, CENTER_X))
    cy = max(HALF_CROP, min(H - HALF_CROP, CENTER_Y))

    start = max(0, N // 2 - NUM_FRAMES // 2)
    end = min(N, start + NUM_FRAMES)
    print(f"  Extracting frames {start}–{end - 1} ({end - start} frames)")

    crops = X[start:end, cy - HALF_CROP : cy + HALF_CROP,
                         cx - HALF_CROP : cx + HALF_CROP]

    lo, hi = float(crops.min()), float(crops.max())
    if hi - lo < 1e-8:
        print("  WARNING: frame values are near-constant; output may be blank.")
        crops_u8 = np.zeros(crops.shape, dtype=np.uint8)
    else:
        crops_u8 = ((crops - lo) / (hi - lo) * 255).astype(np.uint8)

    unique_count = len({f.tobytes() for f in crops_u8})
    print(f"  Unique uint8 frames: {unique_count} / {len(crops_u8)} "
          "(sub-pixel drift produces subtle per-frame changes)")

    os.makedirs(out, exist_ok=True)
    gif_path = os.path.join(out, "input_stimulus.gif")
    mp4_path = os.path.join(out, "input_stimulus.mp4")

    print(f"  Saving GIF  → {gif_path}")
    pil_frames = [Image.fromarray(f, mode="L") for f in crops_u8]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=False,
    )

    print(f"  Saving MP4  → {mp4_path}")
    writer = imageio.get_writer(mp4_path, format="FFMPEG", fps=FPS, macro_block_size=1, codec="libx264")
    for frame in crops_u8:
        writer.append_data(frame)
    writer.close()

    print("Done.")


if __name__ == "__main__":
    main()
