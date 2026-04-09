import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from drift_dataset import DriftSimulationDataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DATA_DIR = os.path.join(
    os.environ.get("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data", "10.12751_g-node.2j3d2i")),
    "processed_data",
)

IMAGES_PATH = os.path.join(_DATA_DIR, "natural_scenes.h5")
SPIKES_PATH = os.path.join(_DATA_DIR, "training_dataset_ns_full.h5")

def visualize():
    print("🎥 Generating Drift Visualization...")
    ds = DriftSimulationDataset(IMAGES_PATH, SPIKES_PATH)
    
    x_drift, y = ds[1000] 
    
    fig, ax = plt.subplots()
    plt.title(f"Simulated Eye Drift (Input to CNN)\nSpike: {y.item()}")
    im = ax.imshow(x_drift[0], cmap='gray', vmin=-2, vmax=2)
    
    def update(i):
        im.set_array(x_drift[i])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(x_drift), interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    visualize()