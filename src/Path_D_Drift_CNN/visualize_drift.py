import matplotlib.pyplot as plt
import matplotlib.animation as animation
from drift_dataset import DriftSimulationDataset

IMAGES_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/natural_scenes.h5"
SPIKES_PATH = "/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i/processed_data/training_dataset_ns_full.h5"

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