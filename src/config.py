import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

BASE_PATH = os.environ.get(
    "DATA_ROOT",
    os.path.join(PROJECT_ROOT, 'data', '10.12751_g-node.2j3d2i'),
)

SESSION = '20171116_sr_le_fp'


def get_device():
    """Return the best available torch device: CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
