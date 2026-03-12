import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.optimize import differential_evolution

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CENTER_X = 27
CENTER_Y = 47
DT = 1.0 / 60.0

TRAIN_END = 20000
TEST_END = 25000

PARAM_BOUNDS = [
    (0.001, 0.5),   # tau (seconds)
    (0.1, 10.0),    # gain
    (-5.0, 5.0),    # bias
]

print("Initializing Retinomorphic ODE Optimizer - Path B...")


def simulate_ode(stimulus, tau, gain, bias, dt):
    """Vectorized leaky-integrator ODE followed by softplus nonlinearity."""
    alpha = np.exp(-dt / tau)
    y = lfilter([1 - alpha], [1, -alpha], stimulus)
    rate = np.log1p(np.exp(gain * y + bias))
    return rate


def objective(params, stimulus, spikes):
    """Negative Pearson correlation — minimized by the optimizer."""
    tau, gain, bias = params
    pred = simulate_ode(stimulus, tau, gain, bias, DT)
    if np.std(pred) < 1e-9:
        return 0.0
    corr = np.corrcoef(pred, spikes)[0, 1]
    return -corr if not np.isnan(corr) else 0.0


def load_data_center_surround():
    with h5py.File(DATA_FILE, 'r') as f:
        X_full = f['X'][:]
        Y_full = f['Y'][:]

    c_h = 1
    center = np.mean(
        X_full[:, CENTER_Y - c_h:CENTER_Y + c_h + 1,
               CENTER_X - c_h:CENTER_X + c_h + 1],
        axis=(1, 2),
    )

    s_h = 4
    surround = np.mean(
        X_full[:, CENTER_Y - s_h:CENTER_Y + s_h + 1,
               CENTER_X - s_h:CENTER_X + s_h + 1],
        axis=(1, 2),
    )

    contrast_on = center - surround
    contrast_off = surround - center

    contrast_on = (contrast_on - np.mean(contrast_on)) / (np.std(contrast_on) + 1e-6)
    contrast_off = (contrast_off - np.mean(contrast_off)) / (np.std(contrast_off) + 1e-6)

    return contrast_on, contrast_off, Y_full.astype(np.float64)


def optimize_polarity(stim_train, y_train, label):
    """Run differential_evolution for a single polarity and return the result."""
    print(f"  Optimizing {label} polarity...")
    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        args=(stim_train, y_train),
        seed=42,
        maxiter=200,
        tol=1e-6,
        polish=True,
    )
    best_corr = -result.fun
    tau, gain, bias = result.x
    print(f"  {label}: Pearson r = {best_corr:.4f}  "
          f"(tau={tau:.4f}, gain={gain:.4f}, bias={bias:.4f})")
    return result


def run_training():
    stim_on, stim_off, spikes = load_data_center_surround()
    print(f"Loaded {len(stim_on)} frames.")

    stim_on_train, stim_on_test = stim_on[:TRAIN_END], stim_on[TRAIN_END:TEST_END]
    stim_off_train, stim_off_test = stim_off[:TRAIN_END], stim_off[TRAIN_END:TEST_END]
    y_train, y_test = spikes[:TRAIN_END], spikes[TRAIN_END:TEST_END]

    print(f"Train: {len(y_train)} frames | Test: {len(y_test)} frames")
    print("Running differential_evolution on both polarities...")

    res_on = optimize_polarity(stim_on_train, y_train, "ON")
    res_off = optimize_polarity(stim_off_train, y_train, "OFF")

    if -res_off.fun > -res_on.fun:
        best_pol, best_res, best_stim_test = "OFF", res_off, stim_off_test
    else:
        best_pol, best_res, best_stim_test = "ON", res_on, stim_on_test

    tau, gain, bias = best_res.x
    print(f"\nBest polarity: {best_pol}-Cell")
    print(f"Parameters: tau={tau:.4f}, gain={gain:.4f}, bias={bias:.4f}")

    y_pred = simulate_ode(best_stim_test, tau, gain, bias, DT)

    if np.std(y_pred) < 1e-9 or np.std(y_test) < 1e-9:
        pcc = 0.0
    else:
        pcc = np.corrcoef(y_pred, y_test)[0, 1]
        if np.isnan(pcc):
            pcc = 0.0

    print(f"Test Pearson Correlation: {pcc:.4f}")

    plt.figure(figsize=(14, 5))
    plt.plot(y_test[:300], 'k', linewidth=1, label='Ground Truth (spikes)')
    plt.plot(y_pred[:300], 'orange', linewidth=1, label='ODE Prediction')
    plt.title(f"Retinomorphic ODE ({best_pol}-Cell) | Pearson r = {pcc:.4f}")
    plt.xlabel("Test Sample")
    plt.ylabel("Firing Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_training()
