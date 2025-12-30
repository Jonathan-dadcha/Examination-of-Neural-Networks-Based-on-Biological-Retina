import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# הוספת התיקייה שמעל (src) לנתיב
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import BASE_PATH
except ImportError:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CENTER_X = 27
CENTER_Y = 47
DT = 1.0 / 60.0 

print("🚀 Starting Hyperparameter Optimization for Path B...")

class SustainedCell:
    def __init__(self, tau, gain, bias, noise):
        self.tau = tau
        self.gain = gain
        self.bias = bias
        self.noise = noise
        self.y = 0.0

    def step_rk4(self, I_in, dt):
        def derivative(y, I): return (I - y) / self.tau
        
        k1 = derivative(self.y, I_in)
        k2 = derivative(self.y + 0.5 * dt * k1, I_in)
        k3 = derivative(self.y + 0.5 * dt * k2, I_in)
        k4 = derivative(self.y + dt * k3, I_in)
        
        self.y = self.y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Softplus activation
        val = self.gain * self.y + self.bias
        rate = np.log(1 + np.exp(val)) 
        # Add noise but clip at 0
        rate += np.random.normal(0, self.noise)
        return max(0, rate)

def load_data():
    with h5py.File(DATA_FILE, 'r') as f:
        X_full = f['X'][:]
        Y_full = f['Y'][:]
    
    # Center-Surround Calculation (OFF polarity)
    c_h = 1 # Center Radius
    s_h = 4 # Surround Radius
    
    center = np.mean(X_full[:, CENTER_Y-c_h:CENTER_Y+c_h+1, CENTER_X-c_h:CENTER_X+c_h+1], axis=(1,2))
    surround = np.mean(X_full[:, CENTER_Y-s_h:CENTER_Y+s_h+1, CENTER_X-s_h:CENTER_X+s_h+1], axis=(1,2))
    
    # OFF Cell Logic: Surround - Center
    contrast = (surround - center)
    contrast = (contrast - np.mean(contrast)) / (np.std(contrast) + 1e-6)
    
    return contrast, Y_full

def optimize():
    stimulus, spikes = load_data()
    print(f"Loaded {len(stimulus)} frames. Searching for best parameters...")
    
    best_corr = -1.0
    best_params = {}
    best_trace = None
    
    # --- הגדרת טווחי החיפוש ---
    num_trials = 50  # מספר הניסיונות (אפשר להגדיל אם יש זמן)
    
    for i in range(num_trials):
        # הגרלת פרמטרים אקראית
        p_tau = np.random.uniform(0.01, 0.5)    # מהירות תגובה
        p_gain = np.random.uniform(1.0, 50.0)   # רגישות
        p_bias = np.random.uniform(-5.0, 0.0)   # סף (חשוב לספארסיות!)
        p_noise = np.random.uniform(0.0, 3.0)   # רעש
        
        # הרצת סימולציה
        cell = SustainedCell(p_tau, p_gain, p_bias, p_noise)
        trace = []
        for t in range(len(stimulus)):
            trace.append(cell.step_rk4(stimulus[t], DT))
        
        trace = np.array(trace)
        
        # בדיקת קורלציה
        # (מונעים קריסה אם הכל אפסים)
        if np.std(trace) < 1e-9:
            curr_corr = 0.0
        else:
            curr_corr = np.corrcoef(trace, spikes)[0, 1]
            if np.isnan(curr_corr): curr_corr = 0.0
        
        print(f"Trial {i+1}/{num_trials}: Corr={curr_corr:.4f} | tau={p_tau:.2f}, bias={p_bias:.2f}")
        
        # שמירת הטוב ביותר
        if curr_corr > best_corr:
            best_corr = curr_corr
            best_params = {'tau': p_tau, 'gain': p_gain, 'bias': p_bias, 'noise': p_noise}
            best_trace = trace
            print(f"   🌟 New Best found! Corr: {best_corr:.4f}")

    print("\n" + "="*40)
    print(f"🏆 Optimization Complete.")
    print(f"Best Correlation: {best_corr:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*40)
    
    # שמירת התוצאה הטובה ביותר לגרף
    plt.figure(figsize=(12, 6))
    s, e = 200, 500
    
    plt.subplot(2,1,1)
    plt.plot(spikes[s:e], 'k', label='Biology (GT)')
    plt.plot(best_trace[s:e], 'g', alpha=0.8, label=f'Best ODE (Corr: {best_corr:.3f})')
    plt.title(f"Optimized Retinomorphic Model\nParams: {best_params}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2,1,2)
    plt.plot(stimulus[s:e], 'gray', linestyle='--', label='Input Contrast')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    optimize()