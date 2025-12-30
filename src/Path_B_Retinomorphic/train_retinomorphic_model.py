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

# --- הגדרות ---
DATA_FILE = os.path.join(BASE_PATH, 'processed_data', 'training_dataset_ns_full.h5')
CENTER_X = 27
CENTER_Y = 47
DT = 1.0 / 60.0 

print("🚀 Initializing Retinomorphic Simulator (Sustained Center-Surround)...")

class SustainedCell:
    def __init__(self, tau=0.05, gain=10.0, bias=-2.0, noise=2.0):
        self.tau = tau          # קבוע זמן אינטגרציה (Low Pass)
        self.gain = gain        # הגבר
        self.bias = bias        # סף הפעלה (שלילי = צריך קלט חזק כדי לירות)
        self.noise = noise
        self.y = 0.0            # מצב פנימי

    # משוואת Leaky Integrator פשוטה
    # התא צובר את הקלט לאורך זמן, אבל דולף (Sustained)
    def step_rk4(self, I_in, dt):
        # dy/dt = (Input - y) / tau
        # RK4 Implementation
        def derivative(y, I): return (I - y) / self.tau
        
        k1 = derivative(self.y, I_in)
        k2 = derivative(self.y + 0.5 * dt * k1, I_in)
        k3 = derivative(self.y + 0.5 * dt * k2, I_in)
        k4 = derivative(self.y + dt * k3, I_in)
        
        self.y = self.y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # המרה לקצב ירי: Softplus + Noise
        # Softplus הוא חלק יותר מ-ReLU ומתאים יותר לביולוגיה
        val = self.gain * self.y + self.bias
        rate = np.log(1 + np.exp(val)) # Softplus function
        
        # הוספת רעש אקראי (Poisson-like variability)
        rate += np.random.normal(0, self.noise)
        
        return max(0, rate)

def load_data_center_surround():
    with h5py.File(DATA_FILE, 'r') as f:
        X_full = f['X'][:]
        Y_full = f['Y'][:]
    
    # חישוב Center (3x3)
    c_h = 1
    center = np.mean(X_full[:, CENTER_Y-c_h:CENTER_Y+c_h+1, CENTER_X-c_h:CENTER_X+c_h+1], axis=(1,2))
    
    # חישוב Surround (9x9)
    s_h = 4
    surround = np.mean(X_full[:, CENTER_Y-s_h:CENTER_Y+s_h+1, CENTER_X-s_h:CENTER_X+s_h+1], axis=(1,2))
    
    # חישוב ה-Contrast (זה מבטל את ה-DC הכללי ומשאיר את האות המקומי)
    # נחזיר את שתי האופציות: ON (מרכז בהיר) ו-OFF (מרכז כהה)
    contrast_on = center - surround
    contrast_off = surround - center # הפוך
    
    # נרמול ל-Z-Score (קריטי כדי שהמודל יעבוד בטווחים נכונים)
    contrast_on = (contrast_on - np.mean(contrast_on)) / (np.std(contrast_on) + 1e-6)
    contrast_off = (contrast_off - np.mean(contrast_off)) / (np.std(contrast_off) + 1e-6)
    
    return contrast_on, contrast_off, Y_full

def run_simulation():
    stim_on, stim_off, spikes = load_data_center_surround()
    print(f"Loaded {len(stim_on)} frames. Testing ON vs OFF polarity...")
    
    # הרצת סימולציה כפולה לבדיקת קוטביות
    traces = {}
    corrs = {}
    
    for polarity, stim in [("ON", stim_on), ("OFF", stim_off)]:
        cell = SustainedCell(tau=0.1, gain=8.0, bias=-1.0, noise=1.5)
        trace = []
        for t in range(len(stim)):
            trace.append(cell.step_rk4(stim[t], DT))
        
        trace = np.array(trace)
        pcc = np.corrcoef(trace, spikes)[0, 1]
        traces[polarity] = trace
        corrs[polarity] = pcc
        print(f"   👉 {polarity} Model Correlation: {pcc:.4f}")

    # בחירת המנצח
    best_pol = "OFF" if corrs["OFF"] > corrs["ON"] else "ON"
    best_trace = traces[best_pol]
    best_corr = corrs[best_pol]
    
    print(f"\n🏆 Winner: {best_pol}-Cell with Correlation = {best_corr:.4f}")
    
    # הצגה גרפית
    plt.figure(figsize=(12, 6))
    s, e = 200, 400
    
    plt.subplot(2,1,1)
    plt.plot(spikes[s:e], 'k', label='Biology (GT)')
    plt.plot(best_trace[s:e], 'b', alpha=0.7, label=f'Model ({best_pol} Sustained)')
    plt.title(f"Final Retinomorphic Model | Type: {best_pol} | Corr: {best_corr:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.subplot(2,1,2)
    # נציג את הקלט של המודל המנצח
    input_signal = stim_off if best_pol == "OFF" else stim_on
    plt.plot(input_signal[s:e], 'gray', linestyle='--', label=f'{best_pol} Input (Normalized)')
    plt.legend()
    plt.title("Input to ODE (Contrast)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()