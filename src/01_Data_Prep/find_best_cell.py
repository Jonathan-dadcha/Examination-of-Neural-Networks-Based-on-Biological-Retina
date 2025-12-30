import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
from config import BASE_PATH, SESSION
SPIKES_DIR = os.path.join(BASE_PATH, SESSION, 'spiketimes')

def find_best_cell():
    print(f"Scanning session: {SESSION}")
    
    # Look for White Noise spikes (starts with 5_)
    files = [f for f in os.listdir(SPIKES_DIR) if f.startswith('5_SP')]
    
    cell_stats = []
    
    for f in files:
        path = os.path.join(SPIKES_DIR, f)
        try:
            spikes = pd.read_csv(path, header=None)[0].values
            count = len(spikes)
            cell_id = f.split('_')[2].replace('.txt', '')
            
            cell_stats.append({'id': cell_id, 'count': count, 'file': f})
        except Exception:
            continue
            
    # Sort by spike count (descending)
    df = pd.DataFrame(cell_stats).sort_values('count', ascending=False)
    
    print("\n--- TOP 5 CELLS (Most Spikes) ---")
    print(df.head(5))
    
    # Plot histogram of firing rates
    plt.figure(figsize=(8, 4))
    plt.hist(df['count'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Spikes')
    plt.ylabel('Number of Cells')
    plt.title('Distribution of Spike Counts (White Noise)')
    plt.show()
    
    return df.iloc[0]['file']

if __name__ == "__main__":
    best_file = find_best_cell()
    print(f"\n✅ RECOMMENDATION: Use cell file '{best_file}' in data_loader.py")