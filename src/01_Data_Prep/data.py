import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from config import BASE_PATH


def scan_sessions(base_path):
    sessions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for session in sessions:
        list_file = os.path.join(base_path, session, 'list_of_good_cells.txt')
        if os.path.exists(list_file):
            try:
                df = pd.read_csv(list_file, sep='\t', header=None) 
                print(f"Session {session}: Found {len(df)} good cells.")
            except Exception as e:
                print(f"Session {session}: Could not read list ({e})")

scan_sessions(BASE_PATH)