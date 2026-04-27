import time
import pandas as pd
import hashlib
import sqlite3
import datetime
import os
from hybrid_model import HybridIDS

DB_PATH = 'dashboard.db'
LOG_FILE = 'alerts.log'

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stats
                 (id INTEGER PRIMARY KEY, total_processed INTEGER, total_attacks INTEGER)''')
    
    # Initialize stats if empty
    c.execute('SELECT COUNT(*) FROM stats')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO stats (id, total_processed, total_attacks) VALUES (1, 0, 0)')
        
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  prediction TEXT, 
                  model_used TEXT, 
                  hash_val TEXT)''')
    
    # Clear old data for a fresh real-time run
    c.execute("UPDATE stats SET total_processed=0, total_attacks=0 WHERE id=1")
    c.execute("DELETE FROM predictions")
    
    conn.commit()
    return conn

def append_to_log(timestamp, prediction, model_used):
    """Saves alert to log and generates SHA256 hash for integrity."""
    log_entry = f"{timestamp} | Prediction: {prediction} | Model: {model_used}"
    hash_val = hashlib.sha256(log_entry.encode()).hexdigest()
    final_entry = f"{log_entry} | Hash: {hash_val}\n"
    
    with open(LOG_FILE, 'a') as f:
        f.write(final_entry)
        
    return hash_val

def simulate_realtime(dataset_path='dataset.csv'):
    print("Initializing Real-Time Simulation Dashboard...")
    ids = HybridIDS()
    if not ids.ready:
        return
        
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading {dataset_path}: {e}")
        return

    conn = setup_db()
    c = conn.cursor()
    
    print("\nSystem Online. Starting stream... Press Ctrl+C to stop.\n")
    processed_count = 0
    attack_count = 0
    
    try:
        # Loop through dataset simulating real-time packet arrival
        for index, row in df.iterrows():
            packet_data = row.to_dict()
            
            # Predict using Hybrid Model
            pred_code, label, model_used = ids.predict(packet_data)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            processed_count += 1
            if label == "ATTACK":
                attack_count += 1
                print(f"[ALERT] {timestamp} | Attack Detected! | Used: {model_used}")
            else:
                print(f"[NORMAL] {timestamp} | Traffic Benign   | Used: {model_used}")
                
            # Log for Alert System Integrity Tracking
            hash_val = append_to_log(timestamp, label, model_used)
            
            # Update SQLite DB for Local Flask dashboard synchronization
            c.execute("UPDATE stats SET total_processed=?, total_attacks=? WHERE id=1", 
                      (processed_count, attack_count))
            
            c.execute("INSERT INTO predictions (timestamp, prediction, model_used, hash_val) VALUES (?, ?, ?, ?)",
                      (timestamp, label, model_used, hash_val))
            
            # Sub-table capping to keep fetch times short for visualization
            c.execute("DELETE FROM predictions WHERE id NOT IN (SELECT id FROM predictions ORDER BY id DESC LIMIT 15)")
            
            conn.commit()
            
            # Network Processing Delay Delay (0.5 to 1 second)
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nReal-Time Simulation terminated gracefully.")
    finally:
        conn.close()

if __name__ == "__main__":
    simulate_realtime()
