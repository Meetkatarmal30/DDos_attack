import time
import pandas as pd
import numpy as np
import hashlib
import sqlite3
import datetime
import os
from hybrid_model import HybridIDS

DB_PATH = 'dashboard.db'
LOG_FILE = 'alerts.log'
SLEEP_SECONDS = 0.3  # Reduced from 1.0 for better demo experience

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stats
                 (id INTEGER PRIMARY KEY, 
                  total_processed INTEGER, 
                  total_attacks INTEGER)''')
    c.execute('SELECT COUNT(*) FROM stats')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO stats VALUES (1, 0, 0)')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  prediction TEXT,
                  model_used TEXT,
                  hash_val TEXT,
                  shap_info TEXT)''')
    c.execute("UPDATE stats SET total_processed=0, total_attacks=0 WHERE id=1")
    c.execute("DELETE FROM predictions")
    conn.commit()
    return conn

def generate_hash(timestamp, prediction, model_used, shap_info):
    content = f"{timestamp}|{prediction}|{model_used}|{shap_info}"
    return hashlib.sha256(content.encode()).hexdigest()

def append_to_log(timestamp, prediction, model_used, shap_info, hash_val):
    entry = (f"[{timestamp}] PREDICTION={prediction} | "
             f"MODEL={model_used} | "
             f"SHAP={shap_info} | "
             f"INTEGRITY_HASH={hash_val}\n")
    with open(LOG_FILE, 'a') as f:
        f.write(entry)

def simulate_realtime(dataset_path='dataset.csv'):
    print("="*60)
    print("Hybrid IDS — Real-Time Simulation Starting")
    print("="*60)

    ids = HybridIDS()
    if not ids.ready:
        print("Cannot start: models not loaded.")
        return

    try:
        df = pd.read_csv(dataset_path)
        # Shuffle so we don't get all same-class rows at start
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Loaded {len(df):,} rows from {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    conn = setup_db()
    c = conn.cursor()

    print(f"\nDashboard: http://127.0.0.1:5000")
    print("Streaming rows... Press Ctrl+C to stop.\n")

    processed = 0
    attacks = 0

    try:
        for index, row in df.iterrows():
            packet = row.to_dict()
            pred_code, label, model_used, shap_info = ids.predict(packet)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            processed += 1
            if label == "ATTACK":
                attacks += 1
                print(f"[ALERT]  {timestamp} | Model: {model_used:20s} | {shap_info[:60]}")
            else:
                print(f"[NORMAL] {timestamp} | Model: {model_used:20s} | Benign traffic")

            # Generate integrity hash (now includes SHAP info)
            hash_val = generate_hash(timestamp, label, model_used, shap_info)
            append_to_log(timestamp, label, model_used, shap_info, hash_val)

            # Update database
            c.execute(
                "UPDATE stats SET total_processed=?, total_attacks=? WHERE id=1",
                (processed, attacks)
            )
            c.execute(
                """INSERT INTO predictions 
                   (timestamp, prediction, model_used, hash_val, shap_info) 
                   VALUES (?, ?, ?, ?, ?)""",
                (timestamp, label, model_used, hash_val, shap_info)
            )
            c.execute(
                """DELETE FROM predictions WHERE id NOT IN 
                   (SELECT id FROM predictions ORDER BY id DESC LIMIT 15)"""
            )
            conn.commit()

            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\nSimulation stopped.")
        print(f"Total processed: {processed:,}")
        print(f"Total attacks detected: {attacks:,}")
        print(f"Attack rate: {attacks/processed*100:.1f}%" if processed > 0 else "")
    finally:
        conn.close()

if __name__ == "__main__":
    simulate_realtime()