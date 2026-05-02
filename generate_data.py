import pandas as pd
import numpy as np

def load_real_dataset(parquet_path='cic-collection.parquet', output_filename='dataset.csv', sample_size=50000):
    """
    Load real CIC-IDS data from parquet and save a sample to dataset.csv for training
    """
    print(f"Loading real data from {parquet_path}...")
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
    
    # Sample if dataset is too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for training")
    
    # Find and standardize Label column
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        print("ERROR: No Label column found in dataset")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Rename to 'Label' if different
    if label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
        print(f"Renamed '{label_col}' to 'Label'")
    
    print(f"\nFinal data shape: {df.shape}")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Real dataset saved to {output_filename}")

# def generate_mock_dataset(filename='dataset.csv', num_samples=3000):
#     """
#     Generates a mock dataset resembling CICIDS2017 feature structures.
#     We create generic statistical distributions to mimic networking data.
#     """
#     print(f"Generating mock dataset with {num_samples} samples...")
#     np.random.seed(42)
    
#     # Mock CICIDS2017-like feature names
#     features = [f'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
#                 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
#                 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
#                 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
#                 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
#                 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
#                 'Bwd IAT Total', 'Bwd IAT Mean', 'Fwd Header Length', 'Bwd Header Length']
    
#     data = []
#     labels = []
    
#     for _ in range(num_samples):
#         # 80% benign traffic, 20% attack traffic
#         is_attack = np.random.rand() > 0.8
        
#         row = []
#         for feature in features:
#             if is_attack:
#                 # Malicious anomalies typically have different traffic distributions
#                 row.append(np.random.normal(loc=150.0, scale=80.0))
#             else:
#                 row.append(np.random.normal(loc=15.0, scale=5.0))
#         data.append(row)
        
#         labels.append("ATTACK" if is_attack else "BENIGN")
        
#     df = pd.DataFrame(data, columns=features)
#     df['Label'] = labels
    
#     df.to_csv(filename, index=False)
#     print(f"Success! Mock dataset saved to {filename}")

if __name__ == "__main__":
    # Load REAL data from the CIC parquet file
    load_real_dataset('cic-collection.parquet')
