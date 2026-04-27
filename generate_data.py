import pandas as pd
import numpy as np

def generate_mock_dataset(filename='dataset.csv', num_samples=3000):
    """
    Generates a mock dataset resembling CICIDS2017 feature structures.
    We create generic statistical distributions to mimic networking data.
    """
    print(f"Generating mock dataset with {num_samples} samples...")
    np.random.seed(42)
    
    # Mock CICIDS2017-like feature names
    features = [f'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
                'Bwd IAT Total', 'Bwd IAT Mean', 'Fwd Header Length', 'Bwd Header Length']
    
    data = []
    labels = []
    
    for _ in range(num_samples):
        # 80% benign traffic, 20% attack traffic
        is_attack = np.random.rand() > 0.8
        
        row = []
        for feature in features:
            if is_attack:
                # Malicious anomalies typically have different traffic distributions
                row.append(np.random.normal(loc=150.0, scale=80.0))
            else:
                row.append(np.random.normal(loc=15.0, scale=5.0))
        data.append(row)
        
        labels.append("ATTACK" if is_attack else "BENIGN")
        
    df = pd.DataFrame(data, columns=features)
    df['Label'] = labels
    
    df.to_csv(filename, index=False)
    print(f"Success! Mock dataset saved to {filename}")

if __name__ == "__main__":
    generate_mock_dataset()
