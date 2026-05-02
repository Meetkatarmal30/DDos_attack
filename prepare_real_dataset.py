import pandas as pd
import numpy as np

print("="*60)
print("Loading Real CIC-IDS Dataset...")
print("="*60)

# Load the real dataset
df = pd.read_parquet('cic-collection.parquet')
print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Sample to manageable size (50,000 rows)
print("\nSampling dataset to 50,000 rows...")
df_sampled = df.sample(n=50000, random_state=42)
print(f"✓ Sampled: {df_sampled.shape[0]:,} rows")

# Convert labels: All attacks → "ATTACK", Benign → "BENIGN"
print("\nConverting labels...")
def convert_label(label):
    if label.lower() == 'benign':
        return 'BENIGN'
    else:
        return 'ATTACK'

df_sampled['Label'] = df_sampled['Label'].apply(convert_label)
print("✓ Label distribution:")
print(df_sampled['Label'].value_counts())

# Select relevant columns that exist in the real dataset
available_columns = df_sampled.columns.tolist()
required_columns = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
    'Fwd Packets Length Total', 'Bwd Packets Length Total', 
    'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Fwd Header Length', 'Bwd Header Length',
    'Label'
]

# Filter to only columns that exist
required_columns = [col for col in required_columns if col in available_columns]

# Create output dataframe with required columns
df_output = df_sampled[required_columns].copy()

# Rename columns to match original project expectations
column_mapping = {
    'Fwd Packets Length Total': 'Total Length of Fwd Packets',
    'Bwd Packets Length Total': 'Total Length of Bwd Packets',
}

df_output = df_output.rename(columns=column_mapping)

print(f"\n✓ Selected {len(df_output.columns)} columns")
print("\nColumns in output:")
for i, col in enumerate(df_output.columns, 1):
    print(f"  {i}. {col}")

# Check for missing values
missing = df_output.isnull().sum().sum()
print(f"\n✓ Missing values: {missing}")

# Save as CSV
output_file = 'dataset.csv'
df_output.to_csv(output_file, index=False)
print(f"\n{'='*60}")
print(f"✓ Real dataset saved to: {output_file}")
print(f"✓ Rows: {df_output.shape[0]:,}")
print(f"✓ Columns: {df_output.shape[1]}")
print(f"{'='*60}")
print("\nDataset ready! You can now run:")
print("  1. python train_model.py")
print("  2. python app.py (in terminal 1)")
print("  3. python realtime.py (in terminal 2)")
