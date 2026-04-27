import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def load_and_clean_data(filepath):
    """Loads data and handles missing values."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Original data shape: {df.shape}")
    
    # Handle missing values (Fill with median for numerical columns)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def encode_labels(df, label_col='Label'):
    """Encodes labels (BENIGN=0, ATTACK=1)."""
    df['encoded_label'] = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    return df

def normalize_features(X):
    """Normalizes features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def handle_imbalance(X, y):
    """Uses SMOTE to handle class imbalance."""
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Data shape after SMOTE: {X_res.shape}")
    return X_res, y_res

def optimize_features(X, y, feature_names):
    """
    Uses Random Forest feature importance to select the top features.
    Explanation: We use Random Forest because it is robust, handles non-linear relationships well,
    and provides highly interpretable feature importance scores. This helps in real-time scenarios
    by reducing the number of features to compute and monitor, making the system faster.
    """
    print("Optimizing features using Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    # Select top 15 features for efficiency
    top_indices = importances.argsort()[-15:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    print(f"Top {len(top_features)} selected features: {top_features}")
    
    return top_indices, top_features
