import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Original data shape: {df.shape}")
    
    # Keep only numeric columns (except Label column which will be handled separately)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = 'Label' if 'Label' in df.columns else None
    
    if label_col:
        cols_to_keep = numeric_cols + [label_col]
    else:
        cols_to_keep = numeric_cols
    
    df = df[cols_to_keep]
    print(f"Data shape after keeping numeric columns: {df.shape}")
    
    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    return df

def encode_labels(df, label_col='Label'):
    df['encoded_label'] = df[label_col].apply(
        lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
    )
    return df

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def handle_imbalance(X_train, y_train):
    print("Applying SMOTE only on training data...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Training data shape after SMOTE: {X_res.shape}")
    print(f"Class distribution after SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res

def optimize_features(X_train_resampled, y_train_resampled, feature_names, top_n=20):
    print(f"Selecting top {top_n} features using Random Forest importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)
    importances = rf.feature_importances_
    top_indices = importances.argsort()[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    print(f"Selected features: {top_features}")
    return top_indices, top_features