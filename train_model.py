import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import os

from preprocess import (
    load_and_clean_data, 
    encode_labels, 
    normalize_features, 
    handle_imbalance, 
    optimize_features
)

def run_training(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        print("Please run `generate_data.py` first to create a mock dataset.")
        return

    # 1. Load and clean
    df = load_and_clean_data(dataset_path)
    label_col = 'Label'
    
    # 2. Encode
    df = encode_labels(df, label_col)
    
    # 3. Separate features and target
    X_raw = df.drop(columns=[label_col, 'encoded_label'])
    y_raw = df['encoded_label']
    feature_names = X_raw.columns.tolist()
    
    # 4. Handle Imbalance with SMOTE
    X_res, y_res = handle_imbalance(X_raw, y_raw)
    
    # 5. Optimize Features (Train RF for importance)
    top_indices, top_features = optimize_features(X_res, y_res, feature_names)
    X_optimized = X_res.iloc[:, top_indices]
    
    # Save selected feature indices/names
    os.makedirs('models', exist_ok=True)
    joblib.dump(top_features, 'models/top_features.pkl')
    
    # 6. Normalize
    X_scaled, scaler = normalize_features(X_optimized)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.3, random_state=42)
    
    print("\n--- Training Hybrid Model Components ---")
    
    # 8a. Train Logistic Regression (Stage 1)
    print("Training Logistic Regression (Stage 1: Fast Filter)...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'models/lr_model.pkl')
    
    # 8b. Train Random Forest (Stage 2)
    print("Training Random Forest (Stage 2: Final Classifier)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    
    print("\n--- Evaluation on Test Data ---")
    # Simulate hybrid prediction to evaluate
    print("Simulating Hybrid Prediction Evaluation...")
    hybrid_preds = []
    
    # For evaluation, we predict all using LR and RF separately, then combine logically
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    # Hybrid logic: If LR predicts 0 (Normal), return Normal. Else use RF.
    for lr_p, rf_p in zip(lr_preds, rf_preds):
        if lr_p == 0:
            hybrid_preds.append(0)
        else:
            hybrid_preds.append(rf_p)
            
    print("\n===== Hybrid Model Results =====")
    print(f"Accuracy:  {accuracy_score(y_test, hybrid_preds):.4f}")
    print(f"Precision: {precision_score(y_test, hybrid_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, hybrid_preds):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, hybrid_preds))
    
    print("\n--- Explanation of Hybrid Results ---")
    print("1. Logistic Regression acts as a fast preliminary filter. By quickly classifying obvious regular traffic as NORMAL, it saves computational time.")
    print("2. The Random Forest acts as an expert. If Logistic Regression suspects it might be malicious, the Random Forest double checks with a deeper non-linear analysis.")
    print("3. This combination achieves high real-time throughput while keeping misclassifications low.")
    print("\nModels successfully saved in 'models/' directory.")

if __name__ == "__main__":
    run_training('dataset.csv')
