import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                              recall_score, f1_score,
                              confusion_matrix, classification_report)

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
        return

    # Step 1: Load and clean
    df = load_and_clean_data(dataset_path)
    label_col = 'Label'

    # Step 2: Encode labels
    df = encode_labels(df, label_col)

    # Step 3: Separate features and target
    X_raw = df.drop(columns=[label_col, 'encoded_label'])
    y_raw = df['encoded_label']
    feature_names = X_raw.columns.tolist()

    print(f"\nClass distribution before split:")
    print(y_raw.value_counts())

    # Step 4: Split FIRST before any resampling (critical fix)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"\nTrain size: {X_train_raw.shape[0]}, Test size: {X_test_raw.shape[0]}")

    # Step 5: Apply SMOTE only on training data
    X_train_res, y_train_res = handle_imbalance(X_train_raw, y_train)

    # Step 6: Feature selection using only training data
    top_indices, top_features = optimize_features(
        X_train_res, y_train_res, feature_names, top_n=20
    )

    # Step 7: Apply feature selection to both train and test
    # Robust handling: X_train_res could be numpy array or DataFrame after SMOTE
    if hasattr(X_train_res, 'iloc'):
        # It's a DataFrame - use pandas indexing
        X_train_selected = X_train_res.iloc[:, top_indices].values
    else:
        # It's a numpy array - use numpy indexing
        X_train_selected = X_train_res[:, top_indices]
    
    X_test_selected = X_test_raw.iloc[:, top_indices].values

    # Step 8: Normalize (fit on train, transform both)
    X_train_scaled, X_test_scaled, scaler = normalize_features(
        X_train_selected, X_test_selected
    )

    # Step 9: Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(top_features, 'models/top_features.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    # Save thresholds to ensure inference matches training
    LR_THRESHOLD = 0.3
    RF_THRESHOLD = 0.45
    joblib.dump(LR_THRESHOLD, 'models/lr_threshold.pkl')
    joblib.dump(RF_THRESHOLD, 'models/rf_threshold.pkl')

    print("\n--- Training Hybrid Model Components ---")

    # Step 10a: Train Logistic Regression
    print("Training Logistic Regression (Stage 1: Fast Filter)...")
    lr_model = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train_res)
    joblib.dump(lr_model, 'models/lr_model.pkl')
    print("LR training complete.")

    # Step 10b: Train Random Forest
    print("Training Random Forest (Stage 2: Expert Classifier)...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    rf_model.fit(X_train_scaled, y_train_res)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("RF training complete.")

    print("\n--- Evaluation on Test Data (Unseen) ---")

    # Step 11: Evaluate using correct hybrid logic matching inference
    # Use predict_proba for LR with confidence threshold (thresholds saved with models)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_preds = (rf_proba >= joblib.load('models/rf_threshold.pkl')).astype(int)

    hybrid_preds = []
    model_used_log = []

    # Load LR threshold saved earlier to ensure inference matches training
    lr_threshold = joblib.load('models/lr_threshold.pkl')

    for i, prob in enumerate(lr_proba):
        if prob < lr_threshold:
            # LR is confident it is BENIGN
            hybrid_preds.append(0)
            model_used_log.append("LR")
        else:
            # LR suspects attack, defer to RF
            hybrid_preds.append(rf_preds[i])
            model_used_log.append("RF")

    hybrid_preds = np.array(hybrid_preds)

    print("\n===== HYBRID MODEL RESULTS =====")
    print(f"Accuracy:  {accuracy_score(y_test, hybrid_preds):.4f}")
    print(f"Precision: {precision_score(y_test, hybrid_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, hybrid_preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, hybrid_preds):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, hybrid_preds)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (correctly benign): {tn}")
    print(f"False Positives (benign flagged as attack): {fp}")
    print(f"False Negatives (attacks missed): {fn}")
    print(f"True Positives (correctly detected attacks): {tp}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, hybrid_preds, 
                                 target_names=['BENIGN', 'ATTACK']))

    lr_count = model_used_log.count("LR")
    rf_count = model_used_log.count("RF")
    print(f"\nHybrid routing: LR handled {lr_count} samples, RF handled {rf_count} samples")
    print(f"LR fast-path efficiency: {lr_count/len(model_used_log)*100:.1f}% of traffic handled instantly")

    print("\nAll models saved to models/ directory.")
    print("You can now run: python app.py and python realtime.py")

if __name__ == "__main__":
    run_training('dataset.csv')