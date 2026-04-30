import joblib
import os
import warnings
import numpy as np
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore", category=UserWarning)

LR_THRESHOLD = 0.4  # Lowered to improve recall; must match train_model.py

class HybridIDS:
    def __init__(self, models_dir='models'):
        try:
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            self.top_features = joblib.load(os.path.join(models_dir, 'top_features.pkl'))
            self.lr_model = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
            self.ready = True
            print("Models loaded successfully.")
            
            # Initialize SHAP explainer on RF
            self.explainer = shap.TreeExplainer(self.rf_model)
            print("SHAP explainer ready.")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Please run train_model.py first.")
            self.ready = False

    def predict(self, data_dict):
        if not self.ready:
            return 0, "ERROR", "None", ""

        # Extract top features in correct order
        try:
            features = np.array(
                [[float(data_dict.get(f, 0.0)) for f in self.top_features]]
            )
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return 0, "ERROR", "None", ""

        # Scale using saved scaler
        try:
            scaled_features = self.scaler.transform(features)
        except Exception as e:
            print(f"Scaling error: {e}")
            return 0, "ERROR", "None", ""

        # Stage 1: LR with confidence threshold
        try:
            lr_prob_attack = self.lr_model.predict_proba(scaled_features)[0][1]
        except Exception as e:
            print(f"LR prediction error: {e}")
            return 0, "ERROR", "None", ""

        if lr_prob_attack < LR_THRESHOLD:
            # LR confident it is benign, fast return
            return 0, "BENIGN", "Logistic Regression", ""

        # Stage 2: RF expert decision
        try:
            rf_pred = self.rf_model.predict(scaled_features)[0]
        except Exception as e:
            print(f"RF prediction error: {e}")
            return 0, "ERROR", "None", ""

        label = "ATTACK" if rf_pred == 1 else "BENIGN"

        # Generate SHAP explanation only for confirmed attacks
        shap_info = ""
        if label == "ATTACK":
            try:
                shap_values = self.explainer.shap_values(scaled_features)

                # Handle both old and new SHAP output formats
                if isinstance(shap_values, list):
                    # Old format: list of arrays, index 1 = attack class
                    attack_shap = shap_values[1][0]
                elif hasattr(shap_values, 'values'):
                    # New Explanation object format
                    vals = shap_values.values
                    if len(vals.shape) == 3:
                        attack_shap = vals[0, :, 1]
                    else:
                        attack_shap = vals[0]
                else:
                    # Fallback numpy array
                    if len(shap_values.shape) == 3:
                        attack_shap = shap_values[0, :, 1]
                    else:
                        attack_shap = shap_values[0]

                feature_impacts = list(zip(self.top_features, attack_shap))
                feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                top3 = feature_impacts[:3]
                shap_info = "SHAP: " + ", ".join(
                    [f"{feat} ({val:+.3f})" for feat, val in top3]
                )
            except Exception as e:
                shap_info = f"SHAP unavailable: {str(e)[:50]}"

        return int(rf_pred), label, "Random Forest", shap_info

    def evaluate(self, X_test, y_test):
        """Evaluate hybrid model accuracy on test dataset"""
        if not self.ready:
            print("ERROR: Models not loaded")
            return

        try:
            features = self.scaler.transform(X_test)
        except Exception as e:
            print(f"Scaling error: {e}")
            return

        # Get predictions
        lr_proba = self.lr_model.predict_proba(features)[:, 1]
        rf_preds = self.rf_model.predict(features)

        # Apply hybrid logic
        hybrid_preds = []
        for prob in lr_proba:
            if prob < LR_THRESHOLD:
                hybrid_preds.append(0)
            else:
                hybrid_preds.append(rf_preds[len(hybrid_preds)])

        hybrid_preds = np.array(hybrid_preds)

        # Calculate metrics
        acc = accuracy_score(y_test, hybrid_preds)
        prec = precision_score(y_test, hybrid_preds, zero_division=0)
        rec = recall_score(y_test, hybrid_preds, zero_division=0)
        f1 = f1_score(y_test, hybrid_preds, zero_division=0)

        print("\n" + "="*50)
        print("HYBRID MODEL ACCURACY EVALUATION")
        print("="*50)
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        cm = confusion_matrix(y_test, hybrid_preds)
        print("\nConfusion Matrix:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Negatives:  {tn} (correctly benign)")
        print(f"  False Positives: {fp} (benign flagged as attack)")
        print(f"  False Negatives: {fn} (attacks missed)")
        print(f"  True Positives:  {tp} (correctly detected attacks)")
        
        if (fp + tn) > 0:
            print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
        if (fn + tp) > 0:
            print(f"  False Negative Rate: {fn/(fn+tp):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, hybrid_preds, 
                                     target_names=['BENIGN', 'ATTACK']))
        print("="*50 + "\n")

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm
        }