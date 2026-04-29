import joblib
import os
import warnings
import shap

# Suppress sklearn UserWarnings about feature names during single row prediction
warnings.filterwarnings("ignore", category=UserWarning)

class HybridIDS:
    def __init__(self, models_dir='models'):
        try:
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            self.top_features = joblib.load(os.path.join(models_dir, 'top_features.pkl'))
            self.lr_model = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
            self.ready = True
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.rf_model)
        except FileNotFoundError:
            print("Error: Model files not found. Please run train_model.py first.")
            self.ready = False

    def predict(self, data_dict):
        """
        Takes a dictionary representing a single packet/row of features.
        Returns a tuple (prediction (0 or 1), prediction_label, model_used, shap_info)
        """
        if not self.ready:
            return 0, "ERROR", "None", ""
            
        # Extract only the top features required by the models
        try:
            features = [[data_dict.get(f, 0.0) for f in self.top_features]]
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return 0, "ERROR", "None", ""

        # Scale features using the saved Scaler fit on training data
        scaled_features = self.scaler.transform(features)

        # Stage 1: Logistic Regression Fast Filter
        lr_pred = self.lr_model.predict(scaled_features)[0]

        # Stage 2: Hybrid Logic
        shap_info = ""
        if lr_pred == 0:
            # LR predicts Normal. We trust it and return fast.
            return 0, "BENIGN", "Logistic Regression", ""
        else:
            # LR predicts Attack. We use RF for final confirmation.
            rf_pred = self.rf_model.predict(scaled_features)[0]
            label = "ATTACK" if rf_pred == 1 else "BENIGN"
            
            if label == "ATTACK" and hasattr(self, 'explainer'):
                try:
                    shap_values = self.explainer.shap_values(scaled_features)
                    if isinstance(shap_values, list):
                        attack_shap = shap_values[1][0]
                    else:
                        if len(shap_values.shape) == 3:
                            attack_shap = shap_values[0, :, 1]
                        else:
                            attack_shap = shap_values[0]
                    
                    feature_impacts = list(zip(self.top_features, attack_shap))
                    # Sort by absolute magnitude of impact
                    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Top 3 features
                    top_reasons = [f"{feat} ({val:+.2f})" for feat, val in feature_impacts[:3]]
                    shap_info = "SHAP: " + ", ".join(top_reasons)
                except Exception as e:
                    print(f"SHAP Error: {e}")
                    shap_info = "SHAP Calculation Failed"

            return rf_pred, label, "Random Forest", shap_info
