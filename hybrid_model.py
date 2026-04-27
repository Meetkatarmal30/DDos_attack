import joblib
import os
import warnings

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
        except FileNotFoundError:
            print("Error: Model files not found. Please run train_model.py first.")
            self.ready = False

    def predict(self, data_dict):
        """
        Takes a dictionary representing a single packet/row of features.
        Returns a tuple (prediction (0 or 1), prediction_label, model_used)
        """
        if not self.ready:
            return 0, "ERROR", "None"
            
        # Extract only the top features required by the models
        try:
            features = [[data_dict.get(f, 0.0) for f in self.top_features]]
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return 0, "ERROR", "None"

        # Scale features using the saved Scaler fit on training data
        scaled_features = self.scaler.transform(features)

        # Stage 1: Logistic Regression Fast Filter
        lr_pred = self.lr_model.predict(scaled_features)[0]

        # Stage 2: Hybrid Logic
        if lr_pred == 0:
            # LR predicts Normal. We trust it and return fast.
            return 0, "BENIGN", "Logistic Regression"
        else:
            # LR predicts Attack. We use RF for final confirmation.
            rf_pred = self.rf_model.predict(scaled_features)[0]
            label = "ATTACK" if rf_pred == 1 else "BENIGN"
            return rf_pred, label, "Random Forest"
