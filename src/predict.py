import joblib
import pandas as pd

def load_model(model_path: str):
    """Load the model from a given file path."""
    return joblib.load(model_path)

def predict(model, data: pd.DataFrame):
    """Make predictions using the loaded model."""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    return predictions, probabilities
