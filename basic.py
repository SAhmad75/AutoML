import os
import joblib
import numpy as np

# Test saving a dummy model
class DummyModel:
    def predict(self, X):
        return np.ones(X.shape[0])

dummy_model = DummyModel()

# Define the model directory and filename
model_directory = "models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
model_filename = os.path.join(model_directory, "dummy_model.pkl")

try:
    joblib.dump(dummy_model, model_filename)
    print(f"Model saved as {model_filename}")
except Exception as e:
    print(f"Error saving model: {e}")
