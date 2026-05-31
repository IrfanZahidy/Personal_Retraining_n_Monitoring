import os
import sys
import json
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model():
    print("=" * 50)
    print("RUNNING MULTI-CLASS PERFORMANCE EVALUATION")
    print("=" * 50)

    # 1. Load Parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    metrics_path = params["evaluate"]["metrics_path"]

    # 2. Assert paths
    for path in ["artifacts/X_test_scaled.npy", "artifacts/y_test.npy", "models/model.keras"]:
        if not os.path.exists(path):
            print(f"Error: Missing validation file {path}. Run training first.")
            sys.exit(1)

    X_test = np.load("artifacts/X_test_scaled.npy")
    y_test = np.load("artifacts/y_test.npy")

    # 3. Load Trained Classifier
    model = tf.keras.models.load_model("models/model.keras")

    # 4. Generate Predictions
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    # 5. Compute Metrics
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')

    print(f"Test Set Accuracy : {acc:.4f}")
    print(f"Test Set Macro F1 : {f1:.4f}")

    # Write evaluation output json (metrics.json is parsed automatically by DVC)
    metrics_out = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(acc),
        "macro_f1_score": float(f1)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=4)
    print(f"Metrics record updated at: {metrics_path}")

if __name__ == "__main__":
    evaluate_model()
