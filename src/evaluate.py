import os
import sys
import json
import yaml
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model():
    print("=" * 50)
    print("RUNNING MULTI-CLASS CLASSICAL MODEL EVALUATION")
    print("=" * 50)

    # 1. Assert required file paths are present
    required_paths = [
        "X_test_scaled.npy",
        "y_test.npy",
        "best_model.pkl"
    ]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Required file {path} is missing.")
            sys.exit(1)

    X_test_scaled = np.load("X_test_scaled.npy")
    y_test = np.load("y_test.npy")
    X_train = np.load("X_train.npy")

    # 2. Load and Run Model Predictions
    model = joblib.load("best_model.pkl")
    predictions = model.predict(X_test_scaled)

    # 3. Calculate Performance Metrics
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    precision = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, predictions, average='macro', zero_division=0)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")

    # 4. Save DVC Metrics Target file
    metrics_out = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(acc),
        "macro_f1_score": float(f1)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=4)
    print("Saved metrics.json tracking config.")

    # 5. Output detailed model specifications matching expectations (model_data_info.json)
    data_info = {
        "hyperparameters": {
            "model": "Decision Tree",
            "svm": None,
            "decision_tree": {
                "max_depth": 10,
                "random_state": 42
            }
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(y_test)),
        "features_count": int(X_test_scaled.shape[1]),
        "target_classes": int(len(np.unique(y_test))),
        "target_min": int(np.min(y_test)),
        "target_max": int(np.max(y_test)),
        "target_mean": float(np.mean(y_test)),
        "target_std": float(np.std(y_test)),
        "scaler_type": "StandardScaler",
        "test_performance": {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        },
        "data_info": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(y_test)),
            "features_count": int(X_test_scaled.shape[1]),
            "classes_count": int(len(np.unique(y_test))),
            "target_min": int(np.min(y_test)),
            "target_max": int(np.max(y_test)),
            "target_mean": float(np.mean(y_test)),
            "target_std": float(np.std(y_test))
        },
        "timestamp": datetime.now().isoformat()
    }

    with open("model_data_info.json", "w") as f_info:
        json.dump(data_info, f_info, indent=4)
    print("Saved model_data_info.json specifications file.")

if __name__ == "__main__":
    evaluate_model()