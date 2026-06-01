import os
import sys
import json
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Register custom r2 metric for loading model safely
def r2_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))

def evaluate_model():
    print("=" * 50)
    print("RUNNING 1D CNN REGRESSION PERFORMANCE EVALUATION")
    print("=" * 50)

    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Assert required files
    required_paths = ["X_test_scaled.npy", "y_test.npy", "models/model.keras", "training_history.json"]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Missing required validation file {path}.")
            sys.exit(1)

    # Define AEST Timezone
    aest_tz = timezone(timedelta(hours=10))
    aest_now = datetime.now(aest_tz)

    # Load testing variables safely allowing pickle parsing
    X_test_scaled = np.load("X_test_scaled.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    X_train = np.load("X_train.npy", allow_pickle=True)

    # Load Model weights
    model = tf.keras.models.load_model("models/model.keras", custom_objects={'r2_metric': r2_metric})
    
    # Reshape and predict
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
    predictions = model.predict(X_test_cnn, verbose=0).flatten()

    # Calculate scores
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MSE : {mse:.4f}")
    print(f"Test MAE : {mae:.4f}")
    print(f"Test R2  : {r2:.4f}")

    # Export metric JSON configs with AEST timestamp
    test_metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "timestamp": aest_now.isoformat()
    }
    with open("test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    with open("metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Export overall model configurations profile info (model_info.json)
    model_info = {
        "model_type": "CNN_Regression",
        "input_shape": [int(X_test_scaled.shape[1]), 1],
        "num_features": int(X_test_scaled.shape[1]),
        "num_training_samples": int(len(X_train)),
        "num_test_samples": int(len(y_test)),
        "feature_columns_preview": [f"X{i}" for i in range(min(10, X_test_scaled.shape[1]))],
        "target_column": params["data"].get("target_col", "y"),
        "training_completed": aest_now.isoformat(),
        "hyperparameters": {
            "epochs": params["model"].get("epochs", 10),
            "batch_size": params["model"].get("batch_size", 32),
            "learning_rate": params["model"].get("learning_rate", 0.0001),
            "cnn_filters": params["model"].get("cnn_filters", [64, 128, 64]),
            "kernel_size": params["model"].get("kernel_size", 3),
            "pool_size": params["model"].get("pool_size", 2),
            "dense_units": params["model"].get("dense_units", [256, 128, 64]),
            "dropout_rates": params["model"].get("dropout_rates", [0.3, 0.2])
        },
        "test_performance": {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "timestamp": aest_now.isoformat()
        }
    }
    with open("model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)

    # Generate side-by-side Loss vs R2 Score metrics plot
    with open("training_history.json", "r") as f:
        history = json.load(f)

    epochs_range = range(len(history["loss"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Chart 1: Loss curves
    ax1.plot(epochs_range, history["loss"], label="Train Loss", color="tab:blue", linewidth=1.8)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="tab:orange", linewidth=1.8)
    ax1.set_title("Model Loss", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss (MSE)", fontsize=10)
    ax1.grid(True)
    ax1.legend()

    # Chart 2: R2 Metric curves
    ax2.plot(epochs_range, history["r2_metric"], label="Train R2", color="tab:blue", linewidth=1.8)
    ax2.plot(epochs_range, history["val_r2_metric"], label="Val R2", color="tab:orange", linewidth=1.8)
    ax2.set_title("Model R2 Score", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("R2", fontsize=10)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("model_results.png", dpi=150)
    plt.close()
    print("Successfully generated model_results.png curves.")

if __name__ == "__main__":
    evaluate_model()