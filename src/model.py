import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model():
    print("=" * 50)
    print("STARTING MULTI-CLASS CLASSIFICATION MODEL TRAINING")
    print("=" * 50)

    # 1. Load Parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    epochs = params["model"]["epochs"]
    batch_size = params["model"]["batch_size"]
    lr = params["model"]["learning_rate"]
    seed = params["data"]["random_seed"]
    test_size = params["data"]["test_size"]
    target_col = params["data"]["target_col"]

    # 2. Check and Load Dataset
    train_path = "train/train.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run split_data.py first.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    print(f"Loaded training data. Shape: {df.shape}")

    # Separating features and target
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop ID column if present
    if 'ID' in X.columns:
        X = X.drop(columns=['ID'])

    # 3. Categorical Feature Encoding
    cat_cols = X.select_dtypes(include=['O', 'object']).columns.tolist()
    print(f"Categorical features detected: {cat_cols}")
    
    for col in cat_cols:
        freq_map = X[col].value_counts().to_dict()
        X[f"{col}_freq"] = X[col].map(freq_map).fillna(0)
    X = X.drop(columns=cat_cols)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Save feature names list for alignment checks during preprocessing
    feature_columns = X.columns.tolist()
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=4)
    print(f"Saved feature column schema ({len(feature_columns)} features).")

    # 4. Target Label Encoding (7 Classes)
    y_categories = sorted(y.unique().tolist())
    label_to_index = {name: i for i, name in enumerate(y_categories)}
    y_encoded = y.map(label_to_index).values

    with open("artifacts/label_encoder_mapping.json", "w") as f:
        json.dump(label_to_index, f, indent=4)
    print(f"Mapped {len(y_categories)} classes successfully.")

    # 5. Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
    )

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler statistics for monitoring preprocessing
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open("artifacts/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=4)

    # Save split sets for independent evaluation stage
    np.save("artifacts/X_test_scaled.npy", X_test_scaled)
    np.save("artifacts/y_test.npy", y_test)

    # 7. Build Neural Network (Multi-Class Softmax Classifier)
    tf.random.set_seed(seed)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(params["model"]["dense_units_1"], activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(params["model"]["dropout_rate"]),
        tf.keras.layers.Dense(params["model"]["dense_units_2"], activation='relu'),
        tf.keras.layers.Dropout(params["model"]["dropout_rate"]),
        tf.keras.layers.Dense(len(y_categories), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 8. Model Training
    os.makedirs("models", exist_ok=True)
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save compiled network weights
    model.save("models/model.keras")
    print("Saved trained classifier model weights to models/model.keras")

if __name__ == "__main__":
    train_model()