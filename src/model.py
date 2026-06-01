import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_model():
    print("=" * 50)
    print("STARTING OBLIGATORY CLASSIFICATION TRAINING PIPELINE")
    print("=" * 50)

    # 1. Load Parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    epochs = params["model"].get("epochs", 50)
    batch_size = params["model"].get("batch_size", 32)
    lr = params["model"].get("learning_rate", 0.001)
    seed = params["data"].get("random_seed", 42)
    test_size = params["data"].get("test_size", 0.15)
    target_col = "NObeyesdad"

    # 2. Check and Load Dataset
    train_path = "train/train.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run split_data.py first.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    print(f"Loaded training data. Shape: {df.shape}")

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if 'ID' in X.columns:
        X = X.drop(columns=['ID'])

    # 3. Categorical Feature Encoding
    cat_cols = X.select_dtypes(include=['O', 'object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    # Save feature columns signature
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=4)

    # Encode Target Labels (7 Classes)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    target_info = {
        "target_column": target_col,
        "num_classes": int(num_classes),
        "unique_classes": [int(i) for i in range(num_classes)],
        "classes_mapping": list(label_encoder.classes_)
    }
    with open("target_column.json", "w") as f:
        json.dump(target_info, f, indent=4)

    # Save encoders
    joblib.dump(label_encoder, "label_encoder.pkl")

    # 4. Train-Test Split (with stratification for balanced representation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded.values, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
    )

    # 5. Apply Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # Save split arrays for evaluator
    np.save("X_train_scaled.npy", X_train_scaled)
    np.save("X_test_scaled.npy", X_test_scaled)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    # ==========================================
    # PART A: TRADITIONAL MACHINE LEARNING MODELS
    # ==========================================
    print("\n--- Training Traditional ML Models ---")
    
    # Model 1: Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_model.fit(X_train_scaled, y_train)
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"Random Forest 5-Fold CV Mean Score: {rf_cv_scores.mean():.4f}")

    # Model 2: MLP Classifier (Neural Network)
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    mlp_model.fit(X_train_scaled, y_train)
    mlp_cv_scores = cross_val_score(mlp_model, X_train_scaled, y_train, cv=5)
    print(f"MLP Classifier 5-Fold CV Mean Score: {mlp_cv_scores.mean():.4f}")

    # Identify and save the best traditional model
    best_traditional = rf_model if rf_cv_scores.mean() >= mlp_cv_scores.mean() else mlp_model
    joblib.dump(best_traditional, "best_traditional_model.pkl")
    print(f"Best Traditional Model Saved: {'Random Forest' if best_traditional == rf_model else 'MLP Classifier'}")

    # ==========================================
    # PART B: DEEP LEARNING - ANN CLASSIFIER
    # ==========================================
    print("\n--- Training Deep Learning ANN Model ---")
    tf.random.set_seed(seed)

    # Build ANN Architecture exactly matching PDF Part B instructions
    ann_model = tf.keras.models.Sequential([
        # Input layer automatically determined, First Dense layer: 64 units, ReLU
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2), # Dropout layer: 0.2
        
        # Second Dense layer: 32 units, ReLU
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2), # Dropout layer: 0.2
        
        # Third Dense layer: 16 units, ReLU
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Output Dense layer: number of classes with Softmax activation
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    ann_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks as requested in Step B.3 of instructions
    callbacks = [
        # Early stopping if validation loss does not improve for 10 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate by half if validation loss plateaus for 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            verbose=1
        )
    ]

    # Train Deep Learning Model
    history = ann_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2, # 20% validation split
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save trained ANN Model as .h5 file and history as pickle
    ann_model.save("ann_classifier_model.h5")
    joblib.dump(history.history, "training_history.pkl")
    print("Saved Deep Learning ANN model and history configurations successfully.")

if __name__ == "__main__":
    train_model()