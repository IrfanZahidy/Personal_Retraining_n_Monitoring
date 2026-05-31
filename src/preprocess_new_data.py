import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

def preprocess_new_data():
    print("=" * 50)
    print("PREPROCESSING INCOMING DATASTREAM BATCH")
    print("=" * 50)

    new_data_path = "data/new_data.csv"
    scaler_path = "scaler.pkl"
    feature_cols_path = "feature_columns.json"

    # Assert processing elements are ready
    if not os.path.exists(scaler_path) or not os.path.exists(feature_cols_path):
        print("Error: Scaler or Feature column schemas not found. Run model training first.")
        sys.exit(1)

    if not os.path.exists(new_data_path):
        print("Notice: No incoming dataset batch located at data/new_data.csv.")
        sys.exit(0)

    # 1. Load incoming new data
    new_df = pd.read_csv(new_data_path)
    print(f"Loaded new incoming data batch. Shape: {new_df.shape}")

    # Remove target columns if present
    target_cols = ["NObeyesdad", "obesity_class"]
    for col in target_cols:
        if col in new_df.columns:
            new_df = new_df.drop(columns=[col])

    if 'ID' in new_df.columns:
        new_df = new_df.drop(columns=['ID'])

    # 2. Convert categorical features with dummy columns
    cat_cols = new_df.select_dtypes(include=['O', 'object']).columns.tolist()
    new_df_encoded = pd.get_dummies(new_df, columns=cat_cols, drop_first=True)

    # 3. Align Columns to Match Training Schema Signature
    with open(feature_cols_path, "r") as f:
        feature_columns = json.load(f)

    # Ensure all original dummy columns exist (fill missing ones with 0)
    for col in feature_columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0.0

    # Ensure identical column order and drop extraneous features
    aligned_df = new_df_encoded[feature_columns].copy()
    aligned_df = aligned_df.fillna(aligned_df.mean()).fillna(0.0)

    # 4. Apply Scaler and Save output
    scaler = joblib.load(scaler_path)
    X_new_scaled = scaler.transform(aligned_df.values)

    np.save("X_test_scaled.npy", X_new_scaled)
    print(f"Saved aligned, scaled batch array to X_test_scaled.npy. Shape: {X_new_scaled.shape}")

if __name__ == "__main__":
    preprocess_new_data()