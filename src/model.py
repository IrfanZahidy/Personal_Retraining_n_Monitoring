import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_model():
    print("=" * 50)
    print("STARTING CLASSICAL MACHINE LEARNING PIPELINE")
    print("=" * 50)

    # 1. Load Parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    seed = params["data"].get("random_seed", 42)
    test_size = params["data"].get("test_size", 0.15)
    target_col = params["data"].get("target_col", "NObeyesdad")

    # 2. Check and Load Dataset
    train_path = "train/train.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run split_data.py first.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    print(f"Loaded training dataset. Shape: {df.shape}")

    # Separate features and target
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop ID column if present
    if 'ID' in X.columns:
        X = X.drop(columns=['ID'])

    # 3. Handle Categorical Feature Encodings (Dummy One-Hot Encoding)
    cat_cols = X.select_dtypes(include=['O', 'object']).columns.tolist()
    print(f"Categorical features detected: {cat_cols}")
    
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # CRITICAL FIX: Convert entire DataFrame to float64 to purge Object/Boolean types
    X_encoded = X_encoded.astype(np.float64)
    feature_columns = X_encoded.columns.tolist()

    # Save feature names signature
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=4)
    print(f"Saved feature_columns.json with {len(feature_columns)} features.")

    # 4. Target Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    target_info = {
        "target_column": "obesity_class",
        "num_classes": int(num_classes),
        "unique_classes": [int(i) for i in range(num_classes)]
    }
    with open("target_column.json", "w") as f:
        json.dump(target_info, f, indent=4)
    print("Saved target_column.json mapping configuration.")

    # 5. Train-Test Split (with stratification for balanced representation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded.values, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
    )

    # 6. Apply Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save StandardScaler
    joblib.dump(scaler, "scaler.pkl")
    print("Saved scaler.pkl preprocessing object.")

    # 7. Save NumPy Array Splits casting to match other group's exact descriptors (<i8 and <f8)
    np.save("X_train.npy", X_train.astype(np.int64))
    np.save("X_train_scaled.npy", X_train_scaled.astype(np.float64))
    np.save("X_test.npy", X_test.astype(np.int64))
    np.save("X_test_scaled.npy", X_test_scaled.astype(np.float64))
    np.save("y_train.npy", y_train.astype(np.int64))
    np.save("y_test.npy", y_test.astype(np.int64))
    print("Exported all array split partitions (.npy) with strict numeric types.")

    # 8. Train and Compare Support Vector Machine vs Decision Tree Classifiers
    # A. SVM
    svm_model = SVC(kernel='rbf', probability=True, random_state=seed)
    svm_model.fit(X_train_scaled, y_train)
    svm_preds = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM Test Accuracy: {svm_acc:.4f}")

    # B. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=seed)
    dt_model.fit(X_train_scaled, y_train)
    dt_preds = dt_model.predict(X_test_scaled)
    dt_acc = accuracy_score(y_test, dt_preds)
    print(f"Decision Tree Test Accuracy: {dt_acc:.4f}")

    # 9. Save Best Model and Comparison Visualizations
    best_model = dt_model if dt_acc >= svm_acc else svm_model
    joblib.dump(best_model, "best_model.pkl")
    print(f"Saved best performing model as best_model.pkl")

    # Generate the Comparison Plot (model_comparison.png)
    plt.figure(figsize=(10, 6))
    models = ['SVM', 'Decision Tree']
    accuracies = [svm_acc, dt_acc]
    bars = plt.bar(models, accuracies, color=['#e5e7eb', '#f59e0b'], width=0.6)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    
    # Label the exact values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()
    print("Saved model_comparison.png visualization metrics plot.")

    # 10. Write Model Summary to model_summary.txt
    summary_text = f"""Model: Decision Tree Classifier
=========================================
Hyperparameters:
- Max Depth: 10
- Random State: {seed}
=========================================
Architecture Configuration Info:
- Input Features count: {X_train.shape[1]}
- Output target classes: {num_classes}
- Total Samples processed: {len(X_train_scaled)}
"""
    with open("model_summary.txt", "w") as f_sum:
        f_sum.write(summary_text.strip())
    print("Saved model_summary.txt summary file.")

if __name__ == "__main__":
    train_model()