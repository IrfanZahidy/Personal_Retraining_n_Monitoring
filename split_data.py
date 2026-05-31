import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create the folder structure required by the pipeline
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load your custom obesity dataset
print("Loading raw dataset...")
df = pd.read_csv("ObesityDataSet.csv")

# We use stratified splitting since 'NObeyesdad' is a multi-class target
target_col = "NObeyesdad"

# Split: 85% baseline training data, 15% static test evaluation data
train_df, test_df = train_test_split(
    df, 
    test_size=0.15, 
    random_state=42, 
    stratify=df[target_col]
)

# Save baseline splits (These will be managed by DVC)
train_df.to_csv("train/train.csv", index=False)
test_df.to_csv("test/test.csv", index=False)

# Sample 100 random rows from your test set to simulate "new incoming production data"
# We save this in data/new_data.csv to trigger our automated monitoring pipeline later
new_data = test_df.sample(n=100, random_state=101)
new_data.to_csv("data/new_data.csv", index=False)

print("\n--- Split Completed Successfully ---")
print(f"1. Baseline Train Data shape : {train_df.shape} (Saved to train/train.csv)")
print(f"2. Baseline Test Data shape  : {test_df.shape}  (Saved to test/test.csv)")
print(f"3. Simulated New Batch shape : {new_data.shape}   (Saved to data/new_data.csv)")