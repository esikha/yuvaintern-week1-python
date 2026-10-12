"""
Week 1: Python for Machine Learning & Data Preprocessing
Run: python preprocess.py
Requirements: pandas, numpy
This is a learning demo using a small illustrative student dataset.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).with_name("sample_student_data.csv")

def main():
    data = pd.read_csv(DATA_FILE)
    print("=== Original data ===")
    print(data.to_string(index=False))
    print("\nOriginal shape:", data.shape)

    # 1. Remove exact duplicate rows
    data = data.drop_duplicates().copy()
    print("\nAfter removing duplicate rows:", data.shape)

    # 2. Handle missing numeric values using each column's median
    numeric_cols = ["Age", "Study_Hours", "Attendance_Percent"]
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    # 3. Handle missing categorical values using the most frequent value
    data["Department"] = data["Department"].fillna(data["Department"].mode()[0])

    # 4. Encode categorical features (one-hot encoding)
    data = pd.get_dummies(data, columns=["Department"], dtype=int)

    # 5. Encode the target label
    data["Passed"] = data["Passed"].map({"No": 0, "Yes": 1}).astype(int)

    # 6. Normalize selected numeric features using min-max scaling.
    # Keep Student_ID as an identifier, not a model feature.
    scale_cols = ["Age", "Study_Hours", "Attendance_Percent"]
    for col in scale_cols:
        min_val, max_val = data[col].min(), data[col].max()
        if max_val != min_val:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0.0

    # 7. Simple feature selection: exclude ID and target from feature matrix
    feature_cols = [c for c in data.columns if c not in ["Student_ID", "Passed"]]
    X = data[feature_cols]
    y = data["Passed"]

    # 8. Exploratory summary
    print("\n=== Cleaned and preprocessed data ===")
    print(data.to_string(index=False))
    print("\nMissing values after preprocessing:\n", data.isna().sum())
    print("\nFeature columns:", feature_cols)
    print("\nTarget (Passed):")
    print(y.value_counts().sort_index().rename(index={0: "No", 1: "Yes"}))
    print("\nNumeric feature summary:")
    print(X.describe().round(3).to_string())

    output_file = Path(__file__).with_name("preprocessed_student_data.csv")
    data.to_csv(output_file, index=False)
    print(f"\nSaved processed dataset to: {output_file.name}")

if __name__ == "__main__":
    main()
