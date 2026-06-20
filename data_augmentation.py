# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:34:28 2025

@author: Lohith
"""

# Import Required Libraries
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("cleaned_animal_disease_prediction.csv")

# Identify target column (Disease Prediction column)
target_col = [col for col in df.columns if 'Disease_Prediction' in col][0]

# Analyze current class distribution
class_counts = df[target_col].value_counts()

# Set desired minimum samples per class
TARGET_SAMPLES = 50

# Data Augmentation Function
def augment_data(class_df, n_needed):
    augmented = []
    for _ in range(n_needed):
        row = class_df.sample(n=1, replace=True).copy()

        # Apply slight random noise to numeric columns only
        for col in class_df.select_dtypes(include=np.number).columns:
            noise = np.random.normal(0, 0.02)  # Very small noise
            row[col] += noise

        augmented.append(row)

    return pd.concat(augmented)

# Augment dataset
augmented_frames = []
for disease, count in class_counts.items():
    class_df = df[df[target_col] == disease]
    if count < TARGET_SAMPLES:
        n_needed = TARGET_SAMPLES - count
        augmented = augment_data(class_df, n_needed)
        augmented_frames.append(augmented)

# Combine original + augmented data
if augmented_frames:
    df_augmented = pd.concat([df] + augmented_frames, ignore_index=True)
else:
    df_augmented = df.copy()

# Shuffle dataset after augmentation
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the new dataset
df_augmented.to_csv("augmented_animal_disease_prediction.csv", index=False)

# Display New Class Distribution
print("\n=== New Class Distribution ===")
print(df_augmented[target_col].value_counts())
