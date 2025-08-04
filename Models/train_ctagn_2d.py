# train_ctgan.py

import os
import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler

# --- Step 1: Load and preprocess your data ---
input_path = "data\preprocessed_2D.csv"
output_path = "Data\ctgan_generated.csv"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the data
df = pd.read_csv(input_path, encoding='latin1')  # or 'windows-1252'

# Drop non-numerical or timestamp columns (keep only numeric values for CTGAN)
df = df.select_dtypes(include=['number'])

# Handle missing values if present (-200 is a placeholder for missing)
df.replace(-200, pd.NA, inplace=True)
df.dropna(inplace=True)  # Alternatively: df.fillna(df.mean(), inplace=True)

# Scale the data (optional: CTGAN learns distributions, but helps with cleaner input)
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --- Step 2: Train the CTGAN model ---
print("📈 Training CTGAN model...")
model = CTGAN(epochs=1000, verbose=True)
model.fit(scaled_data)

# --- Step 3: Generate synthetic data ---
print("🔁 Generating synthetic samples...")
synthetic_data = model.sample(827)

# Inverse transform to original scale
synthetic_data = pd.DataFrame(scaler.inverse_transform(synthetic_data), columns=scaled_data.columns)

# --- Step 4: Save to file ---
synthetic_data.to_csv(output_path, index=False)
print(f"✅ Synthetic data saved to '{output_path}'")
