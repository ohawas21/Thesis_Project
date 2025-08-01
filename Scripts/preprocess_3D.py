import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Parameters
WINDOW_SIZE = 24
STRIDE = 1

# Paths
input_path = "data\AirQualityUCI.csv"
output_npy_path = "data\preprocessed_3D.npy"
output_csv_path = "data\preprocessed_3D_flattened.csv"

os.makedirs("data", exist_ok=True)

# Load
df = pd.read_csv(input_path, sep=';', decimal=',', encoding='latin1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df_numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
df_numeric[df_numeric == -200] = pd.NA
df_clean = df_numeric.dropna().reset_index(drop=True)

# Normalize
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

# Create sliding windows (3D data)
samples = []
for i in range(0, len(df_scaled) - WINDOW_SIZE + 1, STRIDE):
    window = df_scaled.iloc[i:i+WINDOW_SIZE].values
    samples.append(window)

X = np.stack(samples)
np.save(output_npy_path, X)

# Flatten each 3D sample (24, 13) → (1, 312)
flattened = X.reshape((X.shape[0], -1))
np.savetxt(output_csv_path, flattened, delimiter=",")
print(f"✅ Cleaned 3D data shape: {X.shape}")
print(f"✅ Flattened 3D data shape: {flattened.shape} saved to {output_csv_path}")
