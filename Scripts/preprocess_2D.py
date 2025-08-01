import pandas as pd
import numpy as np  # ✅ Add this import
from sklearn.preprocessing import MinMaxScaler
import os

# Paths
input_path = "data\AirQualityUCI.csv"
output_npy_path = "data\preprocessed_2D.npy"
output_csv_path = "data\preprocessed_2D.csv"

os.makedirs("data", exist_ok=True)

# Load
df = pd.read_csv(input_path, sep=';', decimal=',', encoding='latin1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed cols

# Keep only sensor data (ignore Date/Time)
df_numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

# Treat -200 as NaN
df_numeric[df_numeric == -200] = pd.NA

# Drop rows with any missing values
df_clean = df_numeric.dropna().reset_index(drop=True)

# Normalize
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

# Save both .npy and .csv versions
np.save(output_npy_path, df_scaled.values)
df_scaled.to_csv(output_csv_path, index=False)

print(f"✅ Cleaned 2D data shape: {df_scaled.shape}")
print(f"✅ Saved to: {output_npy_path} and {output_csv_path}")
