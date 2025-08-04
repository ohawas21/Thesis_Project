import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance

# --- File Paths ---
real_data_path = "data\\preprocessed_3D.csv"
synthetic_data_path = "data\\TAEGAN_generated_data.csv"

# --- Load Real Data ---
real = pd.read_csv(real_data_path)
expected_columns = real.columns

# --- Load Synthetic Data (no header) and assign real columns ---
synthetic = pd.read_csv(synthetic_data_path, header=None)
synthetic.columns = expected_columns

# --- Truncate to same length ---
min_len = min(len(real), len(synthetic))
real = real.iloc[:min_len].reset_index(drop=True)
synthetic = synthetic.iloc[:min_len].reset_index(drop=True)

# --- Clean invalid string values and ensure all columns are numeric ---
real = real.apply(pd.to_numeric, errors='coerce')
synthetic = synthetic.apply(pd.to_numeric, errors='coerce')

# --- Drop any columns that couldn't be converted in either ---
real = real.dropna(axis=1)
synthetic = synthetic.dropna(axis=1)

# --- Align columns again after dropping invalid ones ---
common_cols = list(set(real.columns) & set(synthetic.columns))
real = real[common_cols]
synthetic = synthetic[common_cols]

# --- Final check ---
assert len(real.columns) == len(synthetic.columns), "❌ Column mismatch after cleaning!"

# --- Statistical Evaluation ---
print("📊 Statistical Comparison:")
for col in sorted(common_cols):
    real_col = real[col]
    synth_col = synthetic[col]

    mae = mean_absolute_error(real_col, synth_col)
    mse = mean_squared_error(real_col, synth_col)
    emd = wasserstein_distance(real_col, synth_col)

    print(f"{col}: MAE={mae:.4f}, MSE={mse:.4f}, EMD={emd:.4f}")

# --- KDE Plot ---
print("\n📉 Plotting feature distributions...")

num_cols = len(common_cols)
ncols = 3
nrows = (num_cols + ncols - 1) // ncols

plt.figure(figsize=(ncols * 5, nrows * 4))
for i, col in enumerate(sorted(common_cols)):
    plt.subplot(nrows, ncols, i + 1)
    sns.kdeplot(real[col], label='Real', fill=True, linewidth=1)
    sns.kdeplot(synthetic[col], label='Synthetic', fill=True, linewidth=1, linestyle='--')
    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.suptitle("Real vs Synthetic Distributions (TAEGAN)", fontsize=16, y=1.02)
plt.show()
