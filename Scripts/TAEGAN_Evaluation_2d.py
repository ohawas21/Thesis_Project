import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance

# --- Load Data ---
real_data_path = "data\preprocessed_2D.csv"
synthetic_data_path = "data\TAEGAN_generated_data_2D.csv"

real = pd.read_csv(real_data_path)
synthetic = pd.read_csv(synthetic_data_path)

# --- Check same columns ---
assert list(real.columns) == list(synthetic.columns), "❌ Column mismatch!"

# --- Metrics ---
print("📊 Statistical Comparison:")
for col in real.columns:
    real_col = real[col]
    synth_col = synthetic[col]

    mae = mean_absolute_error(real_col, synth_col)
    mse = mean_squared_error(real_col, synth_col)
    emd = wasserstein_distance(real_col, synth_col)

    print(f"{col}: MAE={mae:.4f}, MSE={mse:.4f}, EMD={emd:.4f}")

# --- Visual Comparison ---
print("\n📉 Plotting feature distributions...")

num_cols = len(real.columns)
ncols = 3
nrows = (num_cols + ncols - 1) // ncols

plt.figure(figsize=(ncols * 5, nrows * 4))
for i, col in enumerate(real.columns):
    plt.subplot(nrows, ncols, i + 1)
    sns.kdeplot(real[col], label='Real', fill=True, linewidth=1)
    sns.kdeplot(synthetic[col], label='Synthetic', fill=True, linewidth=1, linestyle='--')
    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.suptitle("Real vs Synthetic Distributions (CTGAN)", fontsize=16, y=1.02)
plt.show()
