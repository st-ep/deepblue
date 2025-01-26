import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV with bootstrap predictions
df = pd.read_csv("test_uncert_predictions.csv")

# Identify the columns that store our 100 bootstrap realizations
c_cols = [c for c in df.columns if c.startswith('C')]
# Extract them as a NumPy array of shape (num_samples_in_test, 100)
pred_matrix = df[c_cols].values

# Compute summary statistics across the 100 realizations
pred_mean = np.mean(pred_matrix, axis=1)  # shape = (num_samples_in_test,)
pred_std  = np.std(pred_matrix, axis=1)   # shape = (num_samples_in_test,)

# Plot: mean predictions vs. sample index, with ±2 std as error bars
plt.figure(figsize=(10, 5))
plt.errorbar(
    x=range(len(pred_mean)),
    y=pred_mean,
    yerr=2 * pred_std,
    fmt='o',  # 'o' for circle markers
    ecolor='red',
    capsize=3
)
plt.title("Mean ± 2*STD for Grid Predictions (Bootstrap Realizations)")
plt.xlabel("Test Sample Index")
plt.ylabel("Predicted Grid")
plt.tight_layout()
plt.show()

row_index = 0  # or any other test sample index
sample_preds = pred_matrix[row_index, :]  # All 100 predictions for that row

plt.figure()
plt.hist(sample_preds, bins=15, alpha=0.7, color='blue')
plt.title(f"Distribution of 100 Bootstrap Predictions for Row {row_index}")
plt.xlabel("Predicted Grid")
plt.ylabel("Frequency")
plt.show()

# Choose how many samples you want to visualize
num_samples_to_plot = 20
sample_indices = range(num_samples_to_plot)

plt.figure(figsize=(12, 6))
# Note: seaborn boxplot requires columns as variables,
# so transpose pred_matrix[sample_indices, :] to shape (100, num_samples_to_plot)
sns.boxplot(data=pred_matrix[sample_indices, :].T, orient='v')
plt.xlabel("Test Sample Index")
plt.ylabel("Predicted Grid (Bootstrap Distribution)")
plt.title("Boxplots of Grid Predictions for the First 20 Rows")
plt.tight_layout()
plt.show()

if "CNG" in df.columns:
    grid_baseline = df["CNG"].values
    plt.figure(figsize=(10, 5))
    plt.errorbar(range(len(pred_mean)), pred_mean, yerr=2*pred_std, fmt='o', label='Bootstrap Mean ± 2 STD')
    plt.plot(range(len(grid_baseline)), grid_baseline, 'r-', label='Single Model Prediction')
    plt.title("Comparing Bootstrap Mean vs. Baseline Prediction")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Predicted Grid")
    plt.legend()
    plt.tight_layout()
    plt.show()


