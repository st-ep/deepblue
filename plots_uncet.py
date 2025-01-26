import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV with bootstrap predictions
df = pd.read_csv("test_uncert_predictions.csv")

# Identify the columns that store our 100 bootstrap realizations
g_cols = [c for c in df.columns if c.startswith('G')]
# Extract them as a NumPy array of shape (num_samples_in_test, 100)
pred_matrix = df[g_cols].values

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
plt.savefig('bootstrap_errorbar_plot_Grid.png', dpi=300, bbox_inches='tight')
plt.close()

row_index = 0  # or any other test sample index
sample_preds = pred_matrix[row_index, :]  # All 100 predictions for that row





