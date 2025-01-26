import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your test set predictions (with D1..D100)
df = pd.read_csv("test_uncert_predictions.csv")

# 2. Identify columns with bootstrap predictions
g_cols = [col for col in df.columns if col.startswith('G')]
pred_matrix = df[g_cols].values  # shape: (n_test_samples, 100)

# 3. Compute mean and std across columns (per test sample)
pred_mean = np.mean(pred_matrix, axis=1)
pred_std  = np.std(pred_matrix, axis=1)

# 4. Create a mask for "non-zero" samples
#    Method A: Use 'mean > 0' (only include rows with a strictly positive mean)
# nonzero_mask = (pred_mean > 0)

#    Method B: Include any row where *at least one* of the 100 predictions is non-zero
nonzero_mask = np.any(pred_matrix != 0, axis=1)

# Filter down to non-zero samples
nonzero_std = pred_std[nonzero_mask]

# 5. Plot the distribution with improved styling
plt.style.use('seaborn-v0_8')  # Updated style name
# OR alternatively use:
# plt.style.use('seaborn-darkgrid')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Histogram with KDE
sns.histplot(data=nonzero_std, bins=30, color='#FF6B6B', alpha=0.7, ax=ax1)
sns.kdeplot(data=nonzero_std, color='#4A90E2', linewidth=2, ax=ax1)

ax1.set_title("Distribution of Prediction Uncertainty\nfor Non-Zero Samples", 
              fontsize=12, pad=15)
ax1.set_xlabel("Standard Deviation of Bootstrap Predictions", fontsize=10)
ax1.set_ylabel("Count", fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add summary statistics annotation
stats_text = f'Mean: {nonzero_std.mean():.3f}\nStd: {nonzero_std.std():.3f}\nMedian: {np.median(nonzero_std):.3f}'
ax1.text(0.95, 0.95, stats_text,
         transform=ax1.transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Enhanced boxplot with violin plot
sns.violinplot(y=nonzero_std, color='#FF6B6B', alpha=0.7, ax=ax2)
sns.boxplot(y=nonzero_std, color='white', width=0.2, 
           showfliers=True, fliersize=5, ax=ax2)

ax2.set_title("Distribution of Prediction Uncertainty\nwith Violin and Box Plot", 
              fontsize=12, pad=15)
ax2.set_ylabel("Standard Deviation of Predictions", fontsize=10)

plt.tight_layout()
plt.savefig('uncertainty_distribution_Grid.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
