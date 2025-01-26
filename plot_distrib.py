import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better-looking plots
plt.style.use('bmh')
sns.set_palette("husl")

# Read the CSV with bootstrap predictions
df = pd.read_csv("solution_3.csv")

# Get R columns and fuel types
r_columns = [col for col in df.columns if col.startswith('R_')]
fuel_types = df['Fuel Type'].unique()

# Create separate plots for each fuel type
for idx, fuel in enumerate(fuel_types):
    # Skip the third plot (index 2)
    if idx == 2:
        continue
        
    plt.figure(figsize=(12, 7))
    
    # Get data for current fuel type
    fuel_data = df[df['Fuel Type'] == fuel][r_columns]
    flat_data = fuel_data.values.flatten()
    
    # Plot distribution with improved styling
    sns.histplot(flat_data, kde=True, color='steelblue', alpha=0.6)
    
    # Add mean and median lines
    plt.axvline(np.mean(flat_data), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(flat_data):.3f}')
    plt.axvline(np.median(flat_data), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(flat_data):.3f}')
    
    # Add statistical information
    plt.text(0.02, 0.95, 
             f'Std Dev: {np.std(flat_data):.3f}\n'
             f'95% CI: [{np.percentile(flat_data, 2.5):.3f}, {np.percentile(flat_data, 97.5):.3f}]',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Distribution of R Values for {fuel}', fontsize=14, pad=20)
    plt.xlabel('R Value', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    
    # Adjust layout and display
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot instead of displaying it
    # Create a valid filename by replacing spaces with underscores
    filename = f'distribution_{fuel.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

