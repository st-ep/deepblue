import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import os
os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("HackathonData2025.csv")         # load the correct data file
my_data = my_data.iloc[:,1:]  

# Select only numeric columns for correlation
numeric_data = my_data.select_dtypes(include=[np.number])
def plot_corr(dataframe,size=10):                       # plots a correlation matrix as a heat map 
    corr = dataframe.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(corr,vmin = -1.0, vmax = 1.0)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='left')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(im, orientation = 'vertical')
    plt.title('Correlation Matrix')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

# Plot correlation using only numeric columns
plot_corr(numeric_data)

