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

def pair_plot_generator(dataframe, field_element_string, categorical_string):
    df_subset = dataframe[dataframe["Field Area"] == field_element_string]
    df_subset.drop('Field Area', axis=1, inplace=True)
    pair_plot_fig = sns.pairplot(dataframe[['Well Name', '# Stages', '# Clusters ', 'Estimated Average Stage Time',
       'Actual Average Stage Time', 'Frac Fleet', 'Fleet Type',
       'Target Formation', 'Ambient Temperature', 'Grid',
       'Diesel', 'CNG', 'Fuel Type', 'Sand Provider ']], hue=categorical_string, corner=True);
    
    return pair_plot_fig
    
def field_area_subset(dataframe, string, formationtype):
    df_subset = dataframe[dataframe["Field Area"] == string]
    df_subset.drop('Field Area', axis=1, inplace=True)
    groupSubset = df_subset.groupby(['Target Formation']).value_counts()[formationtype]
    countsSubset = groupSubset.sum()
    return countsSubset

# Plot correlation using only numeric columns
plot_corr(numeric_data)
<<<<<<< HEAD

=======
>>>>>>> 2f2909637668c3101311edb920e7ba2257a29b25
