import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("HackathonData2025.csv")         # load the correct data file
my_data = my_data.iloc[:,1:]  

# Basic data exploration
print("Dataset Shape:", my_data.shape)
print("\nColumns:", my_data.columns.tolist())

# Select only numeric columns for correlation and statistics
numeric_columns = my_data.select_dtypes(include=[np.number]).columns
print("\nNumeric Columns:", numeric_columns.tolist())

print("\nBasic Statistics:")
print(my_data[numeric_columns].describe())

# Check for missing values
print("\nMissing Values:")
print(my_data.isnull().sum())

# Create correlation matrix heatmap with better formatting
plt.figure(figsize=(12, 8))
correlation_matrix = my_data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Distribution plots for key energy metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribution of Energy Usage by Type')

sns.histplot(data=my_data['Grid'].dropna(), ax=axes[0], kde=True)
axes[0].set_title('Grid Energy Usage')
axes[0].set_xlabel('kWh')

sns.histplot(data=my_data['Diesel'].dropna(), ax=axes[1], kde=True)
axes[1].set_title('Diesel Usage')
axes[1].set_xlabel('gallons')

sns.histplot(data=my_data['CNG'].dropna(), ax=axes[2], kde=True)
axes[2].set_title('CNG Usage')
axes[2].set_xlabel('MMBTU')

plt.tight_layout()
plt.show()

# Box plots for energy usage by Target Formation
plt.figure(figsize=(12, 6))
sns.boxplot(x='Target Formation', y='Grid', data=my_data)
plt.xticks(rotation=45)
plt.title('Grid Energy Usage by Target Formation')
plt.tight_layout()
plt.show()

# Analysis by Fuel Type
print("\nAverage Energy Usage by Fuel Type:")
print(my_data.groupby('Fuel Type')[['Grid', 'Diesel', 'CNG']].mean())

# Distribution plot for Field Area
plt.figure(figsize=(10, 6))
sns.histplot(data=my_data['Field Area'].dropna(), kde=True)
plt.title('Distribution of Field Area')
plt.xlabel('Field Area')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print selected columns for rows with missing Fleet Type
print("\nRows with missing Fleet Type (selected columns):")
selected_columns = ['Fleet Type', 'Field Area', 'Target Formation', 'Grid']  # add any other columns you want to see
print(my_data[my_data['Fleet Type'].isna()][selected_columns])


