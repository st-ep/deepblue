import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("HackathonData2025+Fleet_Type_units_AT.csv")         # load the correct data file
my_data = my_data.iloc[:,1:]  

# Define missing_columns before using it
missing_columns = [
    'Estimated Average Stage Time',
    'Actual Average Stage Time',
    'Ambient Temperature'
]

# Check for missing values
print("\nMissing values in each column:")
print(my_data.isnull().sum())

'''
# Visual representation of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(my_data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
'''

# Get more detailed information about the dataset
print("\nDetailed information about the dataset:")
print(my_data.info())

# First, let's print the actual column names
print("\nActual column names in the dataset:")
print(my_data.columns.tolist())

# Identify categorical columns (note the space after 'Sand Provider')
categorical_columns = ['Frac Fleet', 'Fleet Type', 'Target Formation', 
                      'Field Area', 'Fuel Type', 'Sand Provider ']  # Added space here

# One-hot encode categorical columns
# First, fix the 'Sand Provider ' column name by stripping whitespace
my_data.columns = my_data.columns.str.strip()
categorical_columns = [col.strip() for col in categorical_columns]

# Create EnergySum column before one-hot encoding
my_data['EnergySum'] = my_data['Grid'] + my_data['Diesel'] + my_data['CNG']

# Create dummy variables for each categorical column
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, drop_first=True)

# Check the new shape of the data
print("\nShape after one-hot encoding:", encoded_data.shape)

# Display some of the new columns to verify encoding
print("\nSample of encoded columns:")
print(encoded_data.columns[:10])

# Quick verification of the new column
print("\nEnergy Sum statistics:")
print(encoded_data['EnergySum'].describe())

# Scale EnergySum column
scaler = StandardScaler()
encoded_data['EnergySum_scaled'] = scaler.fit_transform(encoded_data[['EnergySum']])

# Verify the scaling (should have mean ≈ 0 and std ≈ 1)
print("\nScaled Energy Sum statistics:")
print(encoded_data['EnergySum_scaled'].describe())

# Drop the original unscaled column
encoded_data = encoded_data.drop('EnergySum', axis=1)

# Select only the columns we want to use for prediction
selected_categorical = ['Fleet Type', 'Target Formation', 'Fuel Type']
selected_features = selected_categorical + ['EnergySum_scaled']

# Create mask for rows where target is not missing
mask_not_missing = ~encoded_data['Estimated Average Stage Time'].isna()

# Get the column names after one-hot encoding for our selected categorical features
encoded_columns = [col for col in encoded_data.columns 
                  if any(cat in col for cat in selected_categorical) 
                  or col == 'EnergySum_scaled']

# Prepare X and y
X = encoded_data[mask_not_missing][encoded_columns]
y = encoded_data[mask_not_missing]['Estimated Average Stage Time']

# Prepare the data where we need to predict missing values
X_missing = encoded_data[~mask_not_missing][encoded_columns]

# Print shapes to verify
print("\nTraining data shape:", X.shape)
print("Training target shape:", y.shape)
print("Data with missing values shape:", X_missing.shape)

# Display sample of prepared features
print("\nFeatures used for prediction:")
print(X.columns.tolist())



