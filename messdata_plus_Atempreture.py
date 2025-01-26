import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

#os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("HackathonData2025+Fleet_Type_units.csv")         # load the correct data file
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

# Before scaling, let's interpolate missing values for Ambient Temperature
my_data['Ambient Temperature'] = my_data['Ambient Temperature'].interpolate(method='linear')

# Verify the interpolation worked
print("\nMissing values after interpolation:")
print(my_data['Ambient Temperature'].isnull().sum())

# Update the encoded_data with interpolated values
encoded_data = my_data.copy()

# Apply one-hot encoding using pandas get_dummies
# drop_first=True removes one column to avoid multicollinearity
categorical_columns = ['Frac Fleet', 'Fleet Type', 'Target Formation', 
                      'Field Area', 'Fuel Type', 'Sand Provider ']

encoded_data = pd.get_dummies(encoded_data, 
                             columns=categorical_columns,
                             drop_first=True,
                             prefix=categorical_columns)

# Verify the encoding
print("\nFirst few rows of encoded data:")
print(encoded_data.head())

# Check the new shape
print("\nNew shape of data:", encoded_data.shape)

# Check datatypes after encoding
print("\nDatatypes after encoding:")
print(encoded_data.dtypes)

# Drop redundant fuel-related columns since we already have individual indicators
columns_to_drop = ['Fuel Type_Diesel', 'Fuel Type_Grid', 'Fuel Type_Turbine']
encoded_data = encoded_data.drop(columns=columns_to_drop)

# We already have Grid, Diesel, CNG as individual columns (0/1 values)

# For predicting Ambient Temperature
temp_features = [col for col in encoded_data.columns if col not in [
    'Ambient Temperature',  # target variable
    'Estimated Average Stage Time',  # has missing values
    'Actual Average Stage Time',     # has missing values
]]

# For predicting Stage Times
stage_features = [col for col in encoded_data.columns if col not in [
    'Estimated Average Stage Time',  # target for first model
    'Actual Average Stage Time',     # target for second model
]]

# Create separate dataframes for each prediction task - using .copy() to avoid warnings
X_temp = encoded_data[temp_features].copy()
y_temp = encoded_data['Ambient Temperature'].copy()

X_est_time = encoded_data[stage_features].copy()
y_est_time = encoded_data['Estimated Average Stage Time'].copy()

X_act_time = encoded_data[stage_features].copy()
y_act_time = encoded_data['Actual Average Stage Time'].copy()

# Identify numerical columns (excluding target variables and boolean columns)
# Note the space after '# Clusters'
numerical_cols = ['# Clusters ', 'Grid', 'Diesel', 'CNG']

# Initialize scaler
scaler = StandardScaler()

# Scale numerical features using .loc to avoid warnings
for df in [X_temp, X_est_time, X_act_time]:
    df.loc[:, numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Print shapes to verify
print("\nShapes of prepared datasets:")
print(f"Temperature prediction: {X_temp.shape}")
print(f"Estimated Time prediction: {X_est_time.shape}")
print(f"Actual Time prediction: {X_act_time.shape}")

# Save the updated data to a new CSV file
my_data.to_csv("HackathonData2025+Fleet_Type_units_AT.csv", index=False)
