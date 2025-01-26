import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("TrainData_recovered_forest_main.csv")

# Drop specified columns (safely)
my_data = my_data.drop(['Actual Average Stage Time', 'Sand Provider '], axis=1, errors='ignore')

# Print column names and their data types
print("\nColumn Names and Types:")
print(my_data.dtypes)

# Get list of categorical columns
categorical_columns = my_data.select_dtypes(include=['object']).columns

# Perform one-hot encoding
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, prefix=categorical_columns)

# Print the new shape to see how many features we have after encoding
print("\nShape after one-hot encoding:", encoded_data.shape)

# Separate target variables
target_columns = ['Grid', 'Diesel', 'CNG']
y = encoded_data[target_columns]

# Create feature matrix by dropping target columns
X = encoded_data.drop(target_columns, axis=1)

# Print shapes to verify the split
print("\nFeatures shape (X):", X.shape)
print("Target shape (y):", y.shape)

# Optional: Print feature names
print("\nFeature names:")
print(X.columns.tolist())

# Scale target variables
scaler_y = StandardScaler()
y_scaled = pd.DataFrame(scaler_y.fit_transform(y), columns=y.columns)

# Print sample of original vs scaled targets to compare
print("\nOriginal target sample:")
print(y.head())
print("\nScaled target sample:")
print(y_scaled.head())

# Scale numeric features (Ambient Temperature)
numeric_features = ['Ambient Temperature']
scaler_X = StandardScaler()
X[numeric_features] = scaler_X.fit_transform(X[numeric_features])

# Print sample to verify scaling
print("\nScaled Ambient Temperature sample:")
print(X[numeric_features].head())

# Store scaler for later use
# We'll need this to transform new data during predictions

# Set up cross-validation
n_splits = 5
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Lists to store metrics for each target
r2_scores = []
mse_scores = []

# Perform cross-validation for each target
for i, target_name in enumerate(y_scaled.columns):
    # Get scores for current target
    r2 = cross_val_score(rf_model, X, y_scaled[target_name], 
                        cv=cv, scoring='r2')
    mse = -cross_val_score(rf_model, X, y_scaled[target_name], 
                          cv=cv, scoring='neg_mean_squared_error')
    
    # Store average scores
    r2_scores.append(r2.mean())
    mse_scores.append(mse.mean())
    
    # Print results for each target
    print(f"\nResults for {target_name}:")
    print(f"Average R² score: {r2.mean():.3f} (+/- {r2.std() * 2:.3f})")
    print(f"Average MSE: {mse.mean():.3f} (+/- {mse.std() * 2:.3f})")

# Print overall performance
print("\nOverall Performance:")
print(f"Average R² across all targets: {np.mean(r2_scores):.3f}")
print(f"Average MSE across all targets: {np.mean(mse_scores):.3f}")

# Analyze Grid distribution before processing
print("\nGrid value distribution:")
print(y['Grid'].value_counts().sort_index())
print("\nGrid value statistics:")
print(y['Grid'].describe())

# Print percentage of zero values
zero_percentage = (y['Grid'] == 0).mean() * 100
print(f"\nPercentage of zero values in Grid: {zero_percentage:.2f}%")