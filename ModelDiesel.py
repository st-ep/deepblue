import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

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

# Separate target variable - only Diesel
target_column = 'Diesel'
y = encoded_data[target_column]

# Create binary labels for classification (0 vs non-zero)
y_binary = (y > 0).astype(int)

# Create feature matrix by dropping target column
X = encoded_data.drop(target_column, axis=1)

# Split data for classification
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_binary)

# Evaluate classifier
y_pred_binary = clf.predict(X_test)
print("\nClassification Results (Zero vs Non-Zero):")
print(classification_report(y_test_binary, y_pred_binary))

# Now handle regression for non-zero values only
# Get indices of non-zero values
non_zero_mask_train = y_train_binary == 1
non_zero_mask_test = y_test_binary == 1

# Filter data for regression
X_train_reg = X_train[non_zero_mask_train]
X_test_reg = X_test[non_zero_mask_test]
y_train_reg = y.iloc[X_train.index][non_zero_mask_train]
y_test_reg = y.iloc[X_test.index][non_zero_mask_test]

# Scale features for regression
scaler_X = StandardScaler()
X_train_reg_scaled = X_train_reg.copy()
X_test_reg_scaled = X_test_reg.copy()
X_train_reg_scaled[['Ambient Temperature']] = scaler_X.fit_transform(X_train_reg[['Ambient Temperature']])
X_test_reg_scaled[['Ambient Temperature']] = scaler_X.transform(X_test_reg[['Ambient Temperature']])

# Scale target for regression
scaler_y = StandardScaler()
y_train_reg_scaled = scaler_y.fit_transform(y_train_reg.values.reshape(-1, 1)).ravel()
y_test_reg_scaled = scaler_y.transform(y_test_reg.values.reshape(-1, 1)).ravel()

# Train regression model on non-zero values
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)

# Evaluate regression model
y_pred_reg_scaled = rf_reg.predict(X_test_reg_scaled)
y_pred_reg = scaler_y.inverse_transform(y_pred_reg_scaled.reshape(-1, 1)).ravel()

# Calculate regression metrics
r2 = r2_score(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)

print("\nRegression Results (Non-Zero Values Only):")
print(f"R² score: {r2:.3f}")
print(f"MSE: {mse:.3f}")

# Combine predictions
y_pred_combined = np.zeros_like(y_test_binary, dtype=float)

# For samples predicted as non-zero by classifier
non_zero_pred_mask = y_pred_binary == 1
y_pred_combined[non_zero_pred_mask] = y_pred_reg

# Get true values
y_test_combined = y.iloc[X_test.index]

# Calculate combined metrics
combined_r2 = r2_score(y_test_combined, y_pred_combined)
combined_mse = mean_squared_error(y_test_combined, y_pred_combined)
combined_rmse = np.sqrt(combined_mse)

print("\nCombined Model Metrics (Including Zero and Non-Zero Predictions):")
print(f"R² Score: {combined_r2:.3f}")
print(f"MSE: {combined_mse:.3f}")
print(f"RMSE: {combined_rmse:.3f}")

# Additional normalized metrics
nmse = combined_mse / np.var(y_test_combined)
mape = np.mean(np.abs((y_test_combined - y_pred_combined) / y_test_combined)) * 100
mean_value = np.mean(y_test_combined)
relative_mse = combined_mse / (mean_value ** 2)

print("\nAdditional Combined Model Metrics:")
print(f"Normalized MSE: {nmse:.3f}")
print(f"MAPE: {mape:.1f}%")
print(f"MSE relative to mean²: {relative_mse:.3f}")
print(f"Mean target value: {mean_value:.1f}")

# Read and process test data
print("\nReading and Processing Test Data:")
test_data = pd.read_csv("testing_with_filled_temperatures.csv")

# Process test data the same way as training data
# One-hot encode categorical columns
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, prefix=categorical_columns)

# Ensure test data has same columns as training data
missing_cols = set(X.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0
test_encoded = test_encoded[X.columns]  # Reorder columns to match training data

# Scale Ambient Temperature only
test_encoded[['Ambient Temperature']] = scaler_X.transform(test_encoded[['Ambient Temperature']])

# Make predictions
test_binary_preds = clf.predict(test_encoded)
test_final_predictions = np.zeros(len(test_encoded))

# Get regression predictions for non-zero cases
non_zero_mask = test_binary_preds == 1
if np.any(non_zero_mask):
    y_pred_reg_scaled = rf_reg.predict(test_encoded[non_zero_mask])
    y_pred_reg_actual = scaler_y.inverse_transform(y_pred_reg_scaled.reshape(-1, 1)).ravel()
    test_final_predictions[non_zero_mask] = y_pred_reg_actual

# Add predictions to test data using existing 'Diesel' column
test_data['Diesel'] = test_final_predictions

# Read existing predictions file and update
existing_predictions = pd.read_csv("test_main_predictions.csv")
existing_predictions['Diesel'] = test_data['Diesel']
existing_predictions.to_csv("test_main_predictions.csv", index=False)

print("\nDiesel Predictions added to 'test_main_predictions.csv'")
print("\nSample of predictions:")
print(existing_predictions[['Grid', 'CNG', 'Diesel']].head())