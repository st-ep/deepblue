import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.svm import SVR

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
target_column = 'CNG'
y = encoded_data[target_column]

# Create binary labels for classification (0 vs non-zero)
y_binary = (y > 0).astype(int)

# Create feature matrix by dropping target column
X = encoded_data.drop(target_column, axis=1)

# Split data for classification
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Use ExtraTrees with best parameters found
best_reg = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_binary)

# Train regressor on non-zero data
X_train_reg = X_train[y_train_binary == 1]
y_train_reg = y.iloc[X_train.index][y_train_binary == 1]
X_test_reg = X_test[y_test_binary == 1]
y_test_reg = y.iloc[X_test.index][y_test_binary == 1]

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

# Train regressor on non-zero data
best_reg.fit(X_train_reg_scaled, y_train_reg_scaled)

# Make predictions
# First, classify zero vs non-zero
y_pred_binary = clf.predict(X_test)

# Then predict values for non-zero cases
X_test_scaled = X_test.copy()
X_test_scaled[['Ambient Temperature']] = scaler_X.transform(X_test[['Ambient Temperature']])

# Initialize predictions array with zeros
y_pred_final = np.zeros_like(y_test_binary, dtype=float)

# Get predictions only for cases where classifier predicted non-zero
non_zero_pred_mask = y_pred_binary == 1
y_pred_reg_scaled = best_reg.predict(X_test_scaled[non_zero_pred_mask])
y_pred_reg_actual = scaler_y.inverse_transform(y_pred_reg_scaled.reshape(-1, 1)).ravel()
y_pred_final[non_zero_pred_mask] = y_pred_reg_actual

# Calculate final metrics
y_test_final = y.iloc[X_test.index]
final_r2 = r2_score(y_test_final, y_pred_final)
final_mse = mean_squared_error(y_test_final, y_pred_final)
final_rmse = np.sqrt(final_mse)

print("\nFinal Model Metrics:")
print(f"R² Score: {final_r2:.3f}")
print(f"MSE: {final_mse:.3f}")
print(f"RMSE: {final_rmse:.3f}")

# Additional metrics
nmse = final_mse / np.var(y_test_final)
mape = np.mean(np.abs((y_test_final - y_pred_final) / y_test_final)) * 100
mean_value = np.mean(y_test_final)
relative_mse = final_mse / (mean_value ** 2)

print("\nAdditional Metrics:")
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

# Scale Ambient Temperature
test_encoded[['Ambient Temperature']] = scaler_X.transform(test_encoded[['Ambient Temperature']])

# Make predictions
test_binary_preds = clf.predict(test_encoded)
test_final_predictions = np.zeros(len(test_encoded))

# Get regression predictions for non-zero cases
non_zero_mask = test_binary_preds == 1
if np.any(non_zero_mask):
    y_pred_reg_scaled = best_reg.predict(test_encoded[non_zero_mask])
    y_pred_reg_actual = scaler_y.inverse_transform(y_pred_reg_scaled.reshape(-1, 1)).ravel()
    test_final_predictions[non_zero_mask] = y_pred_reg_actual

# Add predictions to test data using existing 'CNG' column
test_data['CNG'] = test_final_predictions

# Read existing predictions file and update
existing_predictions = pd.read_csv("test_main_predictions.csv")
existing_predictions['CNG'] = test_data['CNG']
existing_predictions.to_csv("test_main_predictions.csv", index=False)

print("\nCNG Predictions added to 'test_main_predictions.csv'")
print("\nSample of predictions:")
print(existing_predictions[['Grid', 'CNG']].head())