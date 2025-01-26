import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report

# ----------------------------------------------------------------------
# 0. Setup: read training data, drop columns, etc. (Your existing code)
# ----------------------------------------------------------------------
os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  
my_data = pd.read_csv("TrainData_recovered_forest_main.csv")

my_data = my_data.drop(['Actual Average Stage Time','Sand Provider '],
                       axis=1, errors='ignore')

# Print column names & dtypes
print("\nColumn Names and Types:")
print(my_data.dtypes)

# One-hot encode categorical columns
categorical_columns = my_data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, prefix=categorical_columns)
print("\nShape after one-hot encoding:", encoded_data.shape)

# Separate target variable (Diesel)
target_column = 'Diesel'
y_full = encoded_data[target_column].copy()
X_full = encoded_data.drop(target_column, axis=1)

# Create binary labels (0 vs. non-zero)
y_binary_full = (y_full > 0).astype(int)

# Split data for classification/regression
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X_full, y_binary_full, test_size=0.2, random_state=42
)

# For the non-zero regression portion
# Indices where training Diesel is non-zero
non_zero_mask_train = (y_train_binary == 1)
# Actual Diesel values from the original y_full:
y_train_reg_full = y_full.loc[X_train.index]  # align by index
y_train_reg = y_train_reg_full[non_zero_mask_train]

# Same idea for test
non_zero_mask_test = (y_test_binary == 1)
y_test_reg_full = y_full.loc[X_test.index]
y_test_reg = y_test_reg_full[non_zero_mask_test]

# ----------------------------------------------------------------------
# 1. (Optional) Train a "single" classifier/regressor for reference
#    -- This block is your original code, kept for baseline.
# ----------------------------------------------------------------------
print("\n[Baseline Single Model Training]")

clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
clf_baseline.fit(X_train, y_train_binary)
y_pred_binary_baseline = clf_baseline.predict(X_test)
print("\nClassification Results (Zero vs Non-Zero) - Baseline:")
print(classification_report(y_test_binary, y_pred_binary_baseline))

# Scale features (only 'Ambient Temperature' in your example)
scaler_X = StandardScaler()
X_train_reg_scaled = X_train[non_zero_mask_train].copy()
X_test_reg_scaled = X_test[non_zero_mask_test].copy()
X_train_reg_scaled[['Ambient Temperature']] = scaler_X.fit_transform(
    X_train_reg_scaled[['Ambient Temperature']])
X_test_reg_scaled[['Ambient Temperature']] = scaler_X.transform(
    X_test_reg_scaled[['Ambient Temperature']])

# Scale target
scaler_y = StandardScaler()
y_train_reg_scaled = scaler_y.fit_transform(y_train_reg.values.reshape(-1, 1)).ravel()

rf_reg_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg_baseline.fit(X_train_reg_scaled, y_train_reg_scaled)

# Evaluate baseline regression
y_pred_reg_scaled_baseline = rf_reg_baseline.predict(X_test_reg_scaled)
y_pred_reg_baseline = scaler_y.inverse_transform(
    y_pred_reg_scaled_baseline.reshape(-1,1)
).ravel()

r2_baseline = r2_score(y_test_reg, y_pred_reg_baseline)
mse_baseline = mean_squared_error(y_test_reg, y_pred_reg_baseline)
print("\nRegression Results (Non-Zero Values Only) - Baseline:")
print(f"R² score: {r2_baseline:.3f}")
print(f"MSE: {mse_baseline:.3f}")

# Combine for final baseline predictions
y_pred_binary_baseline = clf_baseline.predict(X_test)
y_pred_combined_baseline = np.zeros_like(y_test_binary, dtype=float)
non_zero_pred_mask_baseline = (y_pred_binary_baseline == 1)
y_pred_combined_baseline[non_zero_pred_mask_baseline] = y_pred_reg_baseline

combined_r2_baseline = r2_score(y_test_reg_full, y_pred_combined_baseline)
combined_mse_baseline = mean_squared_error(y_test_reg_full, y_pred_combined_baseline)
combined_rmse_baseline = np.sqrt(combined_mse_baseline)

print("\nCombined Baseline Model Metrics:")
print(f"R² Score: {combined_r2_baseline:.3f}")
print(f"RMSE: {combined_rmse_baseline:.3f}")

# ----------------------------------------------------------------------
# 2. Read the external test data (the new unseen data)
# ----------------------------------------------------------------------
print("\nReading and Processing External Test Data:")
test_data = pd.read_csv("testing_with_filled_temperatures.csv")

# One-hot encode, matching training columns
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, prefix=categorical_columns)

# Ensure same columns as X_full
missing_cols = set(X_full.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0
test_encoded = test_encoded[X_full.columns]

# ----------------------------------------------------------------------
# 3. We now create 100 bootstrap realizations (D1...D100).
#    Each iteration:
#        - sample training data w/ replacement
#        - train classifier
#        - train regressor on non-zero subset
#        - predict on external test_encoded
#        - add noise to non-zero predictions
#        - store in existing_predictions['D_i']
# ----------------------------------------------------------------------
n_bootstrap = 100

# Read your existing predictions file
existing_predictions = pd.read_csv("test_main_predictions.csv")

# We’ll store the 100 new columns here
bootstrap_preds = np.zeros((len(test_encoded), n_bootstrap))

# For reproducibility (optional)
rng = np.random.default_rng(seed=1234)

for i in range(n_bootstrap):
    # -----------------------------
    # (a) Create a bootstrap sample
    # -----------------------------
    # We sample from X_train *and* y_train_binary (and original Diesel y)
    n_train = len(X_train)
    bootstrap_indices = rng.integers(low=0, high=n_train, size=n_train)
    
    X_train_boot = X_train.iloc[bootstrap_indices].copy()
    y_train_binary_boot = y_train_binary.iloc[bootstrap_indices].copy()
    y_train_actual_boot = y_full.loc[X_train_boot.index]  # the actual Diesel values
    
    # Now train classifier on this bootstrap
    clf_boot = RandomForestClassifier(n_estimators=100,
                                      random_state=None)  # no fixed random_state for variety
    clf_boot.fit(X_train_boot, y_train_binary_boot)
    
    # Identify non-zero subset within the bootstrap
    non_zero_mask_boot = (y_train_binary_boot == 1)
    X_train_reg_boot = X_train_boot[non_zero_mask_boot]
    y_train_reg_boot = y_train_actual_boot[non_zero_mask_boot]
    
    # -----------------------------
    # (b) Scale for regression
    # -----------------------------
    # Just as you do in baseline:
    scaler_X_boot = StandardScaler()
    # For safety, copy the subset
    X_train_reg_scaled_boot = X_train_reg_boot.copy()
    if 'Ambient Temperature' in X_train_reg_scaled_boot.columns:
        X_train_reg_scaled_boot[['Ambient Temperature']] = scaler_X_boot.fit_transform(
            X_train_reg_scaled_boot[['Ambient Temperature']]
        )
    
    scaler_y_boot = StandardScaler()
    y_train_reg_scaled_boot = scaler_y_boot.fit_transform(
        y_train_reg_boot.values.reshape(-1, 1)
    ).ravel()
    
    # Train regressor
    rf_reg_boot = RandomForestRegressor(n_estimators=100, random_state=None)
    rf_reg_boot.fit(X_train_reg_scaled_boot, y_train_reg_scaled_boot)
    
    # -----------------------------
    # (c) Predict on external test
    # -----------------------------
    # 1) Classification prediction
    test_binary_preds = clf_boot.predict(test_encoded)
    
    # 2) Regression only for predicted non-zero
    #    First scale test's 'Ambient Temperature' using the same scaler as training
    test_encoded_reg = test_encoded.copy()
    if 'Ambient Temperature' in test_encoded_reg.columns:
        test_encoded_reg[['Ambient Temperature']] = scaler_X_boot.transform(
            test_encoded_reg[['Ambient Temperature']]
        )
    
    # Regress only on rows predicted non-zero
    y_pred_reg_scaled_boot = rf_reg_boot.predict(test_encoded_reg[test_binary_preds == 1])
    y_pred_reg_actual_boot = scaler_y_boot.inverse_transform(
        y_pred_reg_scaled_boot.reshape(-1, 1)
    ).ravel()
    
    # Combine zero & non-zero
    test_preds_boot = np.zeros(len(test_encoded))
    test_preds_boot[test_binary_preds == 1] = y_pred_reg_actual_boot
    
    # -----------------------------
    # (d) Add noise to non-zero predictions (optional)
    # -----------------------------
    #   For example: normal(0, sigma), where sigma might be
    #   the residual std from the bootstrap training itself.
    #   Let's estimate training MSE on the bootstrap for an approximate noise level:
    if len(X_train_reg_boot) > 1:
        # Evaluate on the bootstrap's own training data for an approximate residual
        train_preds_scaled = rf_reg_boot.predict(X_train_reg_scaled_boot)
        train_preds = scaler_y_boot.inverse_transform(
            train_preds_scaled.reshape(-1,1)
        ).ravel()
        residuals = y_train_reg_boot.values - train_preds
        sigma_est = np.std(residuals)
    else:
        sigma_est = 0.0  # Edge case if there's almost no non-zero training data
    
    # Add random noise only to the non-zero predictions
    noise = rng.normal(loc=0.0, scale=sigma_est, size=len(test_preds_boot))
    test_preds_boot = np.where(test_binary_preds == 1,
                               test_preds_boot + noise,
                               0.0)
    
    # Store this realization
    bootstrap_preds[:, i] = test_preds_boot
    
    print(f"Finished bootstrap iteration {i+1}/{n_bootstrap}")

# ----------------------------------------------------------------------
# 4. Write the predictions back to existing_predictions
#    Now each column D1...D100 will have one bootstrap realization
# ----------------------------------------------------------------------
for i in range(n_bootstrap):
    col_name = f"D{i+1}"
    existing_predictions[col_name] = bootstrap_preds[:, i]

# (Optionally keep your original "Diesel" column from the single/baseline model)
# existing_predictions["Diesel"] = test_data["Diesel"]  # If you prefer to keep original

# Finally, save to CSV
existing_predictions.to_csv("test_uncert_predictions.csv", index=False)
print("\nWrote 100 bootstrap realizations (D1...D100) to 'test_uncert_predictions.csv'.")
