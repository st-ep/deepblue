import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, classification_report

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory

# ----------------------------------------------------------------------
# 1. Baseline Code: read training data, drop columns, one-hot encode, etc.
# ----------------------------------------------------------------------
my_data = pd.read_csv("TrainData_recovered_forest_main.csv")

my_data = my_data.drop(['Actual Average Stage Time', 'Sand Provider '], axis=1, errors='ignore')

print("\nColumn Names and Types:")
print(my_data.dtypes)

categorical_columns = my_data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, prefix=categorical_columns)
print("\nShape after one-hot encoding:", encoded_data.shape)

target_column = 'CNG'  # We'll now predict CNG
y = encoded_data[target_column]

# Create binary labels (0 vs non-zero)
y_binary = (y > 0).astype(int)

# Create feature matrix by dropping target column
X = encoded_data.drop(target_column, axis=1)

# Split data for classification
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# We'll use ExtraTreesRegressor as in your code
best_reg = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

# Train the classifier (baseline)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_binary)

# Prepare data for regression (non-zero only)
X_train_reg = X_train[y_train_binary == 1]
y_train_reg = y.loc[X_train.index][y_train_binary == 1]  # actual CNG for those indices
X_test_reg = X_test[y_test_binary == 1]
y_test_reg = y.loc[X_test.index][y_test_binary == 1]

# Scale 'Ambient Temperature' for regression
scaler_X = StandardScaler()
X_train_reg_scaled = X_train_reg.copy()
X_test_reg_scaled = X_test_reg.copy()
if 'Ambient Temperature' in X_train_reg_scaled.columns:
    X_train_reg_scaled[['Ambient Temperature']] = scaler_X.fit_transform(
        X_train_reg_scaled[['Ambient Temperature']])
    X_test_reg_scaled[['Ambient Temperature']] = scaler_X.transform(
        X_test_reg_scaled[['Ambient Temperature']]
    )

# Scale target
scaler_y = StandardScaler()
y_train_reg_scaled = scaler_y.fit_transform(y_train_reg.values.reshape(-1, 1)).ravel()
y_test_reg_scaled = scaler_y.transform(y_test_reg.values.reshape(-1, 1)).ravel()

# Train the baseline regressor on non-zero data
best_reg.fit(X_train_reg_scaled, y_train_reg_scaled)

# Evaluate on test set
y_pred_binary = clf.predict(X_test)
X_test_scaled = X_test.copy()
if 'Ambient Temperature' in X_test_scaled.columns:
    X_test_scaled[['Ambient Temperature']] = scaler_X.transform(X_test_scaled[['Ambient Temperature']])

# Initialize final predictions with zeros
y_pred_final = np.zeros_like(y_test_binary, dtype=float)

# Regress for predicted non-zero
non_zero_pred_mask = (y_pred_binary == 1)
y_pred_reg_scaled = best_reg.predict(X_test_scaled[non_zero_pred_mask])
y_pred_reg_actual = scaler_y.inverse_transform(
    y_pred_reg_scaled.reshape(-1, 1)
).ravel()
y_pred_final[non_zero_pred_mask] = y_pred_reg_actual

# Calculate metrics
y_test_final = y.loc[X_test.index]
final_r2 = r2_score(y_test_final, y_pred_final)
final_mse = mean_squared_error(y_test_final, y_pred_final)
final_rmse = np.sqrt(final_mse)

print("\n[Baseline Model] Final Metrics:")
print(f"R² Score: {final_r2:.3f}")
print(f"MSE: {final_mse:.3f}")
print(f"RMSE: {final_rmse:.3f}")

# ----------------------------------------------------------------------
# 2. Read external test data (the 'unseen' dataset) and process
# ----------------------------------------------------------------------
print("\nReading and Processing External Test Data:")
test_data = pd.read_csv("testing_with_filled_temperatures.csv")

# One-hot encode
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, prefix=categorical_columns)

# Ensure columns match training set
missing_cols = set(X.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0
test_encoded = test_encoded[X.columns]  # reorder

# Scale 'Ambient Temperature' in external test data (based on baseline scaler_X)
if 'Ambient Temperature' in test_encoded.columns:
    test_encoded[['Ambient Temperature']] = scaler_X.transform(
        test_encoded[['Ambient Temperature']]
    )

# Make baseline predictions on external test data
test_binary_preds = clf.predict(test_encoded)
test_final_predictions = np.zeros(len(test_encoded))

non_zero_mask = (test_binary_preds == 1)
if np.any(non_zero_mask):
    y_pred_reg_scaled_ext = best_reg.predict(test_encoded[non_zero_mask])
    y_pred_reg_actual_ext = scaler_y.inverse_transform(
        y_pred_reg_scaled_ext.reshape(-1, 1)
    ).ravel()
    test_final_predictions[non_zero_mask] = y_pred_reg_actual_ext

test_data['CNG'] = test_final_predictions

# Read existing predictions CSV
existing_predictions = pd.read_csv("test_uncert_predictions.csv")
existing_predictions['CNG'] = test_data['CNG']

# Add empty C1..C100 columns (we'll overwrite them in the next step)
for i in range(1, 101):
    existing_predictions[f'C{i}'] = ''
    
# Save once so columns exist (optional)
existing_predictions.to_csv("test_uncert_predictions.csv", index=False)

print("\n[Baseline] Wrote 'CNG' and blank columns C1..C100 to 'test_uncert_predictions.csv'")
print("\nSample of predictions (baseline):")
print(existing_predictions[['Grid','CNG']+[f'C{i}' for i in range(1,6)]].head())

# ----------------------------------------------------------------------
# 3. Bootstrap Realizations: 100 columns C1...C100
# ----------------------------------------------------------------------
n_bootstrap = 100

# We'll store the 100 new predictions in a matrix
bootstrap_preds = np.zeros((len(test_encoded), n_bootstrap))

# For reproducibility, you can set a fixed seed or None
rng = np.random.default_rng(seed=123)

# Convert training sets to arrays so we can index easily
# (or keep them as DataFrames but must sample .iloc)
X_train_array = X_train.to_numpy()
y_train_binary_array = y_train_binary.to_numpy()
y_train_array = y.loc[X_train.index].to_numpy()  # actual CNG

n_train = len(X_train_array)

for i in range(n_bootstrap):
    # ------------------------------------------------
    # (a) Create a bootstrap sample from the training set
    # ------------------------------------------------
    # sample indices from [0, n_train)
    bootstrap_indices = rng.integers(low=0, high=n_train, size=n_train)
    
    X_train_boot = X_train_array[bootstrap_indices, :]
    y_train_binary_boot = y_train_binary_array[bootstrap_indices]
    y_train_actual_boot = y_train_array[bootstrap_indices]
    
    # Convert back to DataFrame for convenience
    X_train_boot_df = pd.DataFrame(X_train_boot, columns=X_train.columns)
    
    # ------------------------------------------------
    # (b) Train classifier on the bootstrap sample
    # ------------------------------------------------
    clf_boot = RandomForestClassifier(
        n_estimators=100,
        random_state=None  # no fixed seed => variation in each iteration
    )
    clf_boot.fit(X_train_boot_df, y_train_binary_boot)
    
    # ------------------------------------------------
    # (c) Identify non-zero subset & prepare for regression
    # ------------------------------------------------
    non_zero_mask_boot = (y_train_binary_boot == 1)
    X_train_reg_boot = X_train_boot_df[non_zero_mask_boot]
    y_train_reg_boot = y_train_actual_boot[non_zero_mask_boot]
    
    # Re-fit scalers for the bootstrapped subset
    scaler_X_boot = StandardScaler()
    X_train_reg_boot_scaled = X_train_reg_boot.copy()
    if 'Ambient Temperature' in X_train_reg_boot_scaled.columns:
        X_train_reg_boot_scaled[['Ambient Temperature']] = scaler_X_boot.fit_transform(
            X_train_reg_boot_scaled[['Ambient Temperature']]
        )
    
    scaler_y_boot = StandardScaler()
    y_train_reg_boot_scaled = scaler_y_boot.fit_transform(
        y_train_reg_boot.reshape(-1, 1)
    ).ravel()
    
    # ------------------------------------------------
    # (d) Train a new ExtraTreesRegressor on the bootstrap subset
    # ------------------------------------------------
    best_reg_boot = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=None  # vary each iteration
    )
    if len(X_train_reg_boot_scaled) > 0:
        best_reg_boot.fit(X_train_reg_boot_scaled, y_train_reg_boot_scaled)
    else:
        # Edge case: if there's no non-zero sample in the bootstrap,
        # the regressor is not trained. We'll handle it below.
        pass
    
    # ------------------------------------------------
    # (e) Predict on the external test set
    # ------------------------------------------------
    # 1) Classifier for zero vs. non-zero
    test_binary_preds_boot = clf_boot.predict(test_encoded)
    
    # 2) Regress only where predicted non-zero
    test_encoded_reg = test_encoded.copy()
    if 'Ambient Temperature' in test_encoded_reg.columns and len(X_train_reg_boot_scaled) > 0:
        test_encoded_reg[['Ambient Temperature']] = scaler_X_boot.transform(
            test_encoded_reg[['Ambient Temperature']]
        )
    
    # Initialize all predictions to zero
    test_preds_boot = np.zeros(len(test_encoded_reg))
    
    # Only if we have a trained regressor (i.e., some non-zero data in boot)
    if len(X_train_reg_boot_scaled) > 0:
        idx_non_zero = np.where(test_binary_preds_boot == 1)[0]
        if len(idx_non_zero) > 0:
            test_reg_scaled_preds = best_reg_boot.predict(test_encoded_reg.iloc[idx_non_zero])
            test_reg_preds = scaler_y_boot.inverse_transform(
                test_reg_scaled_preds.reshape(-1, 1)
            ).ravel()
            test_preds_boot[idx_non_zero] = test_reg_preds
    
    # ------------------------------------------------
    # (f) (Optional) Add noise to non-zero predictions
    # ------------------------------------------------
    # Estimate residual std from the bootstrap training set
    if len(X_train_reg_boot_scaled) > 1:
        train_preds_scaled = best_reg_boot.predict(X_train_reg_boot_scaled)
        train_preds = scaler_y_boot.inverse_transform(
            train_preds_scaled.reshape(-1,1)
        ).ravel()
        residuals = y_train_reg_boot - train_preds
        sigma_est = np.std(residuals)
    else:
        sigma_est = 0.0
    
    # Add noise only to the non-zero predictions
    noise = rng.normal(loc=0.0, scale=sigma_est, size=len(test_preds_boot))
    test_preds_boot = np.where(test_binary_preds_boot == 1,
                               test_preds_boot + noise,
                               0.0)
    
    # Store the resulting predictions in the matrix
    bootstrap_preds[:, i] = test_preds_boot
    
    print(f"Bootstrap iteration {i+1}/{n_bootstrap} complete.")

# ----------------------------------------------------------------------
# 4. Write the bootstrap predictions into columns C1..C100
#    in 'test_uncert_predictions.csv'
# ----------------------------------------------------------------------
existing_predictions = pd.read_csv("test_uncert_predictions.csv")

for i in range(n_bootstrap):
    col_name = f"C{i+1}"
    existing_predictions[col_name] = bootstrap_preds[:, i]

existing_predictions.to_csv("test_uncert_predictions.csv", index=False)

print("\n[Bootstrap] Wrote 100 realizations (C1..C100) to 'test_uncert_predictions.csv'.")
print("Sample of the new columns:\n", existing_predictions[[f"C{i}" for i in range(1,6)]].head())
