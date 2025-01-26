import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.metrics import classification_report, r2_score, mean_squared_error, \
                            roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import SVR

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory

# ----------------------------------------------------------------------
# 1. Original Code: Data Preprocessing
# ----------------------------------------------------------------------
my_data = pd.read_csv("TrainData_recovered_forest_main.csv")
my_data = my_data.drop(['Actual Average Stage Time', 'Sand Provider '], axis=1, errors='ignore')

print("\nColumn Names and Types:")
print(my_data.dtypes)

categorical_columns = my_data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, prefix=categorical_columns)
print("\nShape after one-hot encoding:", encoded_data.shape)

# Target = 'Grid'
target_column = ['Grid']
y = encoded_data[target_column]  # shape (n_samples, 1)
X = encoded_data.drop(target_column, axis=1)

print("\nFeatures shape (X):", X.shape)
print("Target shape (y):", y.shape)
print("\nFeature names:")
print(X.columns.tolist())

# Scale target for inspection (though you also do a log transform later for regression)
scaler_y = StandardScaler()
y_scaled = pd.DataFrame(scaler_y.fit_transform(y), columns=y.columns)

print("\nOriginal target sample:")
print(y.head())
print("\nScaled target sample:")
print(y_scaled.head())

# Scale numeric feature(s)
numeric_features = ['Ambient Temperature']
scaler_X = StandardScaler()
X[numeric_features] = scaler_X.fit_transform(X[numeric_features])

print("\nScaled Ambient Temperature sample:")
print(X[numeric_features].head())

# Analyze distribution
print("\nGrid value distribution:")
print(y['Grid'].value_counts().sort_index())
print("\nGrid value statistics:")
print(y['Grid'].describe())

zero_percentage = (y['Grid'] == 0).mean() * 100
print(f"\nPercentage of zero values in Grid: {zero_percentage:.2f}%")

# ----------------------------------------------------------------------
# 2. Split #1: Binary Classification (0 vs. non-zero)
# ----------------------------------------------------------------------
y_binary = (y['Grid'] > 0).astype(int)
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train_binary, y_train_binary)

y_pred_binary = clf.predict(X_test_binary)
y_pred_proba = clf.predict_proba(X_test_binary)[:, 1]

print("\nBinary Classification Metrics:")
print("ROC-AUC Score:", roc_auc_score(y_test_binary, y_pred_proba))
print("Precision Score:", precision_score(y_test_binary, y_pred_binary))
print("Recall Score:", recall_score(y_test_binary, y_pred_binary))
print("F1 Score:", f1_score(y_test_binary, y_pred_binary))

# ----------------------------------------------------------------------
# 3. Split #2: Regression on non-zero portion (log-transformed)
# ----------------------------------------------------------------------
non_zero_mask = (y['Grid'] > 0)
X_non_zero = X[non_zero_mask]
y_non_zero = y[non_zero_mask]

# Log-transform y for regression
y_non_zero_log = np.log1p(y_non_zero)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_non_zero, y_non_zero_log, test_size=0.2, random_state=42
)

# You chose a RandomForestRegressor with these parameters:
reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=4,
    bootstrap=False,
    random_state=42
)
reg.fit(X_train_reg, y_train_reg.values.ravel())

# Evaluate regression on the hold-out
y_pred_log = reg.predict(X_test_reg)
y_pred_reg = np.expm1(y_pred_log)      # inverse of log1p
y_true = np.expm1(y_test_reg)

r2 = r2_score(y_true, y_pred_reg)
mse = mean_squared_error(y_true, y_pred_reg)
print("\nRandom Forest Regression Results (with log transform):")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")

# Feature importance
print("\nTop 5 important features for binary classification:")
binary_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False).head(5)
print(binary_importance)

print("\nTop 5 important features for regression:")
reg_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': reg.feature_importances_
}).sort_values('importance', ascending=False).head(5)
print(reg_importance)

# ----------------------------------------------------------------------
# 4. Combined Model Evaluation on classification test set
# ----------------------------------------------------------------------
binary_preds = clf.predict(X_test_binary)

final_predictions = np.zeros(len(X_test_binary))
non_zero_mask_test = (binary_preds == 1)
X_test_reg_subset = X_test_binary[non_zero_mask_test]

if len(X_test_reg_subset) > 0:
    reg_predictions_log = reg.predict(X_test_reg_subset)
    reg_predictions = np.expm1(reg_predictions_log)
    final_predictions[non_zero_mask_test] = reg_predictions

true_values = y.loc[X_test_binary.index]  # actual 'Grid'
overall_mse = mean_squared_error(true_values, final_predictions)
overall_r2 = r2_score(true_values, final_predictions)
print("\nCombined Model Evaluation on classification test set:")
print(f"Overall MSE: {overall_mse:.3f}")
print(f"Overall R² Score: {overall_r2:.3f}")

zero_accuracy = np.mean((true_values['Grid'] == 0) == (final_predictions == 0))
print(f"Zero Prediction Accuracy: {zero_accuracy:.3f}")

# ----------------------------------------------------------------------
# 5. Predictions on External Test Data
# ----------------------------------------------------------------------
print("\nReading and Processing Test Data:")
test_data = pd.read_csv("testing_with_filled_temperatures.csv")

# One-hot encode with same columns
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, prefix=categorical_columns)
missing_cols = set(X.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0
test_encoded = test_encoded[X.columns]

# Scale numeric features
test_encoded[numeric_features] = scaler_X.transform(test_encoded[numeric_features])

# Predict zero vs. non-zero
test_binary_preds = clf.predict(test_encoded)
test_final_predictions = np.zeros(len(test_encoded))

# For predicted non-zero, do regression
non_zero_mask_ext = (test_binary_preds == 1)
test_reg_subset = test_encoded[non_zero_mask_ext]

if len(test_reg_subset) > 0:
    reg_predictions_log = reg.predict(test_reg_subset)
    reg_predictions = np.expm1(reg_predictions_log)
    test_final_predictions[non_zero_mask_ext] = reg_predictions

test_data['Grid'] = test_final_predictions

print("\nTest Data Predictions Summary (baseline):")
print(test_data[['Grid']].describe())

# Add empty columns G1..G100 (we will overwrite them below)
for i in range(1, 101):
    test_data[f'G{i}'] = ''

test_data.to_csv("test_uncert_predictions.csv", index=False)
print("\nBaseline predictions + G1..G100 empty columns saved to 'test_uncert_predictions.csv'")
print("\nSample of predictions:")
print(test_data[['Grid'] + [f'G{i}' for i in range(1,6)]].head())

# ----------------------------------------------------------------------
# 6. [MODIFIED] Bootstrap Realizations: G1..G100 (Regression-only uncertainty)
# ----------------------------------------------------------------------
n_bootstrap = 100
rng = np.random.default_rng(seed=123)

# Convert your regression TRAIN sets (already log-transformed) to arrays
X_train_reg_arr = X_train_reg.to_numpy()         # shape = (n_train_reg, n_features)
y_train_reg_arr = y_train_reg.to_numpy().ravel() # shape = (n_train_reg,)

# We'll store the predictions for each bootstrap iteration here
bootstrap_preds = np.zeros((len(test_data), n_bootstrap))

# Get deterministic binary predictions once
test_binary_preds = clf.predict(test_encoded)

def inverse_log1p(pred_log):
    return np.expm1(pred_log)

# Only do regression bootstrapping where binary classifier predicted non-zero
idx_non_zero = np.where(test_binary_preds == 1)[0]
test_reg_subset = test_encoded.iloc[idx_non_zero]

for i in range(n_bootstrap):
    # Skip classification - use the original deterministic predictions
    test_preds_boot = np.zeros(len(test_encoded))
    
    if len(idx_non_zero) > 0:
        # Bootstrap only for regression
        n_train_reg = len(X_train_reg_arr)
        bootstrap_idx_reg = rng.integers(0, n_train_reg, size=n_train_reg)
        
        X_train_reg_boot = X_train_reg_arr[bootstrap_idx_reg, :]
        y_train_reg_boot = y_train_reg_arr[bootstrap_idx_reg]
        
        # Train a new RandomForestRegressor on the bootstrap sample
        reg_boot = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=4,
            bootstrap=False,
            random_state=None
        )
        reg_boot.fit(X_train_reg_boot, y_train_reg_boot)

        # Predict regression values
        test_reg_preds_log = reg_boot.predict(test_reg_subset)
        test_reg_preds = inverse_log1p(test_reg_preds_log)
        test_preds_boot[idx_non_zero] = test_reg_preds
    
    # Store
    bootstrap_preds[:, i] = test_preds_boot
    
    if (i + 1) % 10 == 0:
        print(f"Bootstrap iteration {i+1}/{n_bootstrap} complete.")

# ----------------------------------------------------------------------
# 7. Write the bootstrap predictions into G1..G100 columns
# ----------------------------------------------------------------------
df_out = pd.read_csv("test_uncert_predictions.csv")  # read what we wrote above
for i in range(n_bootstrap):
    df_out[f"G{i+1}"] = bootstrap_preds[:, i]

df_out.to_csv("test_uncert_predictions.csv", index=False)
print(f"\n[Bootstrap] Wrote {n_bootstrap} realizations (G1..G100) to 'test_uncert_predictions.csv'.")

print("Sample of the new columns:\n", df_out[[f"G{i}" for i in range(1,6)]].head())
