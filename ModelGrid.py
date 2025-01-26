import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.metrics import classification_report, r2_score, mean_squared_error, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import SVR

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("TrainData_recovered_forest_main.csv")

# Drop specified columns (safely)
my_data = my_data.drop(['Actual Average Stage Time', 'Sand Provider '], axis=1, errors='ignore')

# Print column names and their types
print("\nColumn Names and Types:")
print(my_data.dtypes)

# Get list of categorical columns
categorical_columns = my_data.select_dtypes(include=['object']).columns

# Perform one-hot encoding
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, prefix=categorical_columns)

# Print the new shape to see how many features we have after encoding
print("\nShape after one-hot encoding:", encoded_data.shape)

# Separate target variable
target_column = ['Grid']
y = encoded_data[target_column]

# Create feature matrix by dropping target column
X = encoded_data.drop(target_column, axis=1)

# Print shapes to verify the split
print("\nFeatures shape (X):", X.shape)
print("Target shape (y):", y.shape)

# Optional: Print feature names
print("\nFeature names:")
print(X.columns.tolist())

# Scale target variable
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

# Analyze Grid distribution
print("\nGrid value distribution:")
print(y['Grid'].value_counts().sort_index())
print("\nGrid value statistics:")
print(y['Grid'].describe())

# Print percentage of zero values
zero_percentage = (y['Grid'] == 0).mean() * 100
print(f"\nPercentage of zero values in Grid: {zero_percentage:.2f}%")

# Create binary target for classification (0 vs non-zero)
y_binary = (y['Grid'] > 0).astype(int)

# Split data for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Train simple binary classifier (already perfect)
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train_binary, y_train_binary)

# Evaluate binary classifier
y_pred_binary = clf.predict(X_test_binary)
y_pred_proba = clf.predict_proba(X_test_binary)[:, 1]

print("\nBinary Classification Metrics:")
print("ROC-AUC Score:", roc_auc_score(y_test_binary, y_pred_proba))
print("Precision Score:", precision_score(y_test_binary, y_pred_binary))
print("Recall Score:", recall_score(y_test_binary, y_pred_binary))
print("F1 Score:", f1_score(y_test_binary, y_pred_binary))

# For regression on non-zero values
non_zero_mask = y['Grid'] > 0
X_non_zero = X[non_zero_mask]
y_non_zero = y[non_zero_mask]

# Log transform the target for regression
y_non_zero_log = np.log1p(y_non_zero)

# Split non-zero data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_non_zero, y_non_zero_log, test_size=0.2, random_state=42
)

# Use best parameters found from GridSearch
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

# Make predictions and transform back to original scale
y_pred_log = reg.predict(X_test_reg)
y_pred_reg = np.expm1(y_pred_log)
y_true = np.expm1(y_test_reg)

# Calculate metrics on original scale
r2 = r2_score(y_true, y_pred_reg)
mse = mean_squared_error(y_true, y_pred_reg)

print("\nRandom Forest Regression Results (with log transformation):")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")

# Print feature importance for both models
print("\nTop 5 important features for binary classification:")
binary_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False).head()
print(binary_importance)

print("\nTop 5 important features for regression:")
reg_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': reg.feature_importances_
}).sort_values('importance', ascending=False).head()
print(reg_importance)

estimators = [
    ('rf', RandomForestRegressor()),
    ('svr', SVR()),
    ('lasso', LassoCV())
]
stacking = StackingRegressor(estimators=estimators)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate combined predictions and scores
print("\nCombined Model Evaluation:")
# Get binary predictions for all test data
binary_preds = clf.predict(X_test_binary)

# Initialize array for final predictions
final_predictions = np.zeros(len(X_test_binary))

# For regression predictions, we need to ensure we're using the same indices
# Get the non-zero predictions from binary classifier
non_zero_mask = binary_preds == 1
X_test_reg_subset = X_test_binary[non_zero_mask]

# Get regression predictions for these cases
if len(X_test_reg_subset) > 0:
    reg_predictions = reg.predict(X_test_reg_subset)
    reg_predictions = np.expm1(reg_predictions)  # Transform back from log scale
    final_predictions[non_zero_mask] = reg_predictions

# Get true values for comparison
true_values = y.iloc[X_test_binary.index]

# Calculate overall metrics
overall_mse = mean_squared_error(true_values, final_predictions)
overall_r2 = r2_score(true_values, final_predictions)

print(f"Overall MSE: {overall_mse:.3f}")
print(f"Overall R² Score: {overall_r2:.3f}")

# Optional: Calculate percentage of correct zero predictions
zero_accuracy = np.mean((true_values['Grid'] == 0) == (final_predictions == 0))
print(f"Zero Prediction Accuracy: {zero_accuracy:.3f}")

# Read testing data
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

# Scale numeric features
test_encoded[numeric_features] = scaler_X.transform(test_encoded[numeric_features])

# Make predictions
test_binary_preds = clf.predict(test_encoded)
test_final_predictions = np.zeros(len(test_encoded))

# Get regression predictions for non-zero cases
non_zero_mask = test_binary_preds == 1
test_reg_subset = test_encoded[non_zero_mask]

if len(test_reg_subset) > 0:
    reg_predictions = reg.predict(test_reg_subset)
    reg_predictions = np.expm1(reg_predictions)  # Transform back from log scale
    test_final_predictions[non_zero_mask] = reg_predictions

# Add predictions to original test data using existing 'Grid' column
test_data['Grid'] = test_final_predictions

print("\nTest Data Predictions Summary:")
print(test_data[['Grid']].describe())

# Save predictions to CSV
test_data.to_csv("test_main_predictions.csv", index=False)
print("\nPredictions saved to 'test_main_predictions.csv'")
