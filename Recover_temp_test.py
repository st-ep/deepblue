import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for commented out plot
import seaborn as sns           # for commented out plot
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("TrainData_recovered_forest_main.csv")         # load the correct data file

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

# Drop Sand Provider column completely (note the space at the end)
my_data = my_data.drop('Sand Provider ', axis=1)  # Added space after 'Provider'

# Get more detailed information about the dataset
print("\nDetailed information about the dataset:")
print(my_data.info())

# One-hot encode remaining categorical columns
categorical_columns = ['Frac Fleet', 'Fleet Type', 'Target Formation', 
                      'Field Area', 'Fuel Type']

# Create one-hot encoded features
my_data_encoded = pd.get_dummies(my_data, columns=categorical_columns)

# Get numeric columns excluding 'Ambient Temperature'
numeric_columns = my_data_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('Ambient Temperature')  # Remove target variable
numeric_columns.remove('Estimated Average Stage Time')  # Remove stage time

# Initialize the scaler
scaler = StandardScaler()

# Normalize numeric columns
my_data_encoded[numeric_columns] = scaler.fit_transform(my_data_encoded[numeric_columns])

# Remove unwanted features
features_to_remove = ['Diesel', 'CNG', 'Estimated Average Stage Time']
my_data_encoded = my_data_encoded.drop(features_to_remove, axis=1)

# Print the shape of the final dataset
print("\nShape of dataset after normalization:", my_data_encoded.shape)
print("\nFirst few rows of normalized dataset:")
print(my_data_encoded.head())

# Remove the binning code and keep Ambient Temperature as continuous
y = my_data_encoded['Ambient Temperature']

# Prepare features - drop Ambient Temperature
X = my_data_encoded.drop('Ambient Temperature', axis=1)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base models
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Train models
print("Training Random Forest...")
rf_model.fit(X_train, y_train)
print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

# Make predictions on validation set
rf_pred = rf_model.predict(X_val)
xgb_pred = xgb_model.predict(X_val)

# Combine predictions with weighted average
weights = [0.5, 0.5]  # RF, XGB
ensemble_pred_val = weights[0] * rf_pred + weights[1] * xgb_pred

# Evaluate ensemble on validation set
print("\nValidation Set Performance:")
print(f"Random Forest R²: {r2_score(y_val, rf_pred):.3f}")
print(f"XGBoost R²: {r2_score(y_val, xgb_pred):.3f}")
print(f"Ensemble R²: {r2_score(y_val, ensemble_pred_val):.3f}")
print(f"Ensemble RMSE: {np.sqrt(mean_squared_error(y_val, ensemble_pred_val)):.3f}")

# Make predictions on full dataset
rf_pred_full = rf_model.predict(X)
xgb_pred_full = xgb_model.predict(X)

ensemble_pred = weights[0] * rf_pred_full + weights[1] * xgb_pred_full

# Load the test data
test_data = pd.read_csv("testing_with_predictions.csv")

# Prepare test data the same way as training data
# One-hot encode categorical columns
test_encoded = pd.get_dummies(test_data, columns=categorical_columns)

# Ensure test data has same columns as training data
for col in X.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0

# Keep only the columns that were used in training
X_test = test_encoded[X.columns]

# Make ensemble predictions on test data
rf_test_pred = rf_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)

ensemble_test_pred = (
    weights[0] * rf_test_pred + 
    weights[1] * xgb_test_pred
)

# Fill missing values with ensemble predictions
test_data.loc[test_data['Ambient Temperature'].isna(), 'Ambient Temperature'] = \
    ensemble_test_pred[test_data['Ambient Temperature'].isna()]

# Save results
test_data.to_csv('testing_with_filled_temperatures.csv', index=False)

print("\nPredictions completed and saved to 'testing_with_filled_temperatures.csv'")
print("Number of temperatures predicted:", sum(test_data['Ambient Temperature'].isna()))

# After making predictions, add these visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Remove the problematic style line and use a basic style
plt.style.use('default')

# 1. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y, ensemble_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature (Ensemble Model)')
plt.tight_layout()
plt.show()

# 2. Feature Importance Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15))
plt.title('Top 15 Most Important Features (Random Forest)')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# 3. Residuals Plot
residuals = y - ensemble_pred
plt.figure(figsize=(10, 6))
plt.scatter(ensemble_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Temperature')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Temperature')
plt.tight_layout()
plt.show()

# 4. Residuals Distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residual Value')
plt.ylabel('Count')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.show()

print("\nAll plots have been displayed")