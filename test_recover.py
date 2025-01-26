import numpy as np                                      # model arrays uuuafafafasfadsf
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import KFold

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

# Before preparing X and y, let's add feature importance analysis
# Initialize Random Forest for feature importance
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_selector.fit(X, y)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_selector.feature_importances_
})

# Sort features by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Print top features
print("\nTop 10 most important features:")
print(feature_importance.head(10))
'''
# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()
'''
# Define WeightedKNN class
class WeightedKNN(KNeighborsRegressor):
    def __init__(self, feature_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_weights = feature_weights

    def fit(self, X, y):
        if self.feature_weights is not None:
            self._X = X * self.feature_weights
        return super().fit(X, y)

    def predict(self, X):
        if self.feature_weights is not None:
            X = X * self.feature_weights
        return super().predict(X)

# Select top features based on actual feature importance results
top_features = [
    'Fleet Type_Zipper', 
    'Target Formation_Pecan Tree',
    'Target Formation_Lone Star',
    'Fuel Type_Turbine',
    'Target Formation_Longhorn',
    'Fuel Type_Diesel'    # Added back
]
X = X[top_features]
X_missing = X_missing[top_features]

# Scale all features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_missing = pd.DataFrame(scaler.transform(X_missing), columns=X_missing.columns)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Expand Random Forest parameter search
param_dist = {
    'n_estimators': randint(200, 800),      # Increased range
    'max_depth': [10, 15, 20, 25, 30],      # More options
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2'],# Added parameter
    'bootstrap': [True, False]               # Added parameter
}

# Initialize Random Forest with more trees
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,                              # Use all CPU cores
    oob_score=True                          # Enable out-of-bag score
)

# Increase number of search iterations
rf_random = RandomizedSearchCV(
    rf, 
    param_distributions=param_dist,
    n_iter=50,                              # Increased from 20
    cv=5,
    scoring='r2',                           # Changed scoring metric
    random_state=42,
    n_jobs=-1,
    verbose=1                               # Added progress reporting
)

# Fit the random search model
rf_random.fit(X_train, y_train)

# Get best model
best_rf = rf_random.best_estimator_

# Make predictions
val_pred = best_rf.predict(X_val)

# Calculate metrics
mae = mean_absolute_error(y_val, val_pred)
mse = mean_squared_error(y_val, val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_pred)
mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100

print("\n=== Random Forest Model Performance ===")
print(f"Best parameters: {rf_random.best_params_}")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Predict missing values
missing_predictions = best_rf.predict(X_missing)

# Add predictions back to original dataframe
encoded_data.loc[~mask_not_missing, 'Estimated Average Stage Time'] = missing_predictions

# After the Random Forest makes predictions for the training data missing values
# Create a time scaler using all available data
time_scaler = StandardScaler()
encoded_data['Estimated Average Stage Time'] = time_scaler.fit_transform(
    encoded_data[['Estimated Average Stage Time']]
)

# Load the testing data
testing_data = pd.read_csv("testing.csv")
testing_data = testing_data.iloc[:,1:]  # Remove first column if needed

# Fix column names in testing data by stripping whitespace
testing_data.columns = testing_data.columns.str.strip()

# Create mask for missing values in testing data
test_missing_mask = testing_data['Estimated Average Stage Time'].isna()

# Make sure we only use categorical columns that exist in testing data
categorical_columns = [col for col in categorical_columns if col.strip() in testing_data.columns]

# One-hot encode the testing data using the same categorical columns
testing_encoded = pd.get_dummies(testing_data, columns=categorical_columns, drop_first=True)

# Select the same features we used for training
X_test_missing = testing_encoded[test_missing_mask][top_features]

# Scale the features using the same scaler
X_test_missing = pd.DataFrame(scaler.transform(X_test_missing), columns=X_test_missing.columns)

# Make predictions only for missing values
test_predictions = best_rf.predict(X_test_missing)

# Scale the predictions using the same time_scaler
test_predictions_reshaped = test_predictions.reshape(-1, 1)
test_predictions_scaled = time_scaler.transform(test_predictions_reshaped).flatten()

# Add scaled predictions back to testing dataframe only where values were missing
testing_data.loc[test_missing_mask, 'Estimated Average Stage Time'] = test_predictions_scaled

# Scale all non-missing values in testing data
non_missing_mask = ~test_missing_mask
testing_data.loc[non_missing_mask, 'Estimated Average Stage Time'] = time_scaler.transform(
    testing_data.loc[non_missing_mask, ['Estimated Average Stage Time']]
)

# Save the results
testing_data.to_csv('testing_with_predictions.csv', index=False)

print("\nPredictions for missing values in testing data:")
print(pd.Series(test_predictions_scaled).describe())
print(f"\nNumber of predictions made: {len(test_predictions_scaled)}")
