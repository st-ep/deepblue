import numpy as np                                      # model arrays afafafasfadsf
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

# Select top 5 original features based on actual feature importance results
top_features = [
    'Fleet Type_Zipper', 
    'EnergySum_scaled', 
    'Target Formation_Pecan Tree',
    'Target Formation_Lone Star',  # Changed from incorrect feature
    'Fuel Type_Turbine'           # Changed from incorrect feature
]
X = X[top_features]
X_missing = X_missing[top_features]

# Add engineered features
# Original interactions
X['Fleet_Energy_Interaction'] = X['Fleet Type_Zipper'] * X['EnergySum_scaled']
X['Formation_Energy_Interaction'] = X['Target Formation_Pecan Tree'] * X['EnergySum_scaled']
# New interactions
X['Formation_Lone_Star_Energy'] = X['Target Formation_Lone Star'] * X['EnergySum_scaled']
X['Fuel_Energy_Interaction'] = X['Fuel Type_Turbine'] * X['EnergySum_scaled']

# Add same features to X_missing
X_missing['Fleet_Energy_Interaction'] = X_missing['Fleet Type_Zipper'] * X_missing['EnergySum_scaled']
X_missing['Formation_Energy_Interaction'] = X_missing['Target Formation_Pecan Tree'] * X_missing['EnergySum_scaled']
X_missing['Formation_Lone_Star_Energy'] = X_missing['Target Formation_Lone Star'] * X_missing['EnergySum_scaled']
X_missing['Fuel_Energy_Interaction'] = X_missing['Fuel Type_Turbine'] * X_missing['EnergySum_scaled']

# Polynomial terms
X['EnergySum_squared'] = X['EnergySum_scaled'] ** 2
X['EnergySum_cubed'] = X['EnergySum_scaled'] ** 3
X_missing['EnergySum_squared'] = X_missing['EnergySum_scaled'] ** 2
X_missing['EnergySum_cubed'] = X_missing['EnergySum_scaled'] ** 3

# Add more complex interactions
X['Fleet_Energy_Squared'] = X['Fleet Type_Zipper'] * X['EnergySum_squared']
X['Formation_Energy_Squared'] = X['Target Formation_Pecan Tree'] * X['EnergySum_squared']
X_missing['Fleet_Energy_Squared'] = X_missing['Fleet Type_Zipper'] * X_missing['EnergySum_squared']
X_missing['Formation_Energy_Squared'] = X_missing['Target Formation_Pecan Tree'] * X_missing['EnergySum_squared']

# Scale all features, including categorical ones
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
'''
# Print summary of predictions
print("\nPredicted values statistics:")
print(pd.Series(missing_predictions).describe())

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_val, val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Random Forest)')
plt.tight_layout()
plt.show()
'''
# Add cross-validation evaluation
from sklearn.model_selection import KFold

# Create cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation scores
cv_scores = {
    'r2': [],
    'mae': [],
    'rmse': []
}

# Perform cross-validation
for train_idx, val_idx in kfold.split(X):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    best_rf.fit(X_train_cv, y_train_cv)
    
    # Make predictions
    val_pred_cv = best_rf.predict(X_val_cv)
    
    # Calculate metrics
    cv_scores['r2'].append(r2_score(y_val_cv, val_pred_cv))
    cv_scores['mae'].append(mean_absolute_error(y_val_cv, val_pred_cv))
    cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, val_pred_cv)))

print("\n=== Cross-Validation Results ===")
for metric, scores in cv_scores.items():
    print(f"{metric.upper()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Modify Random Forest to estimate prediction intervals
rf_quantile = RandomForestRegressor(
    **rf_random.best_params_,
    random_state=42,
    n_jobs=-1
)

rf_quantile.fit(X_train, y_train)

# Generate predictions with lower and upper bounds
def predict_with_intervals(model, X_pred):
    predictions = []
    for estimator in model.estimators_:
        predictions.append(estimator.predict(X_pred))
    predictions = np.array(predictions)
    
    mean_pred = predictions.mean(axis=0)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    
    return mean_pred, lower_bound, upper_bound

# Get predictions with confidence intervals for missing values
mean_pred, lower_bound, upper_bound = predict_with_intervals(rf_quantile, X_missing)

print("\n=== Prediction Intervals ===")
print(f"Average prediction interval width: {np.mean(upper_bound - lower_bound):.2f}")

# Get the original data again to preserve all columns
original_data = pd.read_csv("HackathonData2025+Fleet_Type_units_AT.csv")
original_data = original_data.iloc[:,1:]  # Remove first column as done before

# Update the missing values in the original dataset
original_data.loc[original_data['Estimated Average Stage Time'].isna(), 'Estimated Average Stage Time'] = missing_predictions

# Save the updated dataset
output_filename = "HackathonData2025+Fleet_Type_units_AT_updated.csv"
original_data.to_csv(output_filename, index=False)

print(f"\nUpdated dataset saved to: {output_filename}")
print(f"Number of values imputed: {len(missing_predictions)}")