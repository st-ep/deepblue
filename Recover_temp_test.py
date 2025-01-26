import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for commented out plot
import seaborn as sns           # for commented out plot
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

# Create temperature bins with just 2 categories (Cold/Hot)
temp_bins = pd.qcut(my_data_encoded['Ambient Temperature'], q=2, labels=['Cold', 'Hot'])
print("\nTemperature range for each bin:")
print(pd.qcut(my_data_encoded['Ambient Temperature'], q=2, retbins=True)[1])

# Convert target to categorical
y = temp_bins

# Prepare features - just drop Ambient Temperature
X = my_data_encoded.drop('Ambient Temperature', axis=1)

# Import necessary modules
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

# Fit the grid search
print("\nPerforming grid search...")
grid_search.fit(X, y)

# Print results
print("\nBest parameters:", grid_search.best_params_)
print(f"Best accuracy score: {grid_search.best_score_:.3f}")

# Print classification report for best model
y_pred = grid_search.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': grid_search.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Calculate mean temperature for each bin
temp_bins_with_values = pd.qcut(my_data_encoded['Ambient Temperature'], q=2, labels=['Cold', 'Hot'])
bin_means = my_data_encoded.groupby(temp_bins_with_values)['Ambient Temperature'].mean()
print("\nMean temperature for each bin:")
print(bin_means)

# Load the test data
test_data = pd.read_csv("testing_with_predictions.csv")

# Prepare test data the same way as training data
# One-hot encode categorical columns
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

# Ensure test data has same columns as training data
for col in X.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0

# Keep only the columns that were used in training
X_test = test_encoded[X.columns]

# Make predictions
predictions = grid_search.predict(X_test)

# Convert categorical predictions to numeric using bin means
numeric_predictions = pd.Series(predictions).map(bin_means)

# Fill only the missing values with predictions
test_data.loc[test_data['Ambient Temperature'].isna(), 'Ambient Temperature'] = numeric_predictions[test_data['Ambient Temperature'].isna()]

# Save the results
test_data.to_csv('testing_with_filled_temperatures.csv', index=False)

print("\nPredictions completed and saved to 'testing_with_filled_temperatures.csv'")
print("Number of temperatures predicted:", sum(test_data['Ambient Temperature'].isna()))