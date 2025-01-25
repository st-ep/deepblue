import numpy as np                                      # model arrays
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

# Create one-hot encoded variables
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, drop_first=True)

# Print the shape of the new dataset to see how many features we have after encoding
print("\nShape of dataset after one-hot encoding:", encoded_data.shape)

# Display first few columns of the encoded dataset
print("\nFirst few columns of the encoded dataset:")
print(encoded_data.head())

# Identify numeric columns (excluding the encoded categorical columns)
numeric_columns = ['Estimated Average Stage Time', 'Actual Average Stage Time', 
                  'Ambient Temperature', 'Grid', 'Diesel', 'CNG']

# Initialize the scaler
scaler = StandardScaler()

# Scale numeric columns
encoded_data[numeric_columns] = scaler.fit_transform(encoded_data[numeric_columns])

# Create a subset of data where Estimated Average Stage Time is not missing
complete_data = encoded_data.dropna(subset=['Estimated Average Stage Time'])

# Separate features and target
X = complete_data.drop(['Estimated Average Stage Time', 'Actual Average Stage Time'], axis=1)
y = complete_data['Estimated Average Stage Time']

# Train a Random Forest model (using Regressor instead of Classifier)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})

# Sort features by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Display top 15 most important features
print("\nTop 15 most important features for predicting Estimated Average Stage Time:")
print(feature_importance.head(15))

# Define different feature sets based on importance thresholds
feature_sets = {
    'top_5': [
        'Fleet Type_Zipper',
        'Diesel',
        'Target Formation_Pecan Tree',
        'Ambient Temperature',
        'CNG'
    ],
    'top_8': [
        'Fleet Type_Zipper',
        'Diesel',
        'Target Formation_Pecan Tree',
        'Ambient Temperature',
        'CNG',
        'Target Formation_Lone Star',
        'Frac Fleet_Fleet 9',
        'Field Area_The Tower'
    ],
    'top_12': [
        'Fleet Type_Zipper',
        'Diesel',
        'Target Formation_Pecan Tree',
        'Ambient Temperature',
        'CNG',
        'Target Formation_Lone Star',
        'Frac Fleet_Fleet 9',
        'Field Area_The Tower',
        'Target Formation_Longhorn',
        'Field Area_Gregory Gym',
        'Target Formation_Cowboy',
        'Field Area_Moody Center'
    ]
}

best_overall_score = -float('inf')
best_overall_model = None
best_feature_set = None

# Try each feature set
for feature_set_name, features in feature_sets.items():
    # Prepare data
    X = encoded_data[features]
    y = encoded_data['Estimated Average Stage Time']
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Define models and parameters as before
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elasticnet': ElasticNet()
    }
    
    param_grid = {
        'linear': {
            'polynomialfeatures__degree': [1, 2],
        },
        'ridge': {
            'polynomialfeatures__degree': [1, 2],
            'ridge__alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'lasso': {
            'polynomialfeatures__degree': [1, 2],
            'lasso__alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'elasticnet': {
            'polynomialfeatures__degree': [1, 2],
            'elasticnet__alpha': [0.01, 0.1, 1.0, 10.0],
            'elasticnet__l1_ratio': [0.1, 0.5, 0.9]
        }
    }
    
    # Find best model for this feature set
    best_score = -float('inf')
    best_model = None
    best_params = None
    best_name = None
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('polynomialfeatures', PolynomialFeatures()),
            (name, model)
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid[name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_name = name
    
    # Update overall best if this feature set performed better
    if best_score > best_overall_score:
        best_overall_score = best_score
        best_overall_model = best_model
        best_feature_set = feature_set_name
    
    # Print results for this feature set
    print(f"\nResults for {feature_set_name} ({len(features)} features):")
    print(f"Best Model: {best_name}")
    print(f"Best Parameters: {best_params}")
    print(f"R² Score: {best_score:.3f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error'))
    print(f"CV R² Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    print(f"CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")

print(f"\nBest Overall Feature Set: {best_feature_set}")
print(f"Best Overall R² Score: {best_overall_score:.3f}")

# Use the absolute best model configuration found
best_model = ElasticNet(alpha=0.01, l1_ratio=0.1)  # Changed l1_ratio to 0.1
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# Use top 12 features that gave best results
features = [
    'Fleet Type_Zipper',
    'Diesel',
    'Target Formation_Pecan Tree',
    'Ambient Temperature',
    'CNG',
    'Target Formation_Lone Star',
    'Frac Fleet_Fleet 9',
    'Field Area_The Tower',
    'Target Formation_Longhorn',
    'Field Area_Gregory Gym',
    'Target Formation_Cowboy',
    'Field Area_Moody Center'
]

# Prepare data
X = encoded_data[features]
y = encoded_data['Estimated Average Stage Time']
mask = ~y.isna()
X_train = X[mask]
y_train = y[mask]

# Transform features and fit model
X_train_poly = poly.fit_transform(X_train)
best_model.fit(X_train_poly, y_train)

# Predict missing values
missing_mask = encoded_data['Estimated Average Stage Time'].isna()
X_missing = encoded_data[missing_mask][features]
X_missing_poly = poly.transform(X_missing)
predicted_values = best_model.predict(X_missing_poly)

# Fill in the missing values
encoded_data.loc[missing_mask, 'Estimated Average Stage Time'] = predicted_values

print("\nFinal Statistics:")
print(f"Number of missing values filled: {len(predicted_values)}")
print(f"Range of predicted values: Min: {predicted_values.min():.2f}, Max: {predicted_values.max():.2f}")
print(f"Mean of predicted values: {predicted_values.mean():.2f}")

# Save the updated dataset
encoded_data.to_csv("updated_dataset_with_predictions.csv", index=False)
print("\nUpdated dataset saved to 'updated_dataset_with_predictions.csv'")
